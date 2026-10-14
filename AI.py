from aam import AAM
import argparse
from pathlib import Path

from tokenizer import (
    Tokenizer,
    load_dataset,
    initialize_tokens,
    target_length_for_prompt,
)
from model_io import (
    save_model,
    load_model,
    model_exists,
    load_tokenizer_state,
)
from CPU_backend import CPUBackend
from apex_model import Apex_model
from embedding import Embedding
from network import Network
from trainer import Trainer
from RNN import RNN


DATASET_FILE = Path(__file__).resolve().with_name("dataset.txt")
EPOCHS = 600
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 128


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train APEX using persistent parameter collections."
    )

    parser.add_argument(
        "--collection",
        default="collection_1",
        help="ID of the collection to train or load.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Initialize fresh weights for the selected collection.",
    )
    parser.add_argument(
        "--overwrite-collection",
        action="store_true",
        help="Allow fresh training to replace an existing collection when saved.",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Load the selected collection without training.",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Train and exit without opening the prompt.",
    )

    args = parser.parse_args()

    if args.fresh and args.inference:
        parser.error("--fresh cannot be combined with --inference")

    if args.overwrite_collection and not args.fresh:
        parser.error("--overwrite-collection requires --fresh")

    return args


def build_model(vocab_size):
    backend = CPUBackend()

    embedding = Embedding(
        vocab_size,
        EMBEDDING_SIZE,
        backend,
    )

    rnn = RNN(
        EMBEDDING_SIZE,
        HIDDEN_SIZE,
        vocab_size,
        backend,
    )

    network = Network(
        [HIDDEN_SIZE, 64, vocab_size],
        backend,
    )

    model = Apex_model(rnn, network, embedding)

    return backend, embedding, rnn, network, model

def create_query_vector(prompt, tokenizer, embedding):
    # Step 1: Convert user text into token IDs.
    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("The prompt contains no tokens.")

    # Step 2: Look up the embedding vector for each token.
    token_vectors = []

    for token_id in token_ids:
        vector = embedding.embedding_matrix.data[token_id]
        token_vectors.append(vector)

    # Step 3: Mean-pool token vectors using AAM.
    query_vector = AAM.mean_pool(token_vectors)

    return query_vector

def train_tokenizer(dataset):
    texts = []

    for input_text, target_text in dataset:
        texts.append(input_text)
        texts.append(target_text)

    tokenized_texts = initialize_tokens(texts)

    tokenizer = Tokenizer()
    tokenizer.train(tokenized_texts, 10)

    return tokenizer


def encode_dataset(dataset, tokenizer):
    encoded_dataset = []

    for input_text, target_text in dataset:
        input_ids = tokenizer.encode(input_text)
        target_ids = tokenizer.encode(target_text)

        if not input_ids or not target_ids:
            continue

        training_inputs = input_ids + target_ids[:-1]
        training_targets = (
            [None] * (len(input_ids) - 1) + target_ids
        )

        encoded_dataset.append(
            (training_inputs, training_targets)
        )

    return encoded_dataset


def main():
    args = parse_arguments()
    collection_id = args.collection

    if args.inference:
        dataset = (
            load_dataset(DATASET_FILE)
            if DATASET_FILE.exists()
            else []
        )
        tokenizer_state = load_tokenizer_state(collection_id)

        if tokenizer_state is None:
            raise RuntimeError(
                f"No saved tokenizer found for {collection_id!r}."
            )

        tokenizer = Tokenizer()
        tokenizer.load_state(tokenizer_state)

        _, embedding, rnn, network, model = build_model(
            tokenizer.size()
        )

        load_model(
            rnn,
            network,
            embedding,
            collection_id,
            tokenizer,
        )

        while True:
            prompt = input("Enter a prompt (or type 'exit'): ")

            if prompt.casefold() == "exit":
                break
            query_vector = create_query_vector(
                prompt,
                tokenizer,
                embedding,
            )

            print("Query vector:", query_vector)
            print(
                "Generated:",
                model.generate(
                    prompt,
                    tokenizer,
                    max_length=target_length_for_prompt(
                        prompt,
                        dataset,
                        tokenizer,
                    ),
                ),
            )

        return

    if args.fresh and model_exists(collection_id):
        if not args.overwrite_collection:
            raise RuntimeError(
                f"{collection_id!r} already exists. "
                "Choose another collection ID or explicitly pass "
                "--overwrite-collection to replace it."
            )

    dataset = load_dataset(DATASET_FILE)

    if not dataset:
        raise RuntimeError("The dataset is empty.")

    # Restore the tokenizer when resuming a collection.
    # A fresh collection receives a tokenizer trained for this dataset.
    if not args.fresh and model_exists(collection_id):
        tokenizer_state = load_tokenizer_state(collection_id)

        if tokenizer_state is None:
            raise RuntimeError(
                f"Cannot resume {collection_id!r}: "
                "its saved tokenizer is missing."
            )

        tokenizer = Tokenizer()
        tokenizer.load_state(tokenizer_state)

    else:
        tokenizer = train_tokenizer(dataset)

    vocab_size = tokenizer.size()

    print("Collection:", collection_id)
    print("Vocabulary size:", vocab_size)

    encoded_dataset = encode_dataset(dataset, tokenizer)

    if not encoded_dataset:
        raise RuntimeError(
            "No usable training examples were produced."
        )

    _, embedding, rnn, network, model = build_model(vocab_size)

    trainer = Trainer(
        model,
        learning_rate=0.01,
    )

    start_epoch = 0

    if not args.fresh and model_exists(collection_id):
        print("Loading saved collection...")

        start_epoch = load_model(
            rnn,
            network,
            embedding,
            collection_id,
            tokenizer,
        )

        print("Resuming from epoch:", start_epoch)

    for epoch in range(start_epoch, EPOCHS):
        epoch_losses = []

        for inputs, targets in encoded_dataset:
            epoch_losses.extend(
                trainer.train_step(inputs, targets)
            )

        if not epoch_losses:
            raise RuntimeError(
                "No supervised targets were produced."
            )

        completed_epoch = epoch + 1

        if epoch == start_epoch or completed_epoch % 100 == 0:
            average_loss = sum(epoch_losses) / len(epoch_losses)

            print(
                f"Collection {collection_id} | "
                f"Epoch {completed_epoch}/{EPOCHS} | "
                f"Loss: {average_loss:.6f}"
            )

        if completed_epoch % 100 == 0 or completed_epoch == EPOCHS:
            save_model(
                rnn,
                network,
                embedding,
                completed_epoch,
                collection_id,
                tokenizer,
                allow_update=True,
            )

    if not args.no_chat:
        while True:
            prompt = input("Enter a prompt (or type 'exit'): ")

            print(
                "Generated:",
                model.generate(
                    prompt,
                    tokenizer,
                    max_length=target_length_for_prompt(
                        prompt,
                        dataset,
                        tokenizer,
                    ),
                ),
            )


if __name__ == "__main__":
    main()
