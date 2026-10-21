from aam import AAM
from attention import Attention
import argparse
import hashlib
import json
import math
from pathlib import Path

from tokenizer import (
    Tokenizer,
    EOS_TOKEN,
    initialize_tokens,
    load_dataset,
)
from model_io import (
    save_model,
    load_model,
    model_exists,
    load_tokenizer_state,
    load_shared_state,
    initialize_shared_state,
    shared_state_exists,
    next_collection_id,
    archive_checkpoint,
    reinitialize_shared_state,
    model_collection_ids,
)
from im import IM
from im_transform import IMTransform
from collection_index import (
    CHUNK_CENTROID,
    TOKEN_CENTROID,
    CollectionIndexStore,
)
from collection_retrieval import retrieve_and_load
from attention_training import TrainableAttentionModel
from attention_io import AttentionParameterStore
from vocabulary_corpus import (
    VOCABULARY_CORPUS_FILE,
    VocabularyCorpusStore,
)
from irm_io import IRM_PARAMETER_FILE, IRMParameterStore
from CPU_backend import CPUBackend
from apex_model import Apex_model
from embedding import Embedding
from network import Network
from trainer import Trainer
from RNN import RNN


DATASET_FILE = Path(__file__).resolve().with_name("dataset.txt")
EPOCHS = 6000
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 128
IRM_TRAINING_FILE = Path(__file__).resolve().with_name("irm_training.json")
IRM_DEFAULT_EPOCHS = 500
IRM_DEFAULT_LEARNING_RATE = 0.05


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train APEX using persistent parameter collections."
    )

    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "ID to train, resume, or load. Training without this option "
            "creates the next collection_N; inference defaults to collection_1."
        ),
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
        "--retrieve",
        action="store_true",
        help=(
            "Rank collection_indices.json and load the top state "
            "(requires --inference); uses trained IRM W when compatible, "
            "otherwise the identity baseline."
        ),
    )
    parser.add_argument(
        "--train-irm",
        action="store_true",
        help=(
            "Train the IRM transformation from explicit query-to-collection "
            "labels in irm_training.json."
        ),
    )
    parser.add_argument(
        "--irm-epochs",
        type=int,
        default=IRM_DEFAULT_EPOCHS,
        help="Number of relevance-training epochs used with --train-irm.",
    )
    parser.add_argument(
        "--irm-learning-rate",
        type=float,
        default=IRM_DEFAULT_LEARNING_RATE,
        help="IRM learning rate used with --train-irm.",
    )
    parser.add_argument(
        "--index-method",
        choices=(TOKEN_CENTROID, CHUNK_CENTROID),
        default=TOKEN_CENTROID,
        help="Content-derived index method to register after training.",
    )
    parser.add_argument(
        "--prepare-shared",
        action="store_true",
        help=(
            "Rebuild shared character-BPE from accumulated vocabulary sources "
            "and the current dataset. Archives the active model, attention, "
            "and IRM state, then clears all active collections."
        ),
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Train and exit without opening the prompt.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help=(
            "Compute backend: auto selects CUDA when available, cpu forces "
            "CPU, and gpu requires a working CUDA device."
        ),
    )

    args = parser.parse_args()

    if args.fresh and args.inference:
        parser.error("--fresh cannot be combined with --inference")

    if args.overwrite_collection and not args.fresh:
        parser.error("--overwrite-collection requires --fresh")

    if args.retrieve and not args.inference:
        parser.error("--retrieve requires --inference")

    if args.train_irm and (
        args.inference
        or args.retrieve
        or args.fresh
        or args.overwrite_collection
        or args.prepare_shared
        or args.collection is not None
    ):
        parser.error(
            "--train-irm cannot be combined with collection training, "
            "inference, retrieval, or --prepare-shared options"
        )

    if args.irm_epochs <= 0:
        parser.error("--irm-epochs must be a positive integer")
    if not math.isfinite(args.irm_learning_rate) or args.irm_learning_rate <= 0:
        parser.error("--irm-learning-rate must be a finite positive number")

    if args.prepare_shared and (
        args.inference or args.fresh or args.overwrite_collection or args.retrieve
    ):
        parser.error(
            "--prepare-shared cannot be combined with inference or fresh-training options"
        )

    return args


def build_model(vocab_size, backend, embedding):
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


def create_backend(backend_name):
    if backend_name not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unsupported compute backend: {backend_name!r}.")
    if backend_name == "cpu":
        return CPUBackend()

    try:
        from gpu_backend import GPUBackend
    except ModuleNotFoundError as error:
        if error.name not in {"numpy", "numba", "numba_cuda"}:
            raise
        if backend_name == "gpu":
            raise RuntimeError(
                "GPU backend dependencies are unavailable; install NumPy "
                "and Numba CUDA."
            ) from error
        print("CUDA backend dependencies are unavailable; using CPU.")
        return CPUBackend()

    if GPUBackend.is_available():
        return GPUBackend()
    if backend_name == "gpu":
        raise RuntimeError(
            "GPU backend was requested, but Numba CUDA reports no available "
            "CUDA device."
        )
    print("CUDA is unavailable; using CPU.")
    return CPUBackend()


def load_shared_components(trainable=False, backend_name="auto"):
    tokenizer_state = load_tokenizer_state()
    if tokenizer_state is None:
        raise RuntimeError("Shared tokenizer state is missing or invalid.")
    tokenizer = Tokenizer()
    tokenizer.load_state(tokenizer_state)
    backend = create_backend(backend_name)
    embedding = Embedding(
        tokenizer.size(),
        EMBEDDING_SIZE,
        backend,
        trainable=trainable,
    )
    load_shared_state(tokenizer, embedding)
    return backend, tokenizer, embedding


def initialize_shared_components(dataset, trainable=False, backend_name="auto"):
    tokenizer = train_tokenizer(dataset)
    backend = create_backend(backend_name)
    embedding = Embedding(
        tokenizer.size(),
        EMBEDDING_SIZE,
        backend,
        trainable=trainable,
    )
    initialize_shared_state(tokenizer, embedding)
    return backend, tokenizer, embedding


def dataset_from_source_content(source_content):
    lines = [line.strip() for line in source_content.splitlines() if line.strip()]
    if len(lines) % 2:
        raise ValueError(
            "Indexed collection source must contain prompt/answer line pairs."
        )
    return list(zip(lines[::2], lines[1::2]))


def prepare_shared_vocabulary():
    dataset = (
        load_dataset(DATASET_FILE)
        if DATASET_FILE.exists()
        else []
    )
    index_store = CollectionIndexStore()
    indexed_records = index_store.latest_records()
    combined_dataset = []
    seen_pairs = set()

    for record in indexed_records.values():
        for pair in dataset_from_source_content(record["source_content"]):
            if pair not in seen_pairs:
                combined_dataset.append(pair)
                seen_pairs.add(pair)
    for pair in dataset:
        if pair not in seen_pairs:
            combined_dataset.append(pair)
            seen_pairs.add(pair)

    combined_dataset = VocabularyCorpusStore(
        VOCABULARY_CORPUS_FILE
    ).add_pairs(combined_dataset)
    if not combined_dataset:
        raise RuntimeError(
            "Cannot prepare shared vocabulary: no saved, current dataset, or "
            "indexed collection sources were found."
        )

    tokenizer = train_tokenizer(combined_dataset)
    backend = CPUBackend()
    embedding = Embedding(
        tokenizer.size(),
        EMBEDDING_SIZE,
        backend,
        trainable=False,
    )
    checkpoint_backup = archive_checkpoint()
    reinitialize_shared_state(tokenizer, embedding)
    attention_backup = AttentionParameterStore().archive_and_clear()
    irm_backup = IRMParameterStore(IRM_PARAMETER_FILE).archive_and_clear()

    for collection_id, record in indexed_records.items():
        index_store.register(
            collection_id,
            record["source_content"],
            tokenizer,
            embedding,
            method=record["method"],
            chunk_size=record["config"].get("chunk_size", 128),
            allow_new_version=True,
        )
        print(f"Rebuilt compatible index for {collection_id}")

    if checkpoint_backup is not None:
        print("Archived previous checkpoint:", checkpoint_backup)
    if attention_backup is not None:
        print("Archived previous attention parameters:", attention_backup)
    if irm_backup is not None:
        print("Archived previous IRM parameters:", irm_backup)
    print(
        f"Prepared shared character-BPE vocabulary with {tokenizer.size()} tokens "
        f"using {len(combined_dataset)} accumulated prompt/answer pairs."
    )
    print("Accumulated vocabulary sources:", VOCABULARY_CORPUS_FILE)
    print(
        "All active collection parameters were archived and cleared. "
        "This reset happens on every --prepare-shared run; train every "
        "collection you need after the final migration."
    )


def collection_source_content(dataset):
    """Combine each training prompt and answer as the collection's index source."""
    return "\n".join(
        text
        for prompt, answer in dataset
        for text in (prompt, answer)
    )


def register_collection_index(collection_id, dataset, tokenizer, embedding, method):
    CollectionIndexStore().register(
        collection_id,
        collection_source_content(dataset),
        tokenizer,
        embedding,
        method=method,
        allow_new_version=True,
    )
    print(
        f"Registered {method} index for {collection_id} "
        "in collection_indices.json"
    )


def load_attention_for_training(collection_id, attention_store, model):
    try:
        attention_store.load(collection_id, model)
    except KeyError:
        print(
            f"No saved attention weights for {collection_id}; "
            "initializing attention from scratch and resuming the saved "
            "PNSN parameters."
        )
        return False
    return True


def create_query_vector(prompt, tokenizer, embedding):
    # Step 1: Convert user text into token IDs.
    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise ValueError("The prompt contains no tokens.")

    # Step 2: Look up the embedding vector for each token.
    token_vectors = []
    embedding_rows = (
        embedding.embedding_matrix.to_cpu()
        if hasattr(embedding.embedding_matrix, "to_cpu")
        else embedding.embedding_matrix.data
    )

    for token_id in token_ids:
        vector = embedding_rows[token_id]
        token_vectors.append(vector)

    # Step 3: Mean-pool token vectors using AAM.
    query_vector = AAM.mean_pool(token_vectors)

    return query_vector


def train_irm_supervision(
    training_file=IRM_TRAINING_FILE,
    epochs=IRM_DEFAULT_EPOCHS,
    learning_rate=IRM_DEFAULT_LEARNING_RATE,
):
    """Train W from explicit labels against the active collection candidates."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("IRM epochs must be a positive integer.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("IRM learning rate must be a finite positive number.")

    training_file = Path(training_file)
    if not training_file.is_file():
        raise FileNotFoundError(
            f"IRM training labels were not found at {training_file}. "
            "Create a JSON array of query labels before running --train-irm. "
            "Use collection_id for one relevant collection or collection_ids "
            "for multiple relevant collections."
        )
    with training_file.open("r", encoding="utf-8") as file:
        examples = json.load(file)
    if not isinstance(examples, list) or not examples:
        raise ValueError(
            f"{training_file} must contain a non-empty JSON array of labeled queries."
        )

    backend, tokenizer, embedding = load_shared_components(backend_name="cpu")
    index_store = CollectionIndexStore()
    all_records = index_store.latest_records()
    eligible_ids = (
        model_collection_ids()
        & AttentionParameterStore().collection_ids()
        & set(all_records)
    )
    candidate_records = {
        collection_id: all_records[collection_id]
        for collection_id in sorted(eligible_ids)
    }
    if len(candidate_records) < 2:
        raise ValueError(
            "IRM training needs at least two collections with a compatible "
            "index, saved model, and trained attention state."
        )

    from collection_index import require_compatible_index
    from supervised_retrieval import (
        prepare_labeled_queries,
        train_relevance_epoch,
    )

    for collection_id, record in candidate_records.items():
        require_compatible_index(record, tokenizer, embedding)
    candidate_indices = {
        collection_id: list(record["vector"])
        for collection_id, record in candidate_records.items()
    }
    unknown_id = tokenizer.token_to_id_map["<UNK>"]
    for position, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"IRM training example {position} must be an object.")
        query = example.get("query")
        if isinstance(query, str):
            unknown_count = tokenizer.encode(query).count(unknown_id)
            if unknown_count:
                raise ValueError(
                    f"IRM training query {position} contains {unknown_count} "
                    "out-of-vocabulary tokens; run --prepare-shared with these "
                    "queries included first."
                )

    prepared = prepare_labeled_queries(
        examples,
        tokenizer,
        embedding,
        candidate_indices,
    )
    transform = IMTransform(backend=backend)
    initial_loss = math.fsum(
        transform.loss_and_gradient(
            query_vector, collection_ids, candidate_indices
        )["loss"]
        for query_vector, collection_ids in prepared
    ) / len(prepared)
    report_interval = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        epoch_loss = train_relevance_epoch(
            transform,
            prepared,
            candidate_indices,
            learning_rate,
        )
        if epoch == 1 or epoch % report_interval == 0 or epoch == epochs:
            print(
                f"IRM epoch {epoch}/{epochs} | "
                f"Mean training loss: {epoch_loss:.6f}"
            )
    final_loss = math.fsum(
        transform.loss_and_gradient(
            query_vector, collection_ids, candidate_indices
        )["loss"]
        for query_vector, collection_ids in prepared
    ) / len(prepared)
    source_hash = hashlib.sha256(
        training_file.read_bytes()
    ).hexdigest()
    IRMParameterStore(IRM_PARAMETER_FILE).save(
        transform,
        tokenizer,
        embedding,
        candidate_records,
        {
            "examples": len(prepared),
            "epochs": epochs,
            "learning_rate": float(learning_rate),
            "initial_mean_loss": initial_loss,
            "final_mean_loss": final_loss,
            "training_file_sha256": source_hash,
        },
    )
    print(
        f"Saved trained IRM parameters for {len(candidate_records)} candidate "
        f"collections to {IRM_PARAMETER_FILE}"
    )
    print(f"Final mean labeled-query loss: {final_loss:.6f}")


def rank_collections(prompt, tokenizer, embedding, collection_indices):
    """Run the existing tokenizer/embedding/AAM path, then rank indices."""
    query_vector = create_query_vector(prompt, tokenizer, embedding)
    return IM().rank(query_vector, collection_indices)


def attention_context(queries, keys, values, mask=None, backend=None):
    """Run general-purpose attention, independent of AAM and PNSN retrieval.

    This optional path accepts explicit query/key/value sequences and returns
    ``(context, weights)``. Existing query-to-collection calls do not invoke it.
    """
    mechanism = Attention(backend)
    return mechanism.forward(queries, keys, values, mask=mask)

def train_tokenizer(dataset):
    if not dataset:
        raise ValueError("Tokenizer training requires at least one text pair.")
    texts = [
        text
        for input_text, target_text in dataset
        for text in (input_text, target_text)
    ]
    texts.append("\n")
    tokenizer = Tokenizer()
    tokenizer.train(initialize_tokens(texts), num_merges=10)
    return tokenizer


def encode_dataset(dataset, tokenizer):
    encoded_dataset = []

    for input_text, target_text in dataset:
        input_ids = tokenizer.encode(input_text)
        target_ids = tokenizer.encode(target_text)

        if not input_ids or not target_ids:
            continue

        training_inputs = input_ids + target_ids
        training_targets = (
            [None] * (len(input_ids) - 1)
            + target_ids
            + [tokenizer.token_to_id_map[EOS_TOKEN]]
        )

        encoded_dataset.append(
            (training_inputs, training_targets)
        )

    return encoded_dataset


def validate_dataset_vocabulary(dataset, tokenizer):
    if EOS_TOKEN not in tokenizer.token_to_id_map:
        raise ValueError(
            "Shared tokenizer has no end-of-answer token. "
            "Run --prepare-shared before retraining collections."
        )
    unknown_id = tokenizer.token_to_id_map["<UNK>"]
    unknown_count = sum(
        token_id == unknown_id
        for prompt, answer in dataset
        for token_id in tokenizer.encode(prompt) + tokenizer.encode(answer)
    )
    if unknown_count:
        raise ValueError(
            f"Training data contains {unknown_count} out-of-vocabulary tokens. "
            "Run --prepare-shared with all collection data before retraining."
        )


def response_length_limit(prompt, dataset, tokenizer, index_store, collection_id):
    source_dataset = []
    try:
        source_dataset = dataset_from_source_content(
            index_store.latest(collection_id)["source_content"]
        )
    except KeyError:
        pass
    for input_text, target_text in source_dataset + dataset:
        if input_text == prompt:
            return min(64, len(tokenizer.encode(target_text)))
    known_targets = [
        len(tokenizer.encode(target_text))
        for _, target_text in source_dataset + dataset
    ]
    return min(64, max(known_targets, default=30))


def main():
    args = parse_arguments()
    if args.prepare_shared:
        prepare_shared_vocabulary()
        return
    if args.train_irm:
        train_irm_supervision(
            IRM_TRAINING_FILE,
            args.irm_epochs,
            args.irm_learning_rate,
        )
        return
    collection_id = resolve_collection_id(args)

    if not args.inference and args.collection is None:
        print("Creating new collection:", collection_id)

    if args.inference:
        dataset = (
            load_dataset(DATASET_FILE)
            if DATASET_FILE.exists()
            else []
        )
        backend, tokenizer, embedding = load_shared_components(
            backend_name=args.backend
        )
        _, embedding, rnn, network, base_model = build_model(
            tokenizer.size(), backend, embedding
        )
        model = TrainableAttentionModel(base_model)
        attention_store = AttentionParameterStore()
        index_store = CollectionIndexStore()
        print("Compute backend:", backend.name)

        if not args.retrieve:
            load_model(
                rnn,
                network,
                collection_id,
            )
            load_attention_for_training(collection_id, attention_store, model)

        while True:
            prompt = input("Enter a prompt (or type 'exit'): ")

            if prompt.casefold() == "exit":
                break
            query_vector = create_query_vector(
                prompt,
                tokenizer,
                embedding,
            )
            query_token_ids = tokenizer.encode(prompt)
            unknown_count = sum(
                token_id == tokenizer.token_to_id_map["<UNK>"]
                for token_id in query_token_ids
            )
            if unknown_count:
                print(
                    f"Warning: {unknown_count}/{len(query_token_ids)} query "
                    "tokens map to <UNK>."
                )

            if args.retrieve:
                available_ids = (
                    model_collection_ids()
                    & attention_store.collection_ids()
                )
                latest_records = index_store.latest_records()
                indexed_ids = set(latest_records)
                active_indexed_ids = available_ids & indexed_ids
                if not active_indexed_ids:
                    raise ValueError(
                        "No indexed collection has both a valid model checkpoint "
                        "and trained attention parameters."
                    )
                unavailable_ids = sorted(indexed_ids - available_ids)
                if unavailable_ids:
                    print(
                        "Skipping indexed collections without both model and "
                        "trained attention checkpoints:",
                        ", ".join(unavailable_ids),
                    )
                transform = IMTransform(backend=CPUBackend())
                has_trained_irm = IRMParameterStore(IRM_PARAMETER_FILE).load(
                    transform,
                    tokenizer,
                    embedding,
                    {
                        candidate_id: latest_records[candidate_id]
                        for candidate_id in sorted(active_indexed_ids)
                    },
                )
                if has_trained_irm:
                    print("Loaded trained IRM transformation.")
                else:
                    print(
                        "No trained IRM parameters found; using identity W. "
                        "Train it with --train-irm."
                    )
                result = retrieve_and_load(
                    query_vector,
                    tokenizer,
                    embedding,
                    index_store,
                    rnn,
                    network,
                    transform=transform,
                    eligible_collection_ids=available_ids,
                )
                print("Retrieved collection:", result["collection_id"])
                print("Retrieval score:", result["ranking"][0]["score"])
                print("Collection ranking:", result["ranking"])
                attention_store.load(result["collection_id"], model)
                generation_collection_id = result["collection_id"]
            else:
                print("Query vector:", query_vector)
                generation_collection_id = collection_id
            print(
                "Generated:",
                model.generate(
                    prompt,
                    tokenizer,
                    max_length=response_length_limit(
                        prompt,
                        dataset,
                        tokenizer,
                        index_store,
                        generation_collection_id,
                    ),
                    end_token_id=tokenizer.token_to_id_map.get(EOS_TOKEN),
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

    if shared_state_exists():
        backend, tokenizer, embedding = load_shared_components(
            trainable=False,
            backend_name=args.backend,
        )
    else:
        backend, tokenizer, embedding = initialize_shared_components(
            dataset,
            trainable=False,
            backend_name=args.backend,
        )

    vocab_size = tokenizer.size()
    validate_dataset_vocabulary(dataset, tokenizer)

    print("Collection:", collection_id)
    print("Vocabulary size:", vocab_size)
    print("Compute backend:", backend.name)

    encoded_dataset = encode_dataset(dataset, tokenizer)

    if not encoded_dataset:
        raise RuntimeError(
            "No usable training examples were produced."
        )

    _, embedding, rnn, network, base_model = build_model(
        vocab_size, backend, embedding
    )
    model = TrainableAttentionModel(base_model)
    attention_store = AttentionParameterStore()

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
            collection_id,
        )
        attention_store.load(collection_id, model)

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
                allow_update=True,
            )
            attention_store.save(collection_id, model)
            register_collection_index(
                collection_id,
                dataset,
                tokenizer,
                embedding,
                args.index_method,
            )

    if not args.no_chat:
        index_store = CollectionIndexStore()
        while True:
            prompt = input("Enter a prompt (or type 'exit'): ")

            print(
                "Generated:",
                model.generate(
                    prompt,
                    tokenizer,
                    max_length=response_length_limit(
                        prompt,
                        dataset,
                        tokenizer,
                        index_store,
                        collection_id,
                    ),
                    end_token_id=tokenizer.token_to_id_map.get(EOS_TOKEN),
                ),
            )


def resolve_collection_id(args):
    if args.collection is not None:
        return args.collection
    if args.inference:
        return "collection_1"
    return next_collection_id()


if __name__ == "__main__":
    main()
