"""Train and compare the isolated IM query transformation on manual indices."""

import argparse
import json
import math

from AI import create_query_vector, load_shared_components
from im_transform import IMTransform
from im_transform_examples import HELD_OUT_EXAMPLES, TRAIN_EXAMPLES
from manual_indices import index_vectors


DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.05


def evaluate(transform, examples, tokenizer, embedding, indices):
    rows = []
    losses = []
    correct = 0
    for query_text, label in examples:
        query_vector = create_query_vector(query_text, tokenizer, embedding)
        result = transform.loss_and_gradient(query_vector, label, indices)
        ranking = transform.rank(query_vector, indices)
        top_id = ranking[0]["collection_id"] if ranking else None
        correct += top_id == label
        losses.append(result["loss"])
        rows.append({
            "query": query_text,
            "label": label,
            "loss": result["loss"],
            "ranking": ranking,
            "correct_top1": top_id == label,
        })
    count = len(rows)
    return {
        "count": count,
        "mean_loss": math.fsum(losses) / count if count else None,
        "top1_correct": correct,
        "top1_accuracy": correct / count if count else None,
        "examples": rows,
    }


def run_experiment(epochs=DEFAULT_EPOCHS, learning_rate=DEFAULT_LEARNING_RATE):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be a finite positive number.")

    _, tokenizer, embedding = load_shared_components()
    indices = index_vectors()
    identity = IMTransform()
    trained = IMTransform()

    identity_results = {
        "train": evaluate(identity, TRAIN_EXAMPLES, tokenizer, embedding, indices),
        "held_out": evaluate(identity, HELD_OUT_EXAMPLES, tokenizer, embedding, indices),
    }

    epoch_losses = []
    for _ in range(epochs):
        losses = []
        for query_text, label in TRAIN_EXAMPLES:
            query_vector = create_query_vector(query_text, tokenizer, embedding)
            losses.append(
                trained.train_step(query_vector, label, indices, learning_rate)
            )
        epoch_losses.append(math.fsum(losses) / len(losses))

    return {
        "experiment": "IM linear query transformation; W is in-memory only",
        "dimension": IMTransform.DIMENSION,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "candidate_ids": sorted(indices),
        "identity_W": identity_results,
        "trained_W": {
            "training_mean_loss_before_training": identity_results["train"]["mean_loss"],
            "training_mean_loss_after_training": evaluate(
                trained, TRAIN_EXAMPLES, tokenizer, embedding, indices
            )["mean_loss"],
            "first_epoch_mean_step_loss": epoch_losses[0],
            "last_epoch_mean_step_loss": epoch_losses[-1],
            "train": evaluate(trained, TRAIN_EXAMPLES, tokenizer, embedding, indices),
            "held_out": evaluate(
                trained, HELD_OUT_EXAMPLES, tokenizer, embedding, indices
            ),
        },
        "interpretation": (
            "Training examples are the primary manual-index cases; held-out "
            "examples are the existing challenge cases. Results test this small "
            "label set only and do not establish generalization to generated indices."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.epochs, args.learning_rate), indent=2))


if __name__ == "__main__":
    main()
