"""Rank version-compatible collection indices and load the matching model state."""

from im import IM
from model_io import load_model


def retrieve_and_load(
    query_vector,
    tokenizer,
    embedding,
    index_store,
    rnn,
    network,
    transform=None,
    eligible_collection_ids=None,
):
    """Retrieve the top collection ID, then load its saved RNN/network state."""
    collection_indices = index_store.latest_vectors(tokenizer, embedding)
    if not collection_indices:
        raise ValueError("No compatible collection indices are registered.")
    if eligible_collection_ids is not None:
        eligible_collection_ids = set(eligible_collection_ids)
        collection_indices = {
            collection_id: vector
            for collection_id, vector in collection_indices.items()
            if collection_id in eligible_collection_ids
        }
        if not collection_indices:
            raise ValueError(
                "No indexed collection has both a valid model checkpoint and "
                "trained attention parameters."
            )

    retrieval_query = (
        transform.transformed_vector(query_vector)
        if transform is not None
        else query_vector
    )
    ranking = IM().rank(retrieval_query, collection_indices)
    if not ranking:
        raise ValueError("No collection could be retrieved.")

    collection_id = ranking[0]["collection_id"]
    epoch = load_model(rnn, network, collection_id)
    return {
        "collection_id": collection_id,
        "epoch": epoch,
        "ranking": ranking,
    }
