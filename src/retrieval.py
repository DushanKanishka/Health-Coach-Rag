import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_kb_index(kb_index_path):
    kb = pd.read_parquet(kb_index_path)
    kb["embedding"] = kb["embedding"].apply(lambda x: np.array(x))
    return kb


def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


def retrieve_relevant_chunks(query: str, kb_index, model_emb, user_context: dict | None = None, top_k: int = 5):
    """
    Given a question and optional user context, return top_k most similar chunks
    from kb_index using cosine similarity over embeddings.
    """
    full_query = query
    if user_context is not None:
        ctx_parts = []
        age = user_context.get("age")
        bmi = user_context.get("bmi")

        if age is not None:
            ctx_parts.append(f"age {age}")
        if bmi is not None:
            ctx_parts.append(f"BMI {float(bmi):.1f}")

        avg_steps = user_context.get("avg_steps_7d")
        if avg_steps is not None:
            ctx_parts.append(f"avg steps last 7 days {avg_steps}")
        avg_sleep = user_context.get("avg_sleep_7d")
        if avg_sleep is not None:
            ctx_parts.append(f"avg sleep last 7 days {avg_sleep} hours")
        avg_stress = user_context.get("avg_stress_7d")
        if avg_stress is not None:
            ctx_parts.append(f"avg stress {avg_stress}/10")

        if ctx_parts:
            full_query = query + " | " + ", ".join(ctx_parts)

    # Encode query
    q_emb = model_emb.encode([full_query])

    # Stack embeddings
    chunk_embs = np.stack(kb_index["embedding"].values)

    # Cosine similarity + top-k
    sims = cosine_similarity(q_emb, chunk_embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    return kb_index.iloc[top_idx].copy(), sims[top_idx]
