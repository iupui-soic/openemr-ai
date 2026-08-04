"""
ChromaDB retrieval: keyword substring match + embedding similarity search
for CPT/ICD-10 candidate codes. Used by coding_service.py before calling
a model's generate_codes().
"""
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

SHORTLIST_TOP_K = 5
KEYWORD_MATCH_LIMIT = 20


def load_codes(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_or_load_collection(client, model, name, codes):
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        col = client.get_collection(name)
        if col.count() == len(codes):
            logger.info(f"Loaded '{name}' from ChromaDB ({col.count()} codes)")
            return col
        logger.info(f"Rebuilding '{name}'...")
        client.delete_collection(name)
    logger.info(f"Building '{name}' for {len(codes)} codes...")
    col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    code_list = list(codes.keys())
    descs = [codes[c] for c in code_list]
    BATCH = 256
    for i in range(0, len(code_list), BATCH):
        bc = code_list[i:i + BATCH]
        bd = descs[i:i + BATCH]
        embs = model.encode(bd, batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False).tolist()
        col.upsert(ids=bc, embeddings=embs, metadatas=[{"description": d, "code": c} for c, d in zip(bc, bd)], documents=bd)
        logger.info(f"  {min(i + BATCH, len(code_list))}/{len(code_list)}")
    logger.info(f"Built '{name}'")
    return col


def retrieve_candidates(model, col, terms: List[str], k: int = SHORTLIST_TOP_K) -> dict:
    """
    Two-pass retrieval:
      1. Keyword substring match (ChromaDB-side $contains filter).
      2. Embedding similarity search -- fills in candidates for terms that
         don't literally appear in any description (e.g. "headache" vs "R51").
    """
    if not terms:
        return {}

    candidates: dict = {}

    for term in terms:
        try:
            keyword_res = col.get(
                where_document={"$contains": term.lower()},
                include=["metadatas"],
                limit=KEYWORD_MATCH_LIMIT,
            )
        except Exception as e:
            logger.warning(f"Keyword match failed for '{term}': {e}")
            keyword_res = {"metadatas": []}
        for meta in keyword_res["metadatas"]:
            code = meta.get("code")
            if code and code not in candidates:
                candidates[code] = meta.get("description", "")

    for term in terms:
        query_emb = model.encode([term], normalize_embeddings=True).tolist()
        res = col.query(query_embeddings=query_emb, n_results=k, include=["metadatas"])
        for meta in res["metadatas"][0]:
            code = meta.get("code")
            if code and code not in candidates:
                candidates[code] = meta.get("description", "")

    return candidates
