"""
Rebuild the Fareez note-generation prompts exactly as in the first rating round.

Inputs, all unchanged from round one:
  - transcript: Section A of the released rating packet (identical to the Fareez clean transcript)
  - OpenEMR extract: data/fareez_openemr_extracts/{conversation}_openemr.txt
  - detected condition: data/fareez_selected_40.json
  - two retrieved schemas: all-MiniLM-L6-v2 cosine similarity against the diagnosis lists
    in the local Chroma store (k=2), as in models/run_fareez_local.py

The output JSON holds MIMIC-IV-derived schema outlines, so write it outside the repository.

Usage:
    python models/build_fareez_prompts.py --chroma /path/to/chroma_schema_improved --out /path/outside/repo/prompts.json
"""
import argparse, json, os, re, sys

RAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAG_ROOT, "models"))
from run_fareez_local import SUMMARY_PROMPT_TEMPLATE  # noqa: E402

SYSTEM = ("You are an expert medical scribe specialized in clinical documentation. "
          "Generate comprehensive SOAP-format medical summaries.")


def packet_transcripts():
    text = open(os.path.join(RAG_ROOT, "rating_packets", "packet_rater_1.txt"), encoding="utf-8").read()
    out = {}
    for item in re.split(r"#{80}\n# SUMMARY ", text)[1:]:
        conv = re.search(r"# Conversation: (\w+)", item).group(1)
        body = re.search(r"SECTION A: ORIGINAL TRANSCRIPT\n-+\n\n(.*?)\n\n-+\nSECTION B", item, re.S).group(1)
        assert out.setdefault(conv, body) == body
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer, util

    client = chromadb.PersistentClient(path=a.chroma)
    col = client.get_collection(client.list_collections()[0].name)
    data = col.get(include=["metadatas", "documents"])
    diseases = [m.get("diseases", "Unspecified") for m in data["metadatas"]]
    docs = data["documents"]
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    cand = sbert.encode(diseases, convert_to_tensor=True)

    selection = json.load(open(os.path.join(RAG_ROOT, "data", "fareez_selected_40.json")))
    transcripts = packet_transcripts()
    prompts = []
    for s in selection:
        conv, cond = s["file_name"], s["detected_condition"]
        assert len(transcripts[conv]) == s["char_count"], conv
        scores = util.cos_sim(sbert.encode(cond, convert_to_tensor=True), cand)[0]
        top = scores.topk(2).indices.tolist()
        schema = "".join(f"\n\n=== SCHEMA {r+1} ({diseases[i]}) ===\n{docs[i]}" for r, i in enumerate(top))
        openemr = open(os.path.join(RAG_ROOT, "data", "fareez_openemr_extracts", f"{conv}_openemr.txt"),
                       encoding="utf-8").read()
        user = SUMMARY_PROMPT_TEMPLATE.format(transcript=transcripts[conv],
                                              openemr=openemr or "No OpenEMR data available.", schema=schema)
        prompts.append({"conversation": conv, "detected_condition": cond, "system": SYSTEM, "user": user,
                        "top_similarity": round(float(scores[top[0]]), 4)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(prompts, open(a.out, "w"))
    print(f"wrote {len(prompts)} prompts to {a.out}")


if __name__ == "__main__":
    main()
