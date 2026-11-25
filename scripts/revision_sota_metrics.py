"""
Upgraded locally runnable automated metrics for all 239 rated Fareez notes.

Addresses three limitations of default transcript-referenced metrics:
1. Silent truncation: computes exact untruncated full-text BERTScore (roberta-large, layer 17)
   via sliding-block token encoding, plus full-text chunk-pooled embeddings from all-MiniLM-L6-v2,
   BAAI/bge-large-en-v1.5, and biomedical pritamdeka/S-PubMedBert-MS-MARCO. Also logs exact
   token-level truncation rates under default 256-token (SBERT) and 510-token (BERTScore) limits.
2. Semantic domain mismatch: compares general-domain vs biomedical embeddings (S-PubMedBert-MS-MARCO).
3. Factual consistency / faithfulness: runs RAG-retrieved sentence-level natural language inference
   (NLI) with MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli against (a) the source
   transcript alone and (b) the combined transcript + paired OpenEMR chart extract.

Runs 100% locally on a single GPU using cached Hugging Face weights (no external API calls).
Output: rag_models/results/revision/fareez_sota_metrics.csv
"""
import argparse
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODELS_ALL = ["qwen3.8-27b", "gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b", "llama-4-scout"]
FAILED = {("MSK0037", "medgemma-4b")}


def load_summary(path):
    lines = open(path, encoding="utf-8").readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("=" * 10):
            start = i + 1
            break
    return "".join(lines[start:]).strip()


def load_transcripts(repo, names):
    tdir = f"{repo}/openemr_whisper_wer/data/fareez_osce/Data/Clean Transcripts"
    if os.path.isdir(tdir):
        out = {}
        for n in names:
            try:
                out[n] = open(f"{tdir}/{n}.txt", encoding="utf-8-sig").read()
            except UnicodeDecodeError:
                out[n] = open(f"{tdir}/{n}.txt", encoding="latin-1").read()
        return out
    # Fallback: read from rating packet (exact transcripts shown to raters and models)
    R = f"{repo}/rag_models/rating_packets"
    pkt = open(f"{R}/packet_rater_1.txt", encoding="utf-8").read()
    key = pd.read_csv(f"{R}/ANSWER_KEY_DO_NOT_SHARE.csv")
    key = key[key.rater == "rater_1"].set_index("summary_id")
    parts = re.split(r"\n#{20,}\n# SUMMARY (S\d+)", pkt)
    out = {}
    for i in range(1, len(parts), 2):
        conv = key.loc[parts[i], "conversation"]
        if conv not in out:
            sec_a = parts[i + 1].split("SECTION A:")[1].split("SECTION B:")[0]
            # Drop the header line under SECTION A
            sec_lines = sec_a.strip().splitlines()
            body = "\n".join(l for l in sec_lines[1:] if not set(l.strip()) <= set("-=")).strip()
            out[conv] = body
    return {n: out[n] for n in names}


def load_charts(repo, names):
    cdir = f"{repo}/rag_models/data/fareez_openemr_extracts"
    return {n: open(f"{cdir}/{n}_openemr.txt", encoding="utf-8").read().strip() for n in names}


def split_note_sentences(text):
    """Split a SOAP note into substantive clinical sentences for NLI verification."""
    sents = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        # Strip common section label prefixes like "Chief Complaint:" if followed by sentence text
        if len(para.split()) < 5:
            continue
        if para.lower().strip(" .") in {"no information available", "none", "not applicable", "n/a"}:
            continue
        # Split on sentence boundaries while avoiding decimals / abbreviations like Dr., Mr., Ms., mg.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", para)
        for p in parts:
            p = p.strip()
            if len(p.split()) >= 5 and "no information available" not in p.lower():
                words = p.split()
                if len(words) > 120:
                    # Break unpunctuated run-on blocks into <=80-word segments
                    for k in range(0, len(words), 80):
                        seg = " ".join(words[k:k + 80]).strip()
                        if len(seg.split()) >= 5:
                            sents.append(seg)
                else:
                    sents.append(p)
    return sents


def chunk_words(text, win=120, stride=60):
    words = text.split()
    if len(words) <= win:
        return [text.strip()] if text.strip() else []
    chunks = []
    for i in range(0, len(words), stride):
        c = " ".join(words[i:i + win]).strip()
        if c:
            chunks.append(c)
        if i + win >= len(words):
            break
    return chunks


@torch.no_grad()
def encode_roberta_full_tokens(text, tokenizer, model, device, layer=17, block_size=500):
    """Encode 100% of subword tokens using roberta-large layer 17 via non-overlapping blocks."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    n_tok = len(ids)
    if n_tok == 0:
        return torch.zeros((1, model.config.hidden_size), device=device), 0
    chunks = [ids[i:i + block_size] for i in range(0, n_tok, block_size)]
    embs = []
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    for ch in chunks:
        inp = torch.tensor([[bos] + ch + [eos]], dtype=torch.long, device=device)
        out = model(input_ids=inp, output_hidden_states=True)
        h = out.hidden_states[layer][0, 1:-1, :]  # drop BOS/EOS
        embs.append(F.normalize(h.float(), p=2, dim=-1))
    return torch.cat(embs, dim=0), n_tok


@torch.no_grad()
def compute_full_bertscore(items, device):
    print("\n[1/4] Computing untruncated full-text BERTScore (roberta-large, layer 17)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("roberta-large", local_files_only=True)
    mod = AutoModel.from_pretrained("roberta-large", local_files_only=True).to(device).eval()
    if device.type == "cuda":
        mod = mod.half()

    # Pre-encode the 40 unique transcripts once
    ref_cache = {}
    for it in items:
        c = it["conversation"]
        if c not in ref_cache:
            ref_cache[c] = encode_roberta_full_tokens(it["reference"], tok, mod, device)

    rows = []
    for idx, it in enumerate(items):
        ref_emb, ref_ntok = ref_cache[it["conversation"]]
        gen_emb, gen_ntok = encode_roberta_full_tokens(it["summary"], tok, mod, device)
        # Cosine similarity matrix: [gen_ntok, ref_ntok]
        sim = torch.matmul(gen_emb, ref_emb.T)
        P = float(sim.max(dim=1).values.mean().item())
        R = float(sim.max(dim=0).values.mean().item())
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        rows.append(dict(
            bert_p_full=P,
            bert_r_full=R,
            bert_f1_full=F1,
            roberta_note_tokens=gen_ntok,
            roberta_ref_tokens=ref_ntok,
            roberta_note_trunc_frac=max(0.0, 1.0 - 510.0 / max(gen_ntok, 1)),
            roberta_ref_trunc_frac=max(0.0, 1.0 - 510.0 / max(ref_ntok, 1)),
        ))
        if (idx + 1) % 60 == 0:
            print(f"  BERTScore full: {idx + 1}/{len(items)} ({time.time() - t0:.1f}s)")

    del mod
    torch.cuda.empty_cache()
    print(f"  Done in {time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


@torch.no_grad()
def embed_texts_batched(texts, tokenizer, model, device, max_len=512, batch_size=64, pooling="mean"):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out = model(**enc)
        if pooling == "cls":
            v = out.last_hidden_state[:, 0]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        vecs.append(F.normalize(v.float(), p=2, dim=-1).cpu())
    return torch.cat(vecs, dim=0)


@torch.no_grad()
def compute_chunked_embeddings(items, model_name, col_prefix, device, win_words=320, stride_words=240, max_len=512, pooling="mean", log_trunc_limit=None):
    print(f"\n[2/4] Computing full-text chunked embeddings: {model_name} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mod = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    if device.type == "cuda":
        mod = mod.half()

    # Encode 40 transcripts
    ref_vecs = {}
    ref_ntoks = {}
    for it in items:
        c = it["conversation"]
        if c not in ref_vecs:
            if log_trunc_limit:
                ref_ntoks[c] = len(tok.encode(it["reference"], add_special_tokens=False))
            chunks = chunk_words(it["reference"], win=win_words, stride=stride_words)
            cv = embed_texts_batched(chunks, tok, mod, device, max_len=max_len, pooling=pooling)
            ref_vecs[c] = F.normalize(cv.mean(dim=0, keepdim=True), p=2, dim=-1)[0]

    sims = []
    note_trunc = []
    ref_trunc = []
    for it in items:
        if log_trunc_limit:
            nt = len(tok.encode(it["summary"], add_special_tokens=False))
            note_trunc.append(max(0.0, 1.0 - float(log_trunc_limit) / max(nt, 1)))
            ref_trunc.append(max(0.0, 1.0 - float(log_trunc_limit) / max(ref_ntoks[it["conversation"]], 1)))
        chunks = chunk_words(it["summary"], win=win_words, stride=stride_words)
        cv = embed_texts_batched(chunks, tok, mod, device, max_len=max_len, pooling=pooling)
        nv = F.normalize(cv.mean(dim=0, keepdim=True), p=2, dim=-1)[0]
        sims.append(float(torch.dot(nv, ref_vecs[it["conversation"]]).item()))

    del mod
    torch.cuda.empty_cache()
    print(f"  {model_name} mean={np.mean(sims):.4f} in {time.time() - t0:.1f}s")
    out = {f"{col_prefix}_full": sims}
    if log_trunc_limit:
        out[f"{col_prefix}_note_trunc_frac"] = note_trunc
        out[f"{col_prefix}_ref_trunc_frac"] = ref_trunc
    return pd.DataFrame(out)


@torch.no_grad()
def compute_nli_faithfulness(items, charts, device):
    """
    RAG-retrieved sentence-level NLI faithfulness & contradiction using
    S-PubMedBert-MS-MARCO retrieval + DeBERTa-v3-large-mnli-fever-anli-ling-wanli.
    Evaluates each note sentence against:
      (a) top-2 retrieved transcript windows (conversation grounding)
      (b) top-2 retrieved windows from transcript + paired OpenEMR chart (all-input grounding)
    """
    print("\n[3/4] Retrieving evidence windows with S-PubMedBert-MS-MARCO...")
    t0 = time.time()
    ret_name = "pritamdeka/S-PubMedBert-MS-MARCO"
    rtok = AutoTokenizer.from_pretrained(ret_name, local_files_only=True)
    rmod = AutoModel.from_pretrained(ret_name, local_files_only=True).to(device).eval()
    if device.type == "cuda":
        rmod = rmod.half()

    # Build & embed transcript and chart windows per conversation
    conv_t_chunks, conv_t_embs = {}, {}
    conv_all_chunks, conv_all_embs = {}, {}
    for it in items:
        c = it["conversation"]
        if c not in conv_t_chunks:
            tc = chunk_words(it["reference"], win=130, stride=65)
            cc = chunk_words(charts[c], win=130, stride=80)
            te = embed_texts_batched(tc, rtok, rmod, device, max_len=512, pooling="mean")
            ce = embed_texts_batched(cc, rtok, rmod, device, max_len=512, pooling="mean")
            conv_t_chunks[c], conv_t_embs[c] = tc, te
            conv_all_chunks[c] = tc + cc
            conv_all_embs[c] = torch.cat([te, ce], dim=0)

    # For each note, split into sentences and retrieve top-2 premises for both scopes
    note_sent_counts = []
    pairs_t = []      # (note_idx, sent_idx, premise, hypothesis)
    pairs_all = []
    for nidx, it in enumerate(items):
        sents = split_note_sentences(it["summary"])
        if not sents:
            sents = [it["summary"][:500]]
        note_sent_counts.append(len(sents))
        se = embed_texts_batched(sents, rtok, rmod, device, max_len=256, pooling="mean")
        c = it["conversation"]

        # Top-2 transcript chunks
        sim_t = torch.matmul(se, conv_t_embs[c].T)
        k_t = min(2, sim_t.shape[1])
        top_t = torch.topk(sim_t, k=k_t, dim=1).indices.tolist()
        for sidx, (s, idxs) in enumerate(zip(sents, top_t)):
            for pidx in idxs:
                pairs_t.append((nidx, sidx, conv_t_chunks[c][pidx], s))

        # Top-2 transcript+chart chunks
        sim_a = torch.matmul(se, conv_all_embs[c].T)
        k_a = min(2, sim_a.shape[1])
        top_a = torch.topk(sim_a, k=k_a, dim=1).indices.tolist()
        for sidx, (s, idxs) in enumerate(zip(sents, top_a)):
            for pidx in idxs:
                pairs_all.append((nidx, sidx, conv_all_chunks[c][pidx], s))

    del rmod
    torch.cuda.empty_cache()
    print(f"  Retrieved {len(pairs_t)} transcript NLI pairs + {len(pairs_all)} all-input NLI pairs across {sum(note_sent_counts)} sentences in {time.time() - t0:.1f}s")

    print("\n[4/4] Scoring NLI entailment & contradiction with DeBERTa-v3-large-mnli-fever-anli-ling-wanli...")
    t1 = time.time()
    nli_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    ntok = AutoTokenizer.from_pretrained(nli_name, local_files_only=True)
    nmod = AutoModelForSequenceClassification.from_pretrained(nli_name, local_files_only=True).to(device).eval()
    if device.type == "cuda":
        nmod = nmod.half()

    def score_pairs(pairs, batch_size=96):
        ent = np.zeros(len(pairs), dtype=np.float32)
        con = np.zeros(len(pairs), dtype=np.float32)
        for i in range(0, len(pairs), batch_size):
            b = pairs[i:i + batch_size]
            prem = [x[2] for x in b]
            hyp = [x[3] for x in b]
            enc = ntok(prem, hyp, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = nmod(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # id2label: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            ent[i:i + len(b)] = probs[:, 0]
            con[i:i + len(b)] = probs[:, 2]
            if ((i // batch_size) + 1) % 60 == 0:
                print(f"    scored {min(i + batch_size, len(pairs))}/{len(pairs)} ({time.time() - t1:.1f}s)")
        return ent, con

    ent_t, con_t = score_pairs(pairs_t)
    ent_a, con_a = score_pairs(pairs_all)

    del nmod
    torch.cuda.empty_cache()

    def aggregate(pairs, ent, con, n_notes, sent_counts, suffix):
        # Per (note, sent), take max entailment and max contradiction over top-2 premises
        by_ns = {}
        for (nidx, sidx, _, _), e, c_ in zip(pairs, ent, con):
            key = (nidx, sidx)
            if key not in by_ns:
                by_ns[key] = [e, c_]
            else:
                if e > by_ns[key][0]:
                    by_ns[key][0] = e
                if c_ > by_ns[key][1]:
                    by_ns[key][1] = c_
        rows = []
        for nidx in range(n_notes):
            sc = sent_counts[nidx]
            es = np.array([by_ns[(nidx, s)][0] for s in range(sc)])
            cs = np.array([by_ns[(nidx, s)][1] for s in range(sc)])
            rows.append({
                f"nli_entail_{suffix}": float(es.mean()),
                f"nli_contra_{suffix}": float(cs.mean()),
                f"nli_supported_frac_{suffix}": float((es >= 0.5).mean()),
                f"nli_contradicted_frac_{suffix}": float((cs >= 0.5).mean()),
            })
        return pd.DataFrame(rows)

    df_t = aggregate(pairs_t, ent_t, con_t, len(items), note_sent_counts, "transcript")
    df_a = aggregate(pairs_all, ent_a, con_a, len(items), note_sent_counts, "inputs")
    df_out = pd.concat([df_t, df_a], axis=1)
    df_out["n_sentences"] = note_sent_counts
    print(f"  NLI scoring finished in {time.time() - t1:.1f}s")
    return df_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    R = f"{repo}/rag_models"
    out_path = f"{R}/results/revision/fareez_sota_metrics.csv"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    convs = [s["file_name"] for s in json.load(open(f"{R}/data/fareez_selected_40.json"))]
    transcripts = load_transcripts(repo, convs)
    charts = load_charts(repo, convs)

    items = []
    for m in MODELS_ALL:
        for c in convs:
            if (c, m) in FAILED:
                continue
            p = f"{R}/results/fareez/fareez_summaries/{m}/{c}.txt"
            s = load_summary(p)
            assert s and not s.startswith("Error generating summary"), (c, m)
            items.append(dict(conversation=c, model=m, summary=s, reference=transcripts[c]))
    print(f"Loaded {len(items)} rated notes across {len(MODELS_ALL)} models.")
    assert len(items) == 239

    base = pd.DataFrame([dict(conversation=it["conversation"], model=it["model"]) for it in items])
    df_bert = compute_full_bertscore(items, device)
    df_sbert = compute_chunked_embeddings(items, "sentence-transformers/all-MiniLM-L6-v2", "sbert", device,
                                          win_words=180, stride_words=140, max_len=256, pooling="mean", log_trunc_limit=256)
    df_bge = compute_chunked_embeddings(items, "BAAI/bge-large-en-v1.5", "bge_large", device,
                                        win_words=320, stride_words=240, max_len=512, pooling="cls")
    df_pm = compute_chunked_embeddings(items, "pritamdeka/S-PubMedBert-MS-MARCO", "pubmedbert", device,
                                       win_words=320, stride_words=240, max_len=512, pooling="mean")
    df_nli = compute_nli_faithfulness(items, charts, device)

    res = pd.concat([base, df_bert, df_sbert, df_bge, df_pm, df_nli], axis=1)
    res.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} ({len(res)} rows)")

    # Print quick summary
    print("\n=== EXACT TRUNCATION SUMMARY ===")
    print(f"BERTScore (510 subwords): notes truncated = {(res.roberta_note_tokens > 510).sum()}/{len(res)} "
          f"(mean {100*res.roberta_note_trunc_frac.mean():.1f}% discarded); "
          f"transcripts truncated = {(res.roberta_ref_tokens > 510).sum()}/{len(res)} "
          f"(mean {100*res.roberta_ref_trunc_frac.mean():.1f}% discarded)")
    print(f"SBERT (256 wordpieces):   notes truncated = {(res.sbert_note_trunc_frac > 0).sum()}/{len(res)} "
          f"(mean {100*res.sbert_note_trunc_frac.mean():.1f}% discarded); "
          f"transcripts truncated = {(res.sbert_ref_trunc_frac > 0).sum()}/{len(res)} "
          f"(mean {100*res.sbert_ref_trunc_frac.mean():.1f}% discarded)")

    cols = ["bert_f1_full", "sbert_full", "bge_large_full", "pubmedbert_full",
            "nli_entail_transcript", "nli_contra_transcript", "nli_entail_inputs", "nli_contra_inputs",
            "nli_supported_frac_transcript", "nli_supported_frac_inputs"]
    print("\n=== MEAN SOTA METRICS BY MODEL ===")
    print(res.groupby("model")[cols].mean().reindex(MODELS_ALL).round(3).to_string())


if __name__ == "__main__":
    main()
