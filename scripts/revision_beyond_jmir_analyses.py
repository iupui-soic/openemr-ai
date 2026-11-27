"""
Supplementary analyses addressing R1 internal review items:
1. SBERT coverage vs. chunk-pooling vs. overlap ablation (all-MiniLM-L6-v2, local GPU).
2. Encounter-paired model selection agreement across Round 2 and Round 1 (with word-count baseline).
3. Within-model and residualized correlations (model+length, encounter+length, model+encounter+length)
   with 4,000 encounter-clustered bootstrap resamples.
4. Fareez common-subset WER table updated to include Qwen3-ASR-1.7B, Granite Speech 5.0 2B,
   and Ishiki ASR 1.0, plus normalized WER summary across all 4 corpora.

Run: .venv/bin/python scripts/revision_beyond_jmir_analyses.py
"""
import os
import re
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REV = os.path.join(REPO, "rag_models", "results", "revision")
ASR = os.path.join(REPO, "openemr_whisper_wer", "results")
DIMS = ["accuracy", "completeness", "organization", "conciseness", "clinical_utility", "overall_quality"]
MODELS_ALL = ["gpt-oss-120b", "gpt-oss-20b", "qwen3-32b", "medgemma-4b", "qwen3.8-27b", "llama-4-scout"]
FAILED = {("MSK0037", "medgemma-4b")}


def load_summary(path):
    lines = open(path, encoding="utf-8").readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("=" * 10):
            start = i + 1
            break
    return "".join(lines[start:]).strip()


def load_transcripts(names):
    tdir = os.path.join(REPO, "openemr_whisper_wer", "data", "fareez_osce", "Data", "Clean Transcripts")
    if os.path.isdir(tdir):
        out = {}
        for n in names:
            try:
                out[n] = open(f"{tdir}/{n}.txt", encoding="utf-8-sig").read()
            except UnicodeDecodeError:
                out[n] = open(f"{tdir}/{n}.txt", encoding="latin-1").read()
        return out
    R = os.path.join(REPO, "rag_models", "rating_packets")
    pkt = open(f"{R}/packet_rater_1.txt", encoding="utf-8").read()
    key = pd.read_csv(f"{R}/ANSWER_KEY_DO_NOT_SHARE.csv")
    key = key[key.rater == "rater_1"].set_index("summary_id")
    parts = re.split(r"\n#{20,}\n# SUMMARY (S\d+)", pkt)
    out = {}
    for i in range(1, len(parts), 2):
        conv = key.loc[parts[i], "conversation"]
        if conv not in out:
            sec_a = parts[i + 1].split("SECTION A:")[1].split("SECTION B:")[0]
            sec_lines = sec_a.strip().splitlines()
            body = "\n".join(l for l in sec_lines[1:] if not set(l.strip()) <= set("-=")).strip()
            out[conv] = body
    return {n: out[n] for n in names}


def load_merged_df():
    r = pd.read_csv(f"{REV}/round2_ratings_long.csv")
    r["composite"] = r[DIMS].mean(axis=1)
    notes = r.groupby(["conversation", "model", "round"])[DIMS + ["composite"]].mean().reset_index()
    m1 = pd.read_csv(f"{REV}/fareez_automated_metrics.csv").rename(columns={"sbert_coherence": "sbert_similarity"})
    m2 = pd.read_csv(f"{REV}/fareez_round2_metrics.csv")
    sota = pd.read_csv(f"{REV}/fareez_sota_metrics.csv")
    base = pd.concat([m1, m2], ignore_index=True)
    df = notes.merge(base, on=["conversation", "model"], how="inner").merge(sota, on=["conversation", "model"], how="inner")
    assert len(df) == 239, len(df)
    return df


def spearman_ci_clustered(df, xcol, ycol, adj_len=False, B=2000, seed=2026):
    convs = df["conversation"].unique()
    x_all = df[xcol].values.astype(float)
    y_all = df[ycol].values.astype(float)
    w_all = df["summary_words"].values.astype(float)
    idx_groups = [np.where(df["conversation"].values == c)[0] for c in convs]

    def calc_idx(idxs):
        rx = stats.rankdata(x_all[idxs])
        ry = stats.rankdata(y_all[idxs])
        if not adj_len:
            return float(np.corrcoef(rx, ry)[0, 1])
        rw = stats.rankdata(w_all[idxs])
        # Simple 1-covariate OLS residual
        rw_c = rw - rw.mean()
        denom = float(np.dot(rw_c, rw_c))
        rx_r = rx - rx.mean() - (np.dot(rx - rx.mean(), rw_c) / denom) * rw_c
        ry_r = ry - ry.mean() - (np.dot(ry - ry.mean(), rw_c) / denom) * rw_c
        return float(np.corrcoef(rx_r, ry_r)[0, 1])

    pt = calc_idx(np.arange(len(df)))
    rng = np.random.default_rng(seed)
    n_c = len(idx_groups)
    boots = np.empty(B)
    for b in range(B):
        chosen = rng.integers(0, n_c, size=n_c)
        idxs = np.concatenate([idx_groups[i] for i in chosen])
        boots[b] = calc_idx(idxs)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return pt, float(lo), float(hi)


@torch.no_grad()
def run_sbert_truncation_control(df):
    print("\n=== 1. SBERT COVERAGE VS CHUNK-POOLING VS OVERLAP CONTROL ===")
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mod = AutoModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    if device.type == "cuda":
        mod = mod.half()

    convs = sorted(df["conversation"].unique())
    transcripts = load_transcripts(convs)
    summaries = {}
    for row in df.itertuples():
        p = os.path.join(REPO, "rag_models", "results", "fareez", "fareez_summaries", row.model, f"{row.conversation}.txt")
        summaries[(row.conversation, row.model)] = load_summary(p)

    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id

    def encode_token_blocks(id_blocks, batch_size=64):
        if not id_blocks:
            return torch.zeros((1, mod.config.hidden_size))
        vecs = []
        for i in range(0, len(id_blocks), batch_size):
            batch = id_blocks[i:i + batch_size]
            maxl = max(len(b) for b in batch)
            inp = torch.full((len(batch), maxl), tok.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), maxl), dtype=torch.long, device=device)
            for j, b in enumerate(batch):
                inp[j, :len(b)] = torch.tensor(b, dtype=torch.long, device=device)
                mask[j, :len(b)] = 1
            out = mod(input_ids=inp, attention_mask=mask).last_hidden_state.float()
            m = mask.unsqueeze(-1).float()
            v = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            vecs.append(F.normalize(v, p=2, dim=-1).cpu())
        return torch.cat(vecs, dim=0)

    def make_blocks(ids, mode):
        if mode == "prefix_254":
            return [[cls_id] + ids[:254] + [sep_id]]
        elif mode == "prefix_254_chunk2":
            sub = ids[:254]
            return [[cls_id] + sub[i:i + 127] + [sep_id] for i in range(0, max(1, len(sub)), 127)]
        elif mode == "suffix_254":
            return [[cls_id] + ids[-254:] + [sep_id]]
        elif mode == "prefix_508_chunk2":
            sub = ids[:508]
            return [[cls_id] + sub[i:i + 254] + [sep_id] for i in range(0, max(1, len(sub)), 254)]
        elif mode == "full_nonoverlap_254":
            return [[cls_id] + ids[i:i + 254] + [sep_id] for i in range(0, max(1, len(ids)), 254)]
        elif mode == "full_nonoverlap_127":
            return [[cls_id] + ids[i:i + 127] + [sep_id] for i in range(0, max(1, len(ids)), 127)]
        else:
            raise ValueError(mode)

    ref_ids = {c: tok.encode(transcripts[c], add_special_tokens=False) for c in convs}
    note_ids = {k: tok.encode(txt, add_special_tokens=False) for k, txt in summaries.items()}

    modes = [
        ("prefix_254", "Prefix only (254 tokens, 1 window; default equivalent)"),
        ("prefix_254_chunk2", "Prefix only (254 tokens, 2 x 127-token chunks, pooled)"),
        ("suffix_254", "Suffix only (last 254 tokens, 1 window)"),
        ("prefix_508_chunk2", "Prefix 508 tokens (2 x 254-token chunks, pooled)"),
        ("full_nonoverlap_254", "Full text, non-overlapping 254-token chunks (100% coverage)"),
        ("full_nonoverlap_127", "Full text, non-overlapping 127-token chunks (100% coverage)"),
    ]

    df_ctrl = df.copy()
    rows = []
    for mode, desc in modes:
        ref_vec = {}
        for c in convs:
            cv = encode_token_blocks(make_blocks(ref_ids[c], mode))
            ref_vec[c] = F.normalize(cv.mean(dim=0, keepdim=True), p=2, dim=-1)[0]
        sims = []
        for row in df_ctrl.itertuples():
            cv = encode_token_blocks(make_blocks(note_ids[(row.conversation, row.model)], mode))
            nv = F.normalize(cv.mean(dim=0, keepdim=True), p=2, dim=-1)[0]
            sims.append(float(torch.dot(nv, ref_vec[row.conversation]).item()))
        df_ctrl[mode] = sims

    # Also add existing overlapping sbert_full for direct side-by-side comparison
    all_variants = modes + [("sbert_full", "Full text, overlapping 180-word windows (stride 140; >99% coverage)")]
    r2_convs = sorted(df_ctrl[df_ctrl["round"] == 2]["conversation"].unique())
    idx_df = df_ctrl.set_index(["conversation", "model"])

    for col, desc in all_variants:
        r1_rho, r1_lo, r1_hi = spearman_ci_clustered(df_ctrl[df_ctrl["round"] == 1], col, "composite", adj_len=False, B=2000)
        r2_rho, r2_lo, r2_hi = spearman_ci_clustered(df_ctrl[df_ctrl["round"] == 2], col, "composite", adj_len=False, B=2000)
        p_rho, p_lo, p_hi = spearman_ci_clustered(df_ctrl, col, "composite", adj_len=False, B=2000)
        pa_rho, pa_lo, pa_hi = spearman_ci_clustered(df_ctrl, col, "composite", adj_len=True, B=2000)
        acc_rho, acc_lo, acc_hi = spearman_ci_clustered(df_ctrl, col, "accuracy", adj_len=False, B=2000)
        r2_wins = sum(
            1 for c in r2_convs
            if (idx_df.loc[(c, "qwen3.8-27b"), col] > idx_df.loc[(c, "llama-4-scout"), col])
        )
        rows.append(dict(
            variant=col, description=desc,
            r1_comp_rho=r1_rho, r1_ci_lo=r1_lo, r1_ci_hi=r1_hi,
            r2_comp_rho=r2_rho, r2_ci_lo=r2_lo, r2_ci_hi=r2_hi,
            pooled_comp_rho=p_rho, pooled_ci_lo=p_lo, pooled_ci_hi=p_hi,
            pooled_adj_rho=pa_rho, adj_ci_lo=pa_lo, adj_ci_hi=pa_hi,
            pooled_acc_rho=acc_rho, acc_ci_lo=acc_lo, acc_ci_hi=acc_hi,
            r2_paired_wins=r2_wins,
        ))
        print(f"  {col:22s} | R1={r1_rho:+.2f} | R2={r2_rho:+.2f} | Pool={p_rho:+.2f} ({p_lo:+.2f}..{p_hi:+.2f}) | Adj={pa_rho:+.2f} | Acc={acc_rho:+.2f} | R2 paired={r2_wins}/40")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"{REV}/sbert_truncation_control.csv", index=False)
    print(f"  Saved {REV}/sbert_truncation_control.csv in {time.time() - t0:.1f}s")
    del mod
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_df


def run_paired_selection(df):
    print("\n=== 2. ENCOUNTER-PAIRED SELECTION AGREEMENT ===")
    idx = df.set_index(["conversation", "model"])
    convs = sorted(df["conversation"].unique())
    rules = [
        ("sbert_similarity", "Higher default SBERT (256-token cap)", False),
        ("bert_f1", "Higher default BERTScore F1 (510-token cap)", False),
        ("bert_f1_full", "Higher untruncated BERTScore F1", False),
        ("sbert_full", "Higher full-text SBERT", False),
        ("bge_large_full", "Higher full-text BGE-large", False),
        ("pubmedbert_full", "Higher full-text S-PubMedBERT", False),
        ("medcat_entity_recall", "Higher MedCAT entity recall", False),
        ("scispacy_entity_recall", "Higher scispaCy entity recall", False),
        ("nli_contra_transcript", "Lower NLI contradiction (transcript)", True),
        ("summary_words", "Longer note (word count baseline)", False),
    ]
    pairs = [
        ("Round 2", "Qwen3.8-27B vs LLaMA-4-Scout", "qwen3.8-27b", "llama-4-scout"),
        ("Round 1", "gpt-oss-120b vs MedGemma-4B", "gpt-oss-120b", "medgemma-4b"),
        ("Round 1", "gpt-oss-20b vs MedGemma-4B", "gpt-oss-20b", "medgemma-4b"),
        ("Round 1", "Qwen3-32B vs MedGemma-4B", "qwen3-32b", "medgemma-4b"),
        ("Round 1", "gpt-oss-120b vs gpt-oss-20b", "gpt-oss-120b", "gpt-oss-20b"),
        ("Round 1", "gpt-oss-120b vs Qwen3-32B", "gpt-oss-120b", "qwen3-32b"),
        ("Round 1", "gpt-oss-20b vs Qwen3-32B", "gpt-oss-20b", "qwen3-32b"),
    ]
    rows = []
    for rd, plab, m1, m2 in pairs:
        cs = [c for c in convs if (c, m1) in idx.index and (c, m2) in idx.index]
        nt_comp = [c for c in cs if idx.loc[(c, m1), "composite"] != idx.loc[(c, m2), "composite"]]
        nt_acc = [c for c in cs if idx.loc[(c, m1), "accuracy"] != idx.loc[(c, m2), "accuracy"]]
        m1_comp_wins = sum(1 for c in nt_comp if idx.loc[(c, m1), "composite"] > idx.loc[(c, m2), "composite"])
        m1_acc_wins = sum(1 for c in nt_acc if idx.loc[(c, m1), "accuracy"] > idx.loc[(c, m2), "accuracy"])
        for col, rlab, lower_better in rules:
            ag_comp = sum(
                1 for c in nt_comp
                if ((idx.loc[(c, m1), col] < idx.loc[(c, m2), col]) if lower_better else (idx.loc[(c, m1), col] > idx.loc[(c, m2), col]))
                == (idx.loc[(c, m1), "composite"] > idx.loc[(c, m2), "composite"])
            )
            ag_acc = sum(
                1 for c in nt_acc
                if ((idx.loc[(c, m1), col] < idx.loc[(c, m2), col]) if lower_better else (idx.loc[(c, m1), col] > idx.loc[(c, m2), col]))
                == (idx.loc[(c, m1), "accuracy"] > idx.loc[(c, m2), "accuracy"])
            )
            rows.append(dict(
                round=rd, pair=plab, model_a=m1, model_b=m2,
                n_blocks=len(cs), n_comp_nonties=len(nt_comp), model_a_comp_wins=m1_comp_wins,
                n_acc_nonties=len(nt_acc), model_a_acc_wins=m1_acc_wins,
                metric=col, rule_label=rlab,
                agree_comp=ag_comp, agree_comp_pct=round(100 * ag_comp / len(nt_comp), 1),
                agree_acc=ag_acc, agree_acc_pct=round(100 * ag_acc / len(nt_acc), 1),
            ))
    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"{REV}/paired_selection_agreement.csv", index=False)
    print(out_df[out_df["pair"] == "Qwen3.8-27B vs LLaMA-4-Scout"][["rule_label", "agree_comp", "n_comp_nonties", "agree_acc", "n_acc_nonties"]].to_string(index=False))
    print(f"  Saved {REV}/paired_selection_agreement.csv")
    return out_df


def run_within_model_and_residuals(df, B=4000, seed=2026):
    print("\n=== 3. WITHIN-MODEL AND RESIDUAL CORRELATIONS (4,000 CLUSTERED BOOTSTRAPS) ===", flush=True)
    t0 = time.time()
    convs = sorted(df["conversation"].unique())
    idx_groups = [np.where(df["conversation"].values == c)[0] for c in convs]
    target_metrics = [
        ("sbert_similarity", "SBERT (default, 256 cap)"),
        ("bert_f1", "BERTScore F1 (default, 510 cap)"),
        ("bert_f1_full", "BERTScore F1 (full text)"),
        ("sbert_full", "SBERT (full text)"),
        ("bge_large_full", "BGE-large (full text)"),
        ("pubmedbert_full", "S-PubMedBERT (full text)"),
        ("medcat_entity_recall", "MedCAT entity recall"),
        ("scispacy_entity_recall", "scispaCy entity recall"),
        ("nli_contra_transcript", "NLI contradiction (transcript)"),
        ("nli_contra_inputs", "NLI contradiction (transcript + chart)"),
        ("summary_words", "Note length (words)"),
    ]

    mod_dummies_all = pd.get_dummies(df["model"], drop_first=True, dtype=float).values
    w_all = df["summary_words"].values.astype(float)
    y_comp_all = df["composite"].values.astype(float)
    y_acc_all = df["accuracy"].values.astype(float)
    orig_enc = df["conversation"].astype("category").cat.codes.values

    # Pre-sample 4000 encounter bootstrap index arrays and cluster IDs once
    rng = np.random.default_rng(seed)
    n_c = len(convs)
    boot_samples = []
    for _ in range(B):
        chosen = rng.integers(0, n_c, size=n_c)
        idxs = np.concatenate([idx_groups[i] for i in chosen])
        enc_ids = np.concatenate([np.full(len(idx_groups[i]), ci, dtype=np.int32) for ci, i in enumerate(chosen)])
        boot_samples.append((idxs, enc_ids))

    def demean_by_enc(mat, enc_ids, n_enc):
        # mat: (n, k) float64
        counts = np.bincount(enc_ids, minlength=n_enc).astype(float)
        sums = np.zeros((n_enc, mat.shape[1]), dtype=float)
        np.add.at(sums, enc_ids, mat)
        means = sums / np.maximum(counts[:, None], 1.0)
        return mat - means[enc_ids]

    def calc_all_modes(x_vec, y_vec, w_vec, md_mat, enc_ids, n_enc, include_len):
        rx = stats.rankdata(x_vec)
        ry = stats.rankdata(y_vec)
        rw = stats.rankdata(w_vec)
        # 1) model + len
        X_ml = np.column_stack([np.ones(len(rx)), md_mat] + ([rw] if include_len else []))
        b_ml, *_ = np.linalg.lstsq(X_ml, np.column_stack([rx, ry]), rcond=None)
        res_ml = np.column_stack([rx, ry]) - X_ml @ b_ml
        r_ml = float(np.corrcoef(res_ml[:, 0], res_ml[:, 1])[0, 1])

        # Demean [rx, ry, rw, md_mat] by enc_ids once for both enc_len and model_enc_len
        big = np.column_stack([rx, ry, rw, md_mat])
        dm = demean_by_enc(big, enc_ids, n_enc)
        rx_d, ry_d, rw_d, md_d = dm[:, 0], dm[:, 1], dm[:, 2], dm[:, 3:]

        # 2) enc + len
        if include_len:
            denom = float(np.dot(rw_d, rw_d))
            rx_el = rx_d - (np.dot(rx_d, rw_d) / denom) * rw_d
            ry_el = ry_d - (np.dot(ry_d, rw_d) / denom) * rw_d
            r_el = float(np.corrcoef(rx_el, ry_el)[0, 1])
        else:
            r_el = float(np.corrcoef(rx_d, ry_d)[0, 1])

        # 3) model + enc + len
        X_mel = np.column_stack([md_d, rw_d]) if include_len else md_d
        b_mel, *_ = np.linalg.lstsq(X_mel, np.column_stack([rx_d, ry_d]), rcond=None)
        res_mel = np.column_stack([rx_d, ry_d]) - X_mel @ b_mel
        r_mel = float(np.corrcoef(res_mel[:, 0], res_mel[:, 1])[0, 1])
        return r_ml, r_el, r_mel

    rows = []
    for mcol, mlab in target_metrics:
        x_all = df[mcol].values.astype(float)
        include_len = (mcol != "summary_words")
        per_m_comp = [float(stats.spearmanr(df[df.model == m][mcol], df[df.model == m]["composite"]).statistic) for m in MODELS_ALL]
        per_m_acc = [float(stats.spearmanr(df[df.model == m][mcol], df[df.model == m]["accuracy"]).statistic) for m in MODELS_ALL]
        rec = dict(
            metric=mcol, label=mlab,
            within_model_comp_min=min(per_m_comp), within_model_comp_max=max(per_m_comp),
            within_model_acc_min=min(per_m_acc), within_model_acc_max=max(per_m_acc),
        )
        for dim, y_all in (("composite", y_comp_all), ("accuracy", y_acc_all)):
            pt_ml, pt_el, pt_mel = calc_all_modes(x_all, y_all, w_all, mod_dummies_all, orig_enc, n_c, include_len)
            b_ml = np.empty(B); b_el = np.empty(B); b_mel = np.empty(B)
            for bi, (idxs, enc_ids) in enumerate(boot_samples):
                b_ml[bi], b_el[bi], b_mel[bi] = calc_all_modes(
                    x_all[idxs], y_all[idxs], w_all[idxs], mod_dummies_all[idxs], enc_ids, n_c, include_len
                )
            for mode, pt, arr in (("model_len", pt_ml, b_ml), ("enc_len", pt_el, b_el), ("model_enc_len", pt_mel, b_mel)):
                lo, hi = np.percentile(arr, [2.5, 97.5])
                rec[f"{mode}_{dim}"] = pt
                rec[f"{mode}_{dim}_lo"] = float(lo)
                rec[f"{mode}_{dim}_hi"] = float(hi)
        rows.append(rec)
        print(
            f"  {mcol:22s} | within-mod comp [{rec['within_model_comp_min']:+.2f}, {rec['within_model_comp_max']:+.2f}] "
            f"| mod+len comp={rec['model_len_composite']:+.2f} ({rec['model_len_composite_lo']:+.2f}..{rec['model_len_composite_hi']:+.2f}), acc={rec['model_len_accuracy']:+.2f} ({rec['model_len_accuracy_lo']:+.2f}..{rec['model_len_accuracy_hi']:+.2f}) "
            f"| mod+enc+len comp={rec['model_enc_len_composite']:+.2f} ({rec['model_enc_len_composite_lo']:+.2f}..{rec['model_enc_len_composite_hi']:+.2f}), acc={rec['model_enc_len_accuracy']:+.2f} ({rec['model_enc_len_accuracy_lo']:+.2f}..{rec['model_enc_len_accuracy_hi']:+.2f})",
            flush=True,
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"{REV}/within_model_and_residual_correlations.csv", index=False)
    print(f"  Saved {REV}/within_model_and_residual_correlations.csv in {time.time() - t0:.1f}s", flush=True)
    return out_df


def run_asr_all11_common():
    print("\n=== 4. FAREEZ COMMON-SUBSET WER (ALL 11 MODELS) + SPEECH-LLM NORMALIZED WER ===")
    p_old = pd.read_csv(f"{REV}/fareez_wer_labels_removed.csv", index_col=0)
    new_files = {
        "qwen3-asr": "fareez-qwen3-asr.csv",
        "granite-speech": "fareez-granite-speech.csv",
        "ishiki-asr": "fareez-ishiki-asr.csv",
    }
    name_map = {
        "parakeet-tdt-06b-v2": "Parakeet TDT 0.6B v2",
        "granite-speech": "Granite Speech 5.0 2B",
        "groq": "Whisper Large v3 Turbo (Groq API)",
        "voxtral-mini": "Voxtral Mini 3B",
        "parakeet": "Parakeet TDT 1.1B",
        "qwen3-asr": "Qwen3-ASR-1.7B",
        "canary": "Canary 1B-v2",
        "whisper-v3": "Whisper Large v3",
        "whisper-turbo": "Whisper Large v3 Turbo (self-hosted)",
        "ishiki-asr": "Ishiki ASR 1.0",
        "medasr": "MedASR",
    }
    p_all = p_old.copy()
    wer_norm_fareez = {}
    for k, fn in new_files.items():
        d = pd.read_csv(f"{ASR}/{fn}").set_index("name")
        d = d[~d.index.isin({"RES0002", "RES0054"})]
        p_all[k] = d["wer"]
        wer_norm_fareez[k] = 100.0 * d["wer_norm"].mean()

    common_all11 = p_all.dropna().index
    common_10 = p_all.drop(columns=["whisper-v3"]).dropna().index
    rows = []
    for k in p_all.columns:
        rows.append(dict(
            key=k, model=name_map[k],
            files_completed=int(p_all[k].notna().sum()),
            wer_all=round(100.0 * p_all[k].mean(), 2),
            wer_common_10=(round(100.0 * p_all.loc[common_10, k].mean(), 2) if k != "whisper-v3" else np.nan),
            wer_common_11=round(100.0 * p_all.loc[common_all11, k].mean(), 2),
            wer_norm_all=(round(wer_norm_fareez[k], 2) if k in wer_norm_fareez else np.nan),
        ))
    out = pd.DataFrame(rows).sort_values("wer_all")
    out.to_csv(f"{REV}/fareez_wer_all11_common.csv", index=False)
    print(f"Common to 10 models (excl Whisper Large v3): {len(common_10)}; common to all 11 models: {len(common_all11)}")
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    df = load_merged_df()
    run_sbert_truncation_control(df)
    run_paired_selection(df)
    run_within_model_and_residuals(df, B=4000, seed=2026)
    run_asr_all11_common()
