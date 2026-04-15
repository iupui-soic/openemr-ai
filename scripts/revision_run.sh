#!/usr/bin/env bash
# Orchestrator: launch all reviewer-revision experiments into one tmux session.
# Each window runs its own pipeline; logs tee to /tmp/revisions_<window>.log so
# you can `tail -f` them without attaching.
#
# Usage:
#   bash scripts/revision_run.sh            # launch everything
#   bash scripts/revision_run.sh status     # show window list + last 5 lines per log
#   bash scripts/revision_run.sh kill       # tear down the tmux session
#
# Idempotent: re-running launches only the windows that don't already exist.

set -euo pipefail
SESSION="revisions"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="/tmp"

cd "$REPO"

if ! command -v tmux >/dev/null; then
    echo "tmux not installed"; exit 1
fi

# ----- helpers -----------------------------------------------------------

ensure_session() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux new-session -d -s "$SESSION" -n control \
            "echo 'revisions session started'; bash"
    fi
}

window_exists() {
    tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | grep -qx "$1"
}

# Launch a window; cmd is the bash one-liner (already wrapped in `bash -lc`).
launch() {
    local name="$1"; shift
    local cmd="$*"
    if window_exists "$name"; then
        echo "  [skip] $name (window already exists)"
        return
    fi
    local logf="$LOG_DIR/revisions_${name}.log"
    : > "$logf"
    # Source venv inside the window and tee output to a log
    tmux new-window -t "$SESSION" -n "$name" \
        "bash -lc 'cd \"$REPO\" && source .venv/bin/activate && export PYTHONUNBUFFERED=1 && \
        ($cmd) 2>&1 | tee -a $logf; \
        echo; echo \"[done $name @ \$(date +%H:%M:%S)]\"; \
        echo \"(window left open; press any key to close)\"; read'"
    echo "  launch  $name  -> $logf"
}

# Sequence of commands separated by ';'
seq_cmds() { echo "$1"; }

# ----- subcommands -------------------------------------------------------

cmd_status() {
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "no session '$SESSION'"; exit 0
    fi
    echo "=== windows ==="
    tmux list-windows -t "$SESSION" -F '  #W (#{?window_active,active,idle})'
    echo
    echo "=== logs (last 5 lines each) ==="
    for log in $LOG_DIR/revisions_*.log; do
        [ -e "$log" ] || continue
        echo "--- $(basename "$log") ---"
        tail -n 5 "$log"
        echo
    done
}

cmd_kill() {
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "killed $SESSION" || echo "no session"
}

cmd_launch() {
    ensure_session

    # ---- All Groq ablations serialised in ONE window. Parallel windows trip
    # ---- per-model Groq rate limits; serial keeps it well under the cap.
    launch groq_chain "$(seq_cmds '
        for m in gpt-oss-120b gpt-oss-20b qwen3-32b; do
            python rag_models/ablations/run_ablation_groq.py --mode norag --model $m;
        done;
        for m in gpt-oss-120b gpt-oss-20b qwen3-32b; do
            for k in 1 3 5; do
                python rag_models/ablations/run_ablation_groq.py --mode k --k $k --model $m;
            done;
        done;
        for m in gpt-oss-120b gpt-oss-20b qwen3-32b; do
            for emb in clinicalbert pubmedbert; do
                python rag_models/ablations/run_ablation_groq.py --mode embed --embedding $emb --model $m;
            done;
        done;
        for m in gpt-oss-120b gpt-oss-20b; do
            for t in 0.1 0.5 0.7; do
                python rag_models/ablations/run_ablation_groq.py --mode temp --temperature $t --model $m;
            done;
        done;
        for m in gpt-oss-120b gpt-oss-20b; do
            for v in minimal hallucination_guarded; do
                python rag_models/ablations/run_ablation_groq.py --mode prompt --prompt-variant $v --model $m;
            done;
        done
    ')"

    # ---- ablations (MedGemma on Modal) — all 6 cells in one warm container
    launch medgemma "modal run rag_models/ablations/run_ablation_medgemma.py --cells norag,k1,k3,k5,embed_clinicalbert,embed_pubmedbert"

    # ---- All local-GPU jobs serialized on GPU 1 to avoid OOM. GPU 0 is left
    # ---- for the small SBERT loads in the groq_* windows.
    launch gpu1_chain "$(seq_cmds '
        export CUDA_VISIBLE_DEVICES=1;
        python openemr_whisper_wer/parakeet_06b_v2_wer.py --dataset both;
        python openemr_whisper_wer/diarization/run_der_primock57.py;
        python openemr_whisper_wer/diarization/run_der_institutional.py;
        python openemr_whisper_wer/voxtral_mini_wer.py --dataset both;
        python openemr_whisper_wer/diarization/run_der_fareez.py
    ')"

    echo
    echo "All windows launched. Inspect with:"
    echo "  bash scripts/revision_run.sh status"
    echo "  tmux attach -t $SESSION"
    echo "  tail -f /tmp/revisions_<job>.log"
}

case "${1:-launch}" in
    launch) cmd_launch ;;
    status) cmd_status ;;
    kill)   cmd_kill ;;
    *)      echo "usage: $0 [launch|status|kill]"; exit 1 ;;
esac
