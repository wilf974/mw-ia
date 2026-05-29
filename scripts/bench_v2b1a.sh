#!/usr/bin/env bash
# Bench V2-B1a — factoriel 2x2 (B1a x PER) n=5 same-seed 15x15.
# Bras 3 = B1a seul ; Bras 4 = B1a + PER.
# Auto-gate : attend que le GPU soit libre (AetherLife overnight) avant de demarrer.
# Logs uniques par seed (pas d'ecrasement). Reproductible (corrige l'absence de
# script de la 1ere tentative interrompue).
#
# Usage : bash scripts/bench_v2b1a.sh            (gate GPU actif)
#         NO_GATE=1 bash scripts/bench_v2b1a.sh  (lance immediatement)

set -u
cd "$(dirname "$0")/.." || exit 1
source .venv/Scripts/activate

LOGDIR="logs"
CKPTDIR="checkpoints"
mkdir -p "$LOGDIR" "$CKPTDIR"
SUP="$LOGDIR/v2b1a_bench_supervisor.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$SUP"; }

# Baseline desktop Windows de cette machine ~2100 MiB (Firefox/Teams/Edge/...).
# Un job d'entrainement CUDA ajoute >1.5 GB, donc 3500 distingue idle vs training.
GPU_FREE_MIB="${GPU_FREE_MIB:-3500}"   # GPU libre si memory.used < ce seuil

wait_for_gpu() {
    [ "${NO_GATE:-0}" = "1" ] && { log "GATE desactive (NO_GATE=1) — demarrage immediat."; return; }
    log "GATE GPU actif — attente liberation (seuil < ${GPU_FREE_MIB} MiB)."
    local consecutive=0 ticks=0
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [ -z "$used" ] && used=99999
        if [ "$used" -lt "$GPU_FREE_MIB" ]; then
            consecutive=$((consecutive+1))
            if [ "$consecutive" -ge 2 ]; then
                log "GPU libre (used=${used} MiB, confirme 2x) — demarrage du bench."
                return
            fi
        else
            consecutive=0
        fi
        ticks=$((ticks+1))
        if [ $((ticks % 5)) -eq 0 ]; then
            log "...en attente (used=${used} MiB)"
        fi
        sleep 60
    done
}

run_seed() {
    local arm="$1" seed="$2"; shift 2
    local extra=("$@")
    local lf="$LOGDIR/v2b1a_${arm}_seed${seed}.log"
    local ckpt="$CKPTDIR/v2b1a_${arm}_seed${seed}.pt"
    log "--- ${arm} seed ${seed} BEGIN ---"
    python scripts/train_cnn_lstm_dqn_procedural.py \
        --episodes 5000 --mode obstacles --device cuda --seed "$seed" \
        --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
        --polyak-tau 0.005 --b1a --max-attempts-bfs 500 \
        --eval-target-difficulty 0.30 \
        --best-checkpoint-path "$ckpt" \
        ${extra[@]+"${extra[@]}"} \
        > "$lf" 2>&1
    local rc=$?
    local best final
    best=$(grep "Best @ diff=0.30" "$lf" | tail -1)
    final=$(grep "^Final :" "$lf" | tail -1)
    log "--- ${arm} seed ${seed} END exit=${rc} | ${final} | ${best} ---"
}

log "===== BENCH V2-B1a START ====="
wait_for_gpu

log "===== BRAS 3 (B1a seul) ====="
for s in 0 1 2 3 4; do run_seed "bras3" "$s"; done

log "===== BRAS 4 (B1a + PER) ====="
for s in 0 1 2 3 4; do run_seed "bras4" "$s" --per; done

log "===== BENCH V2-B1a DONE ====="
log "Recap Bras 3 :"
grep "bras3 seed.* END" "$SUP" | tee -a "$SUP" >/dev/null
log "Recap Bras 4 :"
grep "bras4 seed.* END" "$SUP" | tee -a "$SUP" >/dev/null
