#!/usr/bin/env bash
# Verifie la presence et la non-vacuite des 8 fichiers Aether v1.4 du catalogue v1.
# Exit code != 0 si un fichier est absent ou vide.
#
# La validation formelle reelle (verdict pass/fail des @example + @invariant)
# est executee via le MCP `aether_verify` cote Claude Code -- ce script reste
# un smoke check shell pour la CI publique qui n'a pas l'interpreteur Aether.

set -euo pipefail

cd "$(dirname "$0")/invariants"

EXPECTED=(
    "i1_gamma_in_open_unit.aether"
    "i2_bellman_contraction.aether"
    "i3_huber_nonneg.aether"
    "i4_winrate_bounds.aether"
    "i5_epsilon_schedule.aether"
    "i6_replay_buffer_capacity.aether"
    "i7_reward_bounded.aether"
    "i8_episode_termination_exclusive.aether"
)

FAILED=()

for f in "${EXPECTED[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  X $f : absent"
        FAILED+=("$f")
        continue
    fi
    if [[ ! -s "$f" ]]; then
        echo "  X $f : vide"
        FAILED+=("$f")
        continue
    fi
    echo "  OK $f"
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "Echec : ${#FAILED[@]} fichier(s) absent(s) ou vide(s) :"
    printf '  - %s\n' "${FAILED[@]}"
    exit 1
fi

echo ""
echo "OK : 8 fichiers .aether presents et non vides."
echo "Note : pour la validation formelle (@example + @invariant), executer"
echo "mcp__aether__aether_verify sur chaque fichier via Claude Code."
exit 0
