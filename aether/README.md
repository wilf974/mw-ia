# Aether — validation formelle MW_IA V2-A

Chaque fichier `invariants/iN_*.aether` formalise un invariant du catalogue v1
(cf. `docs/superpowers/specs/2026-05-21-mw-ia-v2-aether-guardrails-design.md`).

## Nature de la validation

Aether v1.4 est un **interpréteur Lisp typé avec test runner property-based**,
pas un theorem prover SMT. Les fichiers de ce dossier ne sont donc pas des
*preuves formelles universelles* au sens mathématique, mais des **validations
déclaratives** combinant :

- une (ou plusieurs) `(fn ...)` qui formalisent la propriété de l'invariant ;
- une annotation `@invariant` qui doit tenir sur le résultat retourné ;
- des annotations `@example` représentatives qui couvrent les cas positifs **et**
  négatifs documentés du catalogue.

`mcp__aether__aether_verify` (alias du `judge_agent`) exécute tous les `@example`
et vérifie les `@invariant`, retournant `verdict: ACCEPT` ou `REJECT`.

Cette approche reste précieuse comme :
1. Specification exécutable formellement typée du contrat de chaque invariant ;
2. Documentation lisible des bornes/comportements attendus ;
3. Couche défensive supplémentaire au-dessus des tests property-based runtime
   Python (`tests/guardrails/test_invariants.py`).

Une vraie vérification universelle (Z3 / Lean / Coq) reste un objectif possible
mais hors-scope V2-A.

## Lancer la validation

Une validation = un appel à `mcp__aether__aether_verify` avec le contenu du
fichier. Le script `verify_all.sh` itère sur les 8 fichiers et exit ≠ 0 si l'un
d'eux est absent ou vide ; la validation formelle reste exécutée via le MCP
Aether (pas de binaire shell standalone garanti disponible en CI publique).

## Convention de nommage

`invariants/iN_<snake_case>.aether` ↔ `mw_ia/guardrails/invariants.py::@invariant("IN")`

Cohérence vérifiée par `tests/guardrails/test_aether_python_sync.py`.

## Catalogue v1

| ID | Fichier                                  | Énoncé |
| -- | ---------------------------------------- | ------ |
| I1 | `i1_gamma_in_open_unit.aether`           | γ ∈ (0,1) |
| I2 | `i2_bellman_contraction.aether`          | Bellman γ-Lipschitz |
| I3 | `i3_huber_nonneg.aether`                 | Huber(y, ŷ) ≥ 0 |
| I4 | `i4_winrate_bounds.aether`               | winrate ∈ [0,1] |
| I5 | `i5_epsilon_schedule.aether`             | ε décroît, ∈ [0,1] |
| I6 | `i6_replay_buffer_capacity.aether`       | buffer.size ≤ capacity |
| I7 | `i7_reward_bounded.aether`               | r_min ≤ r_max |
| I8 | `i8_episode_termination_exclusive.aether`| terminated ⊕ truncated |
