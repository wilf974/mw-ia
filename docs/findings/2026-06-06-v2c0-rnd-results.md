# V2-C0 — Résultats RND (exploration intrinsèque apprise)

> Substrat : V2-ZY + Polyak 15×15 (`--max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 --polyak-tau 0.005 --max-attempts-bfs 500`), 5000 ép, eval V2-V (best @ diff=0.30, 10 seeds held-out).
> Métriques : `diff_max` (plafond scheduler) + `best_eval` (greedy) via `BX_PROBE_RESULT` ; `ratio_int_ext` (Σ intrinsèque / Σ extrinsèque) via `RND_RESULT`.
> Références : baseline V2-U **0.36 / 64 %** ; count-based B@0.02 (concurrent BX) **0.40 / 66.7 %**.

## GATE Phase 1 (n=3) : `diff_max` moy > 0.40 OU best_eval moy > 66.7 % → Phase 2 n=5

---

## RND β=0.5 (défaut) — n=3 — ÉCHEC

| Seed | `diff_max` | best_eval @ 0.30 | ratio int/ext | final |
|---|---|---|---|---|
| 0 | 0.05 | 10 % | 1.20 | 15 % @ 0.00 |
| 1 | 0.05 | 0 % | 1.33 | 1 % @ 0.00 |
| 2 | 0.05 | 0 % | 1.31 | 1 % @ 0.00 |
| **moyenne** | **0.05** | 3.3 % | ~1.28 | — |

**ÉCHEC franc** (GATE 0.40, baseline 0.36). β=0.5 casse l'entraînement.

**Diagnostic** : ce n'est PAS le mode « bonus écrase tout » du count-based β=0.1 (ratio ~1.3, bien sous le WARN 10:1). `predictor_loss=0.0000` → le predictor converge, MAIS la normalisation par running-std maintient le bonus normalisé **O(1) en permanence** (divise par la std des erreurs qui décroît → le bonus ne décroît jamais). Bonus dense ~0.5/step partout. En env **procédural** (maze neuf chaque épisode), tout état est « nouveau » → RND ne distingue pas « inexploré dans ce maze » de « nouveau maze » → bonus dense ≈ constant qui **noie le goal sparse**. L'agent vagabonde.

→ Re-essai β=0.1 (plan : re-tune β si le ratio montre une contamination).

---

## RND β=0.1 — n=3

| Seed | `diff_max` | best_eval @ 0.30 | ratio int/ext | final |
|---|---|---|---|---|
| 0 | _(en cours)_ | | | |
| 1 | _(en cours)_ | | | |
| 2 | _(en cours)_ | | | |
| **moyenne** | **_(en cours)_** | | | |

**Verdict β=0.1 : _(à compléter)_**

---

## Finding consolidé

_(à compléter une fois le re-tune parcouru)_
