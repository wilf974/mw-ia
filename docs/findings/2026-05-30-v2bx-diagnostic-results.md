# V2-BX — Résultats du diagnostic causal du bottleneck 15×15

> Substrat fixe : V2-ZY + Polyak à 15×15 (`--max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 --polyak-tau 0.005 --max-attempts-bfs 500`), 5000 ép.
> Baseline V2-U 15×15 de référence : `diff_max ≈ 0.36` (seeds 0.40/0.35/0.40/0.35/0.30), best @ diff=0.30 ≈ 64 %.
> Métrique primaire : `diff_max` (plafond de difficulté atteint par le scheduler). Seuil de déclenchement confirmation : **0.45**.

## Arbre de décision appliqué

```
C1 (scalar) n=3 → moyenne diff_max
  ≥ 0.45  → représentation confirmée → confirmation V2-V n=5
  0.40-0.45 → escalade C2
  < 0.40  → (déviation documentée : on lance C2 quand même comme réfutation maximale)
C2 (field) n=3 → moyenne diff_max
  ≥ 0.45  → représentation (riche) confirmée → confirmation V2-V n=5
  < 0.45  → représentation ÉLIMINÉE → Sonde A (gamma=0.997)
Sonde A n=3 → < 0.45 → Sonde B (novelty beta)
Sonde B n=3 → < 0.45 → bottleneck plus profond
```

---

## Sonde C1 — représentation (scalaire BFS, plan uniforme) — n=3

| Seed | `diff_max` | best_eval @ diff=0.30 | final winrate / diff |
|---|---|---|---|
| 0 | 0.20 | 20 % | 70 % @ 0.20 |
| 1 | 0.35 | 70 % | 66 % @ 0.35 |
| 2 | 0.20 | 20 % | 68 % @ 0.20 |
| **moyenne** | **0.25** | 37 % | — |

**Verdict C1 : NÉGATIF.** Moyenne `diff_max = 0.25`, **sous la baseline V2-U (0.36)**. 2 seeds sur 3 tombent sous tous les seeds de la baseline. Lecture : le plan uniforme à une seule valeur ajoute un 4ᵉ canal peu informatif que le conv doit apprendre à lire → effet de dilution plutôt qu'aide. **Ce n'est pas un test propre de l'hypothèse représentation** (signal scalaire trop pauvre). → escalade C2 (oracle spatial maximal).

---

## Sonde C2 — représentation (champ BFS par cellule) — n=3

> Oracle maximal : gradient de distance par cellule, suivable en greedy. Réfutation forte de la famille représentation.

| Seed | `diff_max` | best_eval @ diff=0.30 | final winrate / diff |
|---|---|---|---|
| 0 | _(en cours)_ | | |
| 1 | _(en cours)_ | | |
| 2 | _(en cours)_ | | |
| **moyenne** | **_(en cours)_** | | |

**Verdict C2 : _(à compléter)_**

---

## Sonde A — horizon (gamma=0.997) — n=3

_(conditionnel : si C2 < 0.45)_

---

## Sonde B — exploration (novelty beta) — n=3

_(conditionnel : si A < 0.45)_

---

## Finding consolidé

_(à compléter une fois l'arbre parcouru)_
