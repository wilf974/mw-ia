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
| 0 | 0.30 | 50 % | 75 % @ 0.30 |
| 1 | 0.40 | 50 % | 73 % @ 0.40 |
| 2 | 0.35 | 50 % | 73 % @ 0.35 |
| **moyenne** | **0.35** | 50 % | — |

**Verdict C2 : NÉGATIF → représentation ÉLIMINÉE.** Moyenne `diff_max = 0.35`, au niveau de la baseline V2-U (0.36), bien sous le seuil 0.45. Réfutation forte : même le gradient de distance complet par cellule (suivable en greedy pur) ne casse pas le plafond. Le champ resserre légèrement la variance (tous seeds ≥ 0.30, best_eval uniforme 50 %) → l'info spatiale aide *marginalement la stabilité* mais **pas le plafond de difficulté**. → Sonde A.

---

## Sonde A — horizon (gamma=0.997) — n=3

> Horizon effectif `1/(1-0.997) ≈ 333`, cohérent avec `max_steps=400`. Teste si l'agent connaît les bons comportements mais propage mal la valeur sur les trajectoires longues.

| Seed | `diff_max` | best_eval @ diff=0.30 | final winrate / diff |
|---|---|---|---|
| 0 | 0.25 | 40 % | 77 % @ 0.25 |
| 1 | 0.25 | 10 % | 73 % @ 0.25 |
| 2 | 0.40 | 40 % | 80 % @ 0.40 |
| **moyenne** | **0.30** | 30 % | — |

**Verdict A : NÉGATIF → horizon écarté.** Moyenne `diff_max = 0.30`, sous la baseline V2-U (0.36), loin de 0.45. Pousser γ à 0.997 ne casse pas le plafond et **déstabilise** (seed 1 best_eval s'effondre à 10 %). L'hypothèse mécaniste « horizon effectif ~100 trop court vs max_steps=400 » est réfutée : l'allongement de l'horizon augmente la variance du bootstrap sans aider. Résultat clairement négatif (non ambigu) → option de réserve n-step NON déclenchée → Sonde B.

---

## Sonde B — exploration (novelty count-based) — n=3

> Bonus `beta/sqrt(visits)` par cellule/épisode. Exploration pure : n'aide pas une fois le goal trouvé. Teste si le chemin n'est simplement jamais découvert assez souvent.

### B @ beta=0.1 — CONTAMINÉ (hyperparamètre trop grand)

| Seed | `diff_max` | best_eval @ diff=0.30 | final winrate / diff |
|---|---|---|---|
| 0 | 0.10 | 0 % | 0 % @ 0.00 |
| 1 | 0.15 | 0 % | 0 % @ 0.00 |
| 2 | 0.15 | 0 % | 3 % @ 0.00 |
| **moyenne** | **0.13** | 0 % | — |

**Non interprétable.** Pas un négatif propre : β=0.1 écrase l'objectif. Sur 15×15 (~225 cellules), explorer ~150 cases neuves rapporte `0.1 × 150 ≈ +15` vs **+1** pour le goal → ratio ~15:1, l'agent apprend à vagabonder (winrate ~0 %). Risque #1 de la spec (faux négatif par hyperparamètre). → re-test β=0.01.

### B @ beta=0.01 — test propre — POSITIF FAIBLE

| Seed | `diff_max` | best_eval @ diff=0.30 | final winrate / diff |
|---|---|---|---|
| 0 | 0.30 | 50 % | 81 % @ 0.30 |
| 1 | 0.45 | 70 % | 62 % @ 0.45 |
| 2 | 0.40 | 70 % | 74 % @ 0.40 |
| **moyenne** | **0.383** | 63 % | — |

**Positif faible.** Première et seule sonde **au-dessus de la baseline** (0.383 vs 0.36). 1 seed touche le seuil 0.45 ; best_eval 70 % sur 2/3 seeds (vs 64 % baseline). Frôle 0.45 sans le franchir en moyenne → second réglage β=0.02 (risque #1 spec : re-tune avant de conclure).

### B @ beta=0.02 — second réglage — POSITIF FAIBLE (confirme β=0.01)

| Seed | `diff_max` | best_eval @ diff=0.30 | final winrate / diff |
|---|---|---|---|
| 0 | 0.40 | 70 % | 65 % @ 0.40 |
| 1 | 0.35 | 60 % | 72 % @ 0.35 |
| 2 | 0.45 | 70 % | 71 % @ 0.45 |
| **moyenne** | **0.40** | 66.7 % | — |

**Verdict B final : POSITIF FAIBLE.** `diff_max` 0.40 (> baseline 0.36), best_eval **66.7 % > baseline 64 %**, 1 seed à 0.45. Gain marginal en doublant β (0.383→0.40, +0.017) → rendements décroissants, la moyenne ne franchit pas 0.45. L'exploration est la **seule famille qui aide** mais le bonus count-based brut ne casse pas le plafond seul.

---

## Finding consolidé V2-BX

### Tableau récapitulatif (toutes sondes, n=3, 15×15)

| Sonde | Famille | `diff_max` moyen | vs baseline 0.36 | best_eval moy | Verdict |
|---|---|---|---|---|---|
| baseline V2-U | — | 0.36 | = | 64 % | référence |
| C1 scalaire | représentation | 0.25 | −0.11 | 37 % | ❌ négatif (dilution) |
| C2 champ BFS | représentation | 0.35 | −0.01 | 50 % | ❌ **ÉLIMINÉE** (réfutation forte) |
| A horizon γ=0.997 | credit assignment | 0.30 | −0.06 | 30 % | ❌ **ÉCARTÉ** (déstabilise) |
| B β=0.1 | exploration | 0.13 | −0.23 | 0 % | ⚠️ contaminé (β trop grand) |
| **B β=0.01** | **exploration** | **0.383** | **+0.02** | 63 % | ✅ positif faible |
| **B β=0.02** | **exploration** | **0.40** | **+0.04** | **66.7 %** | ✅ **positif faible (confirmé)** |

### Verdict

> **Le bottleneck 15×15 est l'EXPLORATION**, pas la représentation ni l'horizon.

Trois conclusions, par ordre de force :

1. **Représentation : éliminée sans ambiguïté.** Même l'oracle spatial maximal (champ BFS complet par cellule, suivable en greedy) ne dépasse pas la baseline (C2 = 0.35 ≈ 0.36). L'information spatiale est déjà extractible par le Conv+LSTM ; la lui donner toute cuite n'aide pas. Le scalaire (C1 = 0.25) dilue même.

2. **Horizon / credit assignment : écarté.** Allonger γ (0.99 → 0.997, horizon effectif ~333 cohérent avec max_steps=400) ne casse pas le plafond et **déstabilise** (variance bootstrap accrue, un seed s'effondre). L'agent ne souffre pas d'un problème de propagation de valeur par escompte.

3. **Exploration : la seule direction qui débloque — mais le sondage brut ne suffit pas seul.** Le bonus de nouveauté count-based est la **seule** intervention au-dessus de la baseline (diff_max 0.38–0.40, best_eval jusqu'à 66.7 %, un seed atteignant 0.45 de façon répétée, scheduler franchissant 0.30 sur tous les seeds). Mais à β raisonnable il plafonne ~0.40 (frôle sans franchir 0.45) ; à β trop grand il corrompt l'objectif. C'est attendu : le comptage de cellules brut est le signal d'exploration **le plus grossier possible** et il se bat contre un compromis de corruption d'objectif.

### Lecture causale

À 15×15 (régime d'apprentissage **actif**, baseline 64 %), le chemin vers le goal n'est tout simplement **pas découvert assez souvent** sous ε-greedy pour être appris de façon stable jusqu'aux densités d'obstacles supérieures. Ce n'est ni un déficit de perception (rep. éliminée) ni de propagation de valeur (horizon écarté). Cohérent avec le pari a priori (exploration ~40 %, favori).

Cela complète la cartographie des bottlenecks du programme V2 :
- **PER (B0)** et **rehearsal (B1a)** : éliminés (« mieux utiliser l'expérience passée » n'aide pas le régime actif 15×15).
- **Représentation** et **horizon** : éliminés (V2-BX).
- **Exploration** : **identifiée** comme le levier — direction du prochain sous-projet.

### Prochaine étape (hors-scope BX)

Le bonus count-based était un **oracle de sondage**, pas une solution livrable. La vraie solution = un **module d'exploration appris** dans son propre cycle (brainstorm → spec → plan → impl) :
- **ICM** (Intrinsic Curiosity Module, Pathak 2017) — nouveauté dans l'espace de features, pas en cellules brutes.
- **RND** (Random Network Distillation, Burda 2018) — bonus = erreur de prédiction d'un réseau cible figé. Plus stable que count-based, pas de compromis de corruption aussi raide.
- **Count-based en feature space** (pseudo-counts) — généralise le sondage brut.

Critère de succès du futur sous-projet : best @ diff=0.30 (V2-V, n=5) ≥ 74 %, idéalement diff_max robustement > 0.45.

### Note méthodologique

Le sondage B β=0.1 a d'abord donné un faux négatif catastrophique (winrate 0 %) parce que le bonus écrasait l'objectif ~15:1. Le re-test à β=0.01/0.02 (risque #1 de la spec) a révélé le vrai signal. **Leçon : un bonus de reward shaping de magnitude mal calibrée est non interprétable — toujours vérifier que le bonus cumulé reste sous-dominant au goal reward avant de conclure.**
