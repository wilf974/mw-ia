# V2-BX — Diagnostic causal du bottleneck 15×15 (spec design)

> Date : 2026-05-30
> Sous-projet : **BX** (étude diagnostique, hors séquence A-F mais préalable à B2/curiosity/n-step/nouvelle archi)
> Statut : spec validée en brainstorming, en attente de plan d'implémentation

---

## 1. Contexte & motivation

Après les sous-projets V2-Z (CNN), V2-W (Double DQN), V2-V (eval rigoureux), V2-ZY (CNN+LSTM), V2-U (Polyak), V2-B0 (PER) et V2-B1a (snapshot rehearsal), l'état scientifique est le suivant :

| Famille d'hypothèses | Statut |
|---|---|
| Meilleur replay (PER, B0) | Écartée comme solution générale (aide à 10×10 saturé, dégrade à 15×15 actif) |
| Rejouer les bonnes trajectoires (B1a) | Écartée (15×15 : −10 pp seul, −16 pp avec PER) |
| Stabilité des targets (Polyak, U) | **Validée** |
| Mémoire temporelle (LSTM, Y) | **Validée** |
| Représentation spatiale (CNN, Z) | **Validée** |
| Mesure honnête (V2-V) | **Validée** |

**Lecture consolidée** : le système sait *apprendre, mémoriser et se stabiliser*. Ce qu'il ne sait pas encore faire, c'est **découvrir efficacement des solutions dans un espace plus grand** (15×15). Les deux familles « mieux utiliser l'expérience passée » (PER, rehearsal) ayant été éliminées, le vrai bottleneck est ailleurs.

**Baseline de référence** : V2-ZY + Polyak à 15×15 (V2-U) → mean best @ diff=0.30 = **64 %**, `diff_max` training ≈ **0.36**, 0/5 collapse.

**Le danger qu'on évite** : partir aveuglément sur B2 (mémoire épisodique), curiosity, n-step ou une nouvelle archi sans savoir lequel adresse le vrai bottleneck. C'est exactement la discipline diagnostique qui a produit le finding **LSTM × Polyak** plutôt qu'une intuition poursuivie en aveugle.

## 2. Objectif

**Trancher** quelle famille est le bottleneck du passage au régime supérieur 15×15, parmi :

- **Représentation** — l'information spatiale utile n'est pas extractible de l'observation actuelle par le Conv+LSTM.
- **Horizon / credit assignment** — l'agent connaît les bons comportements mais propage mal la valeur sur les trajectoires longues.
- **Exploration / découverte** — le chemin n'est simplement jamais découvert assez souvent pour être appris.

**Le livrable n'est PAS une feature** : c'est une **décision mesurée**. Une sonde positive *ou* une réfutation propre des trois familles sont **toutes deux des succès diagnostiques**.

### Pari a priori (à transformer en mesures)

| Hypothèse | Probabilité a priori |
|---|---|
| Exploration / découverte | ~40 % |
| Horizon / credit assignment | ~30 % |
| Représentation spatiale | ~20 % |
| Autre phénomène curriculum | ~10 % |

Note de mécanisme renforçant le prior horizon : à 15×15, `max_steps=400` mais γ=0.99 ⇒ horizon effectif `1/(1−0.99) ≈ 100` pas. L'agent escompte le `goal_reward` quasi à zéro sur les trajectoires longues. Raison concrète de soupçonner un mismatch d'horizon — la Sonde A l'isole proprement.

## 3. Principe : sondes-oracles jetables

Chaque sonde **triche temporairement** (injecte une information oracle ou modifie l'objectif) pour mesurer le *headroom* d'une famille. **Les sondes ne sont PAS livrées.** Elles désignent une famille ; la vraie solution non-oracle (auxiliary distance head, n-step propre, ICM/curiosity…) fait l'objet d'un sous-projet ultérieur.

**Contraintes d'orthogonalité** (actées en brainstorming) :
- Exploration testée par **nouveauté pure** (count-based), qui n'aide pas une fois le but trouvé → pas de confound horizon.
- Horizon testé par **γ / n-step pur**, qui ne change rien à la découverte.
- Un *reward dense* améliorerait à la fois exploration ET credit assignment → **écarté** comme sonde (ambigu).

**Convention de code** : flags `default=off`, opt-in CLI, baselines reproductibles bit-à-bit. Aucune modif en place des modules livrés sans garde `if flag`. Pattern strict de `per_enabled` (B0) / `b1a_enabled` (B1a).

## 4. Substrat fixe (testbed unique)

V2-ZY + Polyak à 15×15, une seule archi, une seule taille :

```bash
python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed $S \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/bx_<sonde>_seed${S}.pt
    # + le flag spécifique de la sonde
```

**Hors-scope explicite** : pas de 10×10, pas de V2-W (sans LSTM), pas de combinaison de sondes en première passe, aucun module appris (curiosity NN, attention) — ce sont des solutions, pas des sondes.

## 5. Les trois sondes

### 5.1 Sonde C — Représentation (oracle gradué, lancée en premier)

Donner à l'agent la **réponse-comme-feature** et voir s'il débloque. Graduée du minimal au maximal :

**Normalisation commune** : `dist_norm = rows * cols` (borne supérieure sûre de toute longueur de chemin BFS 4-connexe dans la grille), fixe et identique pour tous les mazes ⇒ valeurs comparables entre épisodes (un normaliseur par-maze rendrait le scalaire non comparable). Convention : **0.0 = sur le goal**, valeurs croissantes en s'éloignant.

- **C1 — scalaire** : `BFS(agent → goal) / dist_norm`, un seul nombre dans [0,1]. Injecté **après le flatten conv, concaténé avant la LSTM** (dimension d'entrée LSTM `+1`). Teste l'ajout représentationnel **minimal** (« un sens du progrès »). Mappe 1:1 vers une vraie solution cheap = *auxiliary distance head*.
- **C2 — champ (escalade conditionnelle)** : 4ᵉ canal d'observation = **carte de distance-au-goal BFS par cellule** (`BFS(cellule → goal) / dist_norm` en chaque case libre atteignable). Convention sentinelle explicite : **cellules-obstacle, cellules non-atteignables et padding = 1.0** (valeur « plus loin que tout »), goal = 0.0. Oracle **maximal** : l'agent n'a qu'à suivre le gradient descendant. Code le moins invasif (`in_channels 3→4`, déjà paramétré dans `ConvRecurrentQNetwork`).

**BFS** : réutilise `maze_bfs_check` / la logique BFS de `mw_ia/envs/maze_generators.py` pour calculer les distances géodésiques 4-connexes. Calculé au `reset()` (maze figé sur l'épisode).

**Blast radius** : `config.py` (flag `bx_repr_oracle: str = "none" | "scalar" | "field"`), `procedural_env.py` (encodeur produisant scalaire ou 4ᵉ canal), `conv_recurrent.py` (branche concat conditionnelle pour C1 / `in_channels=4` pour C2). Tout conditionnel.

### 5.2 Sonde A — Horizon (si C négatif)

`--gamma 0.997` (horizon effectif ~333, cohérent avec `max_steps=400`). **Zéro logique nouvelle** : `gamma` est déjà un champ de `ConvRecurrentDQNConfig` (validé ∈ (0,1)) ; il suffit d'exposer le flag CLI `--gamma` (actuellement absent du script V2-ZY).

**Réserve** (si γ ambigu) : **n-step returns** (n=5-10), plus invasif — accumulation du retour escompté sur n pas dans le `SequenceReplayBuffer` / `RecurrentDQNTrainer`. Nouveau code, seulement si nécessaire.

**Blast radius** : 1 ligne CLI (`--gamma`). (n-step = code dédié, conditionnel, en réserve.)

### 5.3 Sonde B — Exploration (si A négatif)

Bonus de **nouveauté count-based pur** : table de comptes de visites de cellules **par maze/épisode** (les mazes se régénèrent à chaque `reset()` ; un compte global (r,c) serait dénué de sens). Bonus `β / √(visits_cellule)` ajouté au reward à chaque pas. N'aide **pas** une fois le but trouvé → exploration pure.

**Blast radius** : `config.py` (flag `bx_novelty_beta: float = 0.0`, `β=0 ⇒ no-op`), point d'injection reward dans `gridworld.step()` / `procedural_env` (table de visites reset au `reset()`). Conditionnel.

## 6. Protocole & arbre de décision

**Métrique primaire** : `diff_max` = difficulté maximale atteinte par le scheduler sur le run (le plafond, pas la valeur finale). Passe rapide **n=3 (seeds 0,1,2)**. Confirmation **n=5 (seeds 0-4)** uniquement sur la famille gagnante.

**Barème** :
```text
faible   : diff_max > 0.40
fort     : diff_max ≥ 0.45   ← seuil de déclenchement confirmation
décisif  : diff_max ≥ 0.50
```

**Arbre** :
```text
SONDE C1  (BFS distance scalaire, concat pré-LSTM)         n=3
  diff_max ≥ 0.45  → REPRÉSENTATION confirmée ─────────────► confirmation V2-V n=5
  diff_max 0.40–0.45 → escalade C2
  diff_max < 0.40  → représentation simple insuffisante → A
        │
SONDE C2  (champ BFS par cellule, 4ᵉ canal)  [si ambigu]    n=3
  diff_max ≥ 0.45  → REPRÉSENTATION (riche) confirmée ─────► confirmation V2-V n=5
  diff_max < 0.45  → représentation ÉLIMINÉE (réfutation forte) → A
        │
SONDE A   (γ=0.997 ; n-step en réserve)                     n=3
  diff_max ≥ 0.45  → HORIZON confirmé ────────────────────► confirmation V2-V n=5
  diff_max < 0.45  → (essai n-step si γ ambigu) → B
        │
SONDE B   (count-based novelty β/√visits, par maze)         n=3
  diff_max ≥ 0.45  → EXPLORATION confirmée ───────────────► confirmation V2-V n=5
  diff_max < 0.45  → BOTTLENECK PLUS PROFOND → re-cadrage
                      (réserve : combinaison des sondes qui frôlent 0.45)

Confirmation V2-V (famille gagnante) : best @ diff=0.30 greedy, n=5
  64 % → ≥ 74 %  = famille validée comme levier réel
```

**Compute estimé** (1 run 15×15 5000 ép ≈ 0,75 h GPU RTX 3060) :
- Chemin minimal (C1 positif) : 3 runs (~2,25 h) + confirmation 5 runs (~3,75 h) ≈ **6 h**.
- Chemin maximal (tout négatif jusqu'à B) : 12-15 runs ≈ **10-12 h**.
- Séquentiel, court-circuit à chaque étage.

## 7. Composants & isolation

| Sonde | Fichiers touchés | Mécanisme | Tests unitaires |
|---|---|---|---|
| **C1/C2** | `config.py`, `procedural_env.py`, `conv_recurrent.py` | distance/​champ BFS via logique `maze_generators` | correction distance BFS sur maze connu ; shape obs (scalaire vs 4ᵉ canal) ; forward réseau avec/sans oracle ; no-op si `none` |
| **A** | `scripts/train_cnn_lstm_dqn_procedural.py` | flag `--gamma` → champ config existant | plumbing γ CLI → config → trainer |
| **B** | `config.py`, `gridworld.py`/`procedural_env.py` | bonus reward count-based, table par épisode | calcul bonus `β/√visits` ; reset table au `reset()` ; `β=0` ⇒ reward inchangé |

**Garanties d'isolation** :
- Chaque unité a un but unique, testable indépendamment.
- `bx_repr_oracle="none"`, `gamma=0.99`, `bx_novelty_beta=0.0` ⇒ comportement **identique** à la baseline V2-U (reproductibilité bit-à-bit).
- 1 smoke CI par sonde (10 ép CPU) : oracle activé + cohabitation Polyak/PER.

## 8. Stratégie de test

- **TDD test-first** par tâche (rouge → vert → commit), pattern V1/V2-A/.../V2-B1a.
- Tests déterministes CPU pour : exactitude BFS (distance connue sur un maze fixe), shapes d'observation, forward réseau (dimensions avec/sans oracle), calcul du bonus de nouveauté, et **invariants de no-op** (flags off ⇒ baseline).
- Smoke CI ajouté à `.github/workflows/aether_verify.yml` (≤10 ép CPU par sonde).
- `pytest -q` doit rester vert (baseline 356 + nouveaux tests sondes).

## 9. Livrable & critères de sortie

- **Livrable** : table des `diff_max` par sonde (n=3) + confirmation V2-V de la gagnante (n=5) + finding consolidé (famille identifiée *ou* « bottleneck plus profond »). Documenté dans `docs/.../findings/` + section CLAUDE.md. Mémoire mise à jour. Tag **`v0.2.0-bx`** (cohérent avec B0/B1a taggés même sur finding négatif).
- **Critère de succès du sous-projet** : *trancher*, pas forcément débloquer.
- **Suite hors-scope BX** : si une famille est positive → nouveau cycle brainstorm → spec → plan pour la **vraie solution non-oracle** :
  - C positive → *auxiliary distance head* / représentation de distance apprise / attention spatiale.
  - A positive → n-step returns propre / tuning γ assumé.
  - B positive → curiosity / ICM / count-based appris.

## 10. Risques & pièges

1. **Faux négatif par hyperparamètre mal choisi** : une sonde peut échouer à cause d'un mauvais réglage (β trop faible, γ mal calibré). Mitigation : pour une sonde qui *frôle* 0.45, un second réglage rapide avant de conclure négatif.
2. **Scheduler dynamique masque l'effet** : c'est pourquoi la métrique primaire est `diff_max` (plafond atteint), pas le winrate brut. Une sonde qui aide fait *monter le scheduler*, pas forcément le winrate.
3. **C2 « trivialise » la tâche** : le champ BFS complet rend la descente de gradient quasi triviale. Un C2 positif signifie « la perception spatiale parfaite débloque » → solution = mieux *apprendre* cette représentation. Un C2 négatif est un signal fort (même l'oracle maximal n'aide pas).
4. **`max_attempts_bfs=100` par défaut** : sur seeds qui montent vite + scheduler poussant density ≥ 0.43, risque de crash `RandomObstaclesGenerator` (piège #10 V2-X). Bench BX recommande `--max-attempts-bfs 500`.
5. **Tight coupling potentiel** : la Sonde C accède à la logique BFS et à l'encodeur d'obs ; garder l'oracle BFS dans une fonction isolée et testée, pas dispersé.
6. **ASCII dans help-text / messages** : cohérent piège #8 (Windows cp1252) — pas de γ/β littéraux dans le `--help`, utiliser « gamma »/« beta ».

## 11. Ce qui est explicitement hors-scope

- Toute solution non-oracle (auxiliary head, ICM, attention, n-step en première intention).
- Autres tailles (10×10, 20×20), autres archis (V2-W sans LSTM).
- Combinaisons de sondes en première passe (réserve uniquement si ≥2 sondes frôlent 0.45).
- Reward dense (écarté car confond exploration et horizon).
