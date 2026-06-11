# V2-C0 — RND minimal : exploration intrinsèque apprise (spec design)

> Date : 2026-06-06
> Sous-projet : **C0** (premier pas du programme C — exploration intrinsèque)
> Statut : spec validée en brainstorming, en attente de plan d'implémentation

---

## 1. Contexte & motivation

Le diagnostic **V2-BX** (tag `v0.2.0-bx`) a tranché par sondes-oracles que le bottleneck du régime 15×15 de V2-ZY+Polyak est l'**exploration**, pas la représentation (éliminée : même le champ BFS complet ne dépasse pas la baseline) ni l'horizon (écarté : γ plus long déstabilise). La seule famille au-dessus de la baseline était l'exploration via bonus count-based — mais ce bonus brut :
- **frôle 0.45 sans le franchir** (diff_max moy 0.40 à β=0.02, plafonne) ;
- **corrompt l'objectif à β trop grand** (β=0.1 → winrate 0 %, le bonus écrasait le goal reward ~15:1).

Le count-based était un **oracle de sondage**, pas une solution livrable. V2-C0 le remplace par un mécanisme d'exploration **appris** et **scale-stable** : **RND (Random Network Distillation, Burda et al. 2018)**.

**Pourquoi RND plutôt qu'ICM** : plus simple à isoler, ne dépend pas d'un modèle dynamique correct, bon pour environnements procéduraux, produit un bonus de nouveauté appris plutôt qu'un comptage brut destructeur, et — surtout — la **normalisation par running-std** rend le bonus scale-stable, ce qui répond directement à la leçon centrale de BX : *β est dangereux*.

## 2. Objectif & critères

**Objectif** : un bonus de nouveauté appris peut-il remplacer le count-based oracle sans corrompre le reward extrinsèque, et franchir le plafond que le count-based n'a pas su franchir ?

**Baseline & références mesurées à 15×15** (eval V2-V rigoureux, best @ diff=0.30 greedy, 10 seeds held-out) :
- V2-ZY+Polyak (pas d'exploration) : `diff_max` 0.36, best_eval **64 %**.
- Count-based B@0.02 (le concurrent BX à battre) : `diff_max` 0.40, best_eval **66.7 %**.

**Critères de succès** (bench en deux phases, cf. §7) :
- Phase 1 (n=3) **passe si** : `diff_max` moyen > 0.40 **OU** best_eval moyen > 66.7 % (battre le count-based).
- Phase 2 (n=5) **succès si** : best @ diff=0.30 ≥ **74 %** **OU** `diff_max` moyen > **0.45**.

**Nature** : c'est un **build** (pas un diagnostic), mais avec la discipline A/B de BX : une seule variable changée (`--rnd`), no-op strict par défaut, références déjà mesurées.

## 3. Décision d'architecture : single-stream additif

RND a deux variantes d'intégration. **V2-C0 = single-stream** (décision brainstorming) :

```
reward_total = reward_ext + beta * clip(rnd_bonus_norm)
```

Le bonus intrinsèque normalisé est **ajouté au reward extrinsèque**, exactement au point d'intégration BX (dans `ConvRecurrentProceduralDQNRunner`, après `env.step`, avant `agent.observe`). **L'agent DQN, le trainer et le réseau Q restent INTOUCHÉS** — ils reçoivent juste un reward modifié.

La variante **two-stream** (tête de valeur intrinsèque séparée, discount propre, retours non-épisodiques, façon Burda/PPO) est **hors-scope C0** : elle mélangerait exploration + architecture Q + target learning + nouvelle tête de valeur, rendant l'interprétation A/B floue. Reportée à un éventuel V2-C1+.

**Avantages single-stream** : même point d'intégration que BX, blast radius minimal, A/B propre (drop-in du count-based), opt-in strict, rollback simple.

## 4. Composants & fichiers

| Composant | Fichier | Rôle |
|---|---|---|
| `RNDModule` | `mw_ia/neural/rnd.py` (créer) | RND isolé : 2 réseaux + optimizer + normaliseurs. Possédé par le runner. |
| Réseau `_RNDNet` | idem (interne) | Petit CNN `(3,R,C) → embedding`. Réutilisé pour target (figé) et predictor (entraîné). |
| flags `rnd_*` | `mw_ia/config.py` (`ConvRecurrentDQNConfig`) | 7 champs, défauts no-op. |
| wiring + logging | `mw_ia/training/runner.py` (`ConvRecurrentProceduralDQNRunner` **uniquement**) | Instancie le module, ajoute le bonus, log le ratio. |
| CLI | `scripts/train_cnn_lstm_dqn_procedural.py` | `--rnd/--no-rnd` + hyperparams. |
| smoke CI | `.github/workflows/aether_verify.yml` | `--rnd` seul + `--rnd --polyak-tau 0.005` cohabit. |

### 4.1 `RNDModule` (`mw_ia/neural/rnd.py`)

- **`target`** : `_RNDNet` à init aléatoire **figée** — `requires_grad=False` sur tous ses params, jamais dans un optimizer, jamais entraînée. `(3,R,C) → embedding` (dim `rnd_embed_dim`).
- **`predictor`** : `_RNDNet` (même archi, init indépendante), **entraînée** à imiter `target`. Adam, `lr=rnd_lr`.
- **Normaliseur d'observations** : `RunningMeanStd` (mean/std en ligne) sur les obs d'entrée. Obs normalisée = `clip((obs − mean)/sqrt(var+eps), −5, 5)` (standard Burda).
- **Normaliseur de bonus** : `RunningMeanStd` (ou running-std seul) sur les bonus bruts ; le bonus retourné est divisé par `sqrt(var_bonus + eps)`.
- **`compute_bonus(obs: np.ndarray) -> float`** :
  1. normalise l'obs ;
  2. forward `target` (no_grad) et `predictor` (no_grad pour le calcul du bonus) ;
  3. erreur brute = `mean((predictor(o) − target(o))²)` (MSE sur l'embedding) ;
  4. met à jour le normaliseur de bonus, divise par sa std → `bonus_norm` ;
  5. `clip(bonus_norm, 0, rnd_clip)` ;
  6. pendant le warmup (`step < rnd_warmup_steps`) : met à jour les normaliseurs **mais retourne 0.0**.
- **`update(obs: np.ndarray) -> float`** : un pas de gradient sur le predictor pour minimiser `mean((predictor(o_norm) − target(o_norm).detach())²)`. Retourne la loss (pour logging). Predictor par-step sur l'obs courante (décision brainstorming : option (a), pas de sampling buffer en C0).
- **Architecture `_RNDNet`** : `Conv(3→16, k3, pad1) → ReLU → Conv(16→32, k3, pad1) → ReLU → Flatten → Linear(32·R·C → rnd_embed_dim)`. Petit volontairement (RND n'a pas besoin de gros réseaux).
- L'agent DQN ne connaît pas l'existence de RND.

### 4.2 Flags config (`ConvRecurrentDQNConfig`)

| Champ | Défaut | Validation |
|---|---|---|
| `rnd_enabled: bool` | `False` | — |
| `rnd_beta: float` | `0.5` | `>= 0` |
| `rnd_lr: float` | `1e-4` | `> 0` |
| `rnd_embed_dim: int` | `128` | `> 0` |
| `rnd_clip: float` | `5.0` | `> 0` |
| `rnd_warmup_steps: int` | `1000` | `>= 0` |
| `rnd_ratio_warn: float` | `10.0` | `> 0` |

Défaut `rnd_enabled=False` ⇒ comportement identique à V2-ZY+Polyak (baselines reproductibles bit-à-bit). Validation ASCII (piège #8 Windows cp1252).

### 4.3 Intégration runner (`ConvRecurrentProceduralDQNRunner` uniquement)

- `__init__` : si `dqn_cfg.rnd_enabled`, instancie `self.rnd = RNDModule(...)` (sinon `self.rnd = None`). Init des accumulateurs de ratio (`_rnd_int_sum`, `_rnd_ext_sum` sur fenêtre glissante).
- `run()`, dans la boucle, après `s2, r, terminated, truncated, _ = self.env.step(a)` et **avant** `agent.observe` :
  ```python
  if self.rnd is not None:
      next_obs_rnd = encode_procedural_observation_2d(state=s2, grid=maze, goal=goal,
                                                      max_rows=..., max_cols=...)  # oracle_mode="none"
      bonus = self.rnd.compute_bonus(next_obs_rnd)
      self.rnd.update(next_obs_rnd)
      r_ext = r
      r = r + self.dqn_cfg.rnd_beta * bonus
      # accumuler pour le ratio
      self._rnd_int_sum += self.dqn_cfg.rnd_beta * bonus
      self._rnd_ext_sum += abs(r_ext)
  ```
  (Réutiliser l'`obs`/`next_obs` déjà encodés si disponibles pour éviter un double encodage.)
- Ratio `Σintrinsèque / Σextrinsèque` calculé sur fenêtre glissante, émis dans le log structuré, **WARN** si soutenu > `rnd_ratio_warn`. Run continue, marqué « potentiellement contaminé » (pas d'abort, pas d'auto-correction).
- Ligne de log structurée étendue ou nouvelle ligne `RND_RESULT` portant `rnd_beta`, `ratio_int_ext`, `predictor_loss`, en plus de `diff_max`/`best_eval` déjà présents via `BX_PROBE_RESULT`.

### 4.4 CLI (`train_cnn_lstm_dqn_procedural.py`)

`--rnd / --no-rnd` (BooleanOptionalAction, default False), `--rnd-beta`, `--rnd-lr`, `--rnd-embed-dim`, `--rnd-clip`, `--rnd-warmup-steps`, `--rnd-ratio-warn`. Help-text ASCII. Wirés dans la construction de `ConvRecurrentDQNConfig`.

## 5. Mécanismes anti-β (invariant central BX câblé par construction)

L'invariant — *le bonus intrinsèque ne domine jamais durablement l'extrinsèque* — n'est pas un garde-fou tardif mais une propriété de conception :

1. **Normalisation des obs** : `RunningMeanStd` + clip `[-5,5]`. Empêche les réseaux de voir des échelles dérivantes.
2. **Normalisation du bonus (le mécanisme clé)** : division par la running-std des bonus → le bonus reste **O(1)** quelle que soit l'échelle absolue du MSE (qui décroît à mesure que le predictor apprend). **β n'est plus sur un fil de rasoir** comme le count-based.
3. **Clip per-step** : `clip(bonus_norm, 0, rnd_clip)`. Un état aberrant ne peut pas injecter un bonus géant en un pas.
4. **Warmup** : bonus=0 pendant `rnd_warmup_steps` pendant que les normaliseurs se calent → pas de pic au démarrage.
5. **β faible par défaut (0.5)** : avec la normalisation gardant le bonus O(1) et l'extrinsèque ~O(1) (goal +1), β=0.5 *nudge* sans dominer.
6. **Ratio loggé + WARN** : `Σint/Σext` glissant, émis à chaque log, WARN si soutenu > 10:1. **Jamais d'abort, jamais d'auto-correction** (zéro dynamique cachée → A/B propre). Run marqué potentiellement contaminé.

## 6. Stratégie de test (TDD)

Tests déterministes CPU, test-first par tâche (pattern projet) :

- **`RNDModule`** :
  - bonus ≥ 0 ;
  - **propriété RND centrale** : `bonus(X)` **décroît** après plusieurs `update(X)` répétés (le predictor apprend X → erreur baisse → état moins « nouveau ») ;
  - **target figée** : les params de `target` sont **inchangés** après des `update()` (vérifie qu'elle n'est pas dans l'optimizer) ;
  - clip respecté (bonus ≤ `rnd_clip`) ;
  - warmup : `compute_bonus` retourne 0.0 tant que `step < rnd_warmup_steps`, et la normalisation est mise à jour quand même ;
  - normalisation borne le bonus (pas d'explosion sur obs à grande échelle).
- **config** : 7 flags + validation (β≥0, lr>0, embed>0, clip>0, warmup≥0, ratio_warn>0), défauts no-op.
- **runner** : `rnd_enabled` instancie le module ; bonus ajouté au reward avant `observe` ; ratio loggé ; **no-op strict** quand off (reproductibilité bit-à-bit, A==B comme en BX Task 9).
- **smoke CI** (≤10 ép CPU) : `--rnd` seul + `--rnd --polyak-tau 0.005` cohabit.
- `pytest -q` reste vert (384 baseline + nouveaux tests RND).

## 7. Bench (protocole en deux phases)

Substrat : V2-ZY+Polyak 15×15 (`--max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 --polyak-tau 0.005 --max-attempts-bfs 500`), 5000 ép, eval V2-V (best @ diff=0.30, 10 seeds held-out 10000-10009). Une seule variable changée vs baseline : `--rnd`.

**Phase 1 — n=3 directionnel** (seeds 0,1,2), défauts §4.2 :
```bash
for seed in 0 1 2; do
  python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed $seed \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --max-attempts-bfs 500 \
    --rnd \
    --best-checkpoint-path checkpoints/c0_rnd_seed${seed}.pt
done
```
**Passe si** : `diff_max` moy > 0.40 **OU** best_eval moy > 66.7 % (bat le count-based B@0.02). Sinon : finding négatif/neutre, brainstorm β / variante.

**Phase 2 — n=5 same-seed** (seeds 0-4) si Phase 1 promet. **Succès si** : best @ diff=0.30 ≥ 74 % **OU** `diff_max` moy > 0.45.

**Report systématique** vs les 2 références : baseline (0.36 / 64 %) et count-based B@0.02 (0.40 / 66.7 %).

## 8. Livrable & critères de sortie

- Code : 7 flags off par défaut, `RNDModule` isolé, runner wiring conditionnel.
- Finding : `docs/findings/2026-06-06-v2c0-rnd-results.md` (table n=3 [+n=5], comparaison aux 2 références, ratio int/ext observé, verdict).
- Section CLAUDE.md + mise à jour mémoire.
- Tag `v0.2.0-c0` (posé même sur résultat négatif/neutre, pattern B0/B1a/BX — la livraison code et le finding honnête comptent).

## 9. Risques & pièges

1. **β/normalisation mal réglés = faux négatif** (leçon BX) : la normalisation du bonus est censée rendre β robuste, mais si la running-std est mal initialisée ou le warmup trop court, le bonus peut spiker tôt. Le ratio loggé + WARN sert exactement à détecter ça post-hoc. Un re-réglage rapide (β ou warmup) avant de conclure négatif, comme le re-test β=0.01/0.02 de BX.
2. **Predictor per-step single-sample = update bruité** : accepté en C0 (la normalisation lisse). Le sampling buffer (option b) est la première piste de raffinement si le signal est trop bruité.
3. **Coût compute du double forward RND par step** : 2 petits CNN forward + 1 backward predictor par pas env. Négligeable vs le forward LSTM+Conv de l'agent, mais à surveiller (RND petit volontairement).
4. **`RunningMeanStd` sur Windows / AMP** : garder les normaliseurs en float64 numpy côté CPU (hors graphe torch), pas sous autocast. Les forwards RND restent en float32.
5. **Double encodage de l'obs** : le runner encode déjà `next_obs` pour l'agent ; réutiliser ce tensor pour RND (oracle_mode="none" est le défaut, donc l'obs agent et l'obs RND sont identiques quand le mode oracle BX est off — ce qui est le cas en C0). Éviter un second appel à l'encodeur.
6. **ASCII** : help-text CLI et messages WARN en ASCII (pas de β/τ littéraux), piège #8.

## 10. Hors-scope C0

- Two-stream RND (tête de valeur intrinsèque séparée, retours non-épisodiques) → éventuel V2-C1.
- Sampling du predictor depuis le replay buffer (option b) → raffinement si C0 bruité.
- ICM, planning, pseudo-counts feature-space → autres branches du programme C.
- Annealing de β → la normalisation + clip suffisent en C0 (l'utilisateur a tranché « annealing OU clipping » → clipping).
