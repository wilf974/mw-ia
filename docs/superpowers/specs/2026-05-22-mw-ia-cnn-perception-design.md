# Spec V2-Z — CNN perception spatiale (DQN feedforward)

> **Sous-projet** : V2-Z (MW_IA — Reinforcement Learning éducatif)
> **Date** : 2026-05-22
> **Statut** : Spec validée — implémentation à dérouler via `superpowers:writing-plans` puis `superpowers:subagent-driven-development`
> **Tag livraison cible** : `v0.2.0-z`
> **Sous-projets prérequis livrés** : V1 (`v0.1.0`), V2-A guardrails (`v0.2.0-a`), V2-X procedural env (`v0.2.0-x`), V2-Y DRQN/LSTM (`v0.2.0-y`)

---

## 1. Vue d'ensemble & motivation

### Motivation empirique consolidée (handoff 2026-05-22)

V2-X feedforward et V2-Y LSTM plafonnent tous deux à `diff ≈ 0.05` sur le curriculum procédural. Récap :

| Variante | Final winrate (bucket 0) | Final diff atteinte | Buckets 1+ remplis |
|---|---|---|---|
| V2-X MLP `(256, 256)` consolidé | 72 % | 0.05 | non |
| V2-Y LSTM `fc=256, lstm=128` | 95 % | 0.05 | non |
| V2-Z **CNN** (cible) | **≥ 95 %** | **≥ 0.10** | **bucket 1 ≥ 70 %** |

Le LSTM bat le MLP en qualité de politique (+23 pp winrate au même palier) mais ne franchit pas le palier de difficulté. **Hypothèse V2-Z** : le bottleneck est la représentation spatiale — `concat(position_one_hot, grid_flatten)` (200-dim plat) détruit la structure 2D du maze.

### Pourquoi un CNN pour les mazes

1. **Translation equivariance** : un dead-end ressemble à un dead-end où qu'il soit dans la grille. Le MLP doit ré-apprendre indépendamment chaque position absolue (200 features non-corrélées) ; la conv apprend des filtres locaux qui se déplacent gratis.
2. **Localité** : les actions valides en `(r, c)` dépendent du voisinage 3×3, pas de la coordonnée absolue. Un kernel 3×3 capture exactement ce voisinage.
3. **Partage de poids** : ~10× moins de paramètres pour le bloc d'extraction vs MLP, à capacité expressive comparable. Permet d'allouer plus de paramètres aux couches FC de décision.

### Objectif et critère de succès

**Cible chiffrée** :
1. Final winrate ≥ **95 %** @ diff=0.05 (**match V2-Y**)
2. Scheduler atteint et maintient **diff ≥ 0.10** avec winrate bucket ≥ 70 %

**Pattern d'intégration** : nouveaux fichiers parallèles, V2-X et V2-Y restent intacts (précédent `v0.2.0-x` et `v0.2.0-y` respecté). Tag livraison cible : `v0.2.0-z`.

---

## 2. Composants livrés & arborescence

### Nouveaux fichiers

| Fichier | Rôle |
|---|---|
| `mw_ia/neural/conv_network.py` | `ConvQNetwork` : `Conv(3→32, k=3, pad=1) → ReLU → Conv(32→64, k=3, pad=1) → ReLU → Flatten → Linear(64·R·C → 256) → ReLU → Linear(256 → 4)`. Accepte input tensor `(B, 3, R, C)`. |
| `mw_ia/agents/conv_dqn.py` | `ConvDQNAgent` : wrap `ConvQNetwork` dans le même contrat que `DQNAgent` (`act`, `observe`, `global_step`, `epsilon`, `device`). Conserve `ReplayBuffer` V1. Reshape `(B, 3, R, C)` au moment du `train()`. |
| `scripts/train_cnn_dqn_procedural.py` | CLI parallèle à `train_dqn_procedural.py` / `train_drqn_procedural.py`. Flags : `--conv-channels`, `--fc-hidden`, `--epsilon-decay-steps`, `--scheduler-update-interval`, `--scheduler-step`, `--target-sync-steps`. |
| `tests/neural/test_conv_network.py` | Tests du réseau |
| `tests/envs/test_procedural_env_2d.py` | Tests de l'encoder 2D |
| `tests/agents/test_conv_dqn.py` | Tests de l'agent |
| `tests/training/test_conv_procedural_runner.py` | Tests d'intégration du runner |
| `tests/test_conv_dqn_config.py` | Tests de la config |

### Extensions de fichiers existants

| Fichier | Extension |
|---|---|
| `mw_ia/envs/procedural_env.py` | `+ encode_procedural_observation_2d(state, grid, goal, max_rows, max_cols) → np.ndarray[float32]` shape `(3, max_rows, max_cols)`. V2-X `encode_procedural_observation` 1D **conservé**. |
| `mw_ia/training/runner.py` | `+ class ConvProceduralDQNRunner`. Parallèle à `ProceduralDQNRunner` (V2-X) et `RecurrentProceduralDQNRunner` (V2-Y). Defaults scheduler V2-X (`update=200`, `step=0.05`). |
| `mw_ia/config.py` | `+ class ConvDQNConfig` frozen dataclass : champs nouveaux `conv_channels=(32, 64)`, `kernel_size=3`, `padding=1`, `fc_hidden=256` + duplication explicite des champs partagés `DQNConfig` (gamma, lr, epsilon_*, replay_capacity, target_sync_steps, batch_size, episodes, max_steps_per_episode, min_episodes_to_learn). Pas d'héritage Python — duplication assumée pour rester frozen et explicit, cohérent V2-X `ProceduralEnvConfig` et V2-Y `DRQNConfig`. |
| `mw_ia/gui/widgets/control_panel.py` | `+ bouton "Démarrer (procedural CNN)"` `+ signal start_procedural_cnn_requested`. |
| `mw_ia/gui/app.py` | `+ slot on_start_procedural_cnn` qui lance `ConvProceduralDQNRunner` dans un QThread. Signaux GUI (`maze_changed`, `difficulty_updated`) inchangés. |
| `.github/workflows/aether_verify.yml` | `+ smoke train_cnn_dqn_procedural --episodes 10 --device cpu`. |
| `README.md` | `+ section V2-Z` (motivation, archi, recette gagnante, comparaison). |
| `CLAUDE.md` | `+ section V2-Z` (statut phases, composants, décisions, pièges, recette opérationnelle). |

### Décisions d'intégration

- **Encoder** : `encode_procedural_observation_2d` placé dans `procedural_env.py` aux côtés de la version 1D V2-X. Convention : suffixe `_2d`, pas de remplacement.
- **Agent** : nouveau `ConvDQNAgent` plutôt que paramétriser `DQNAgent`. Raison : `DQNAgent` V1 reçoit obs 1D via `_make_agent` hack `_ObsDimEnv`, refactor casserait V1 et V2-X.
- **Buffer** : pas de nouveau `SpatialReplayBuffer`. Le `ReplayBuffer` V1 stocke des `np.ndarray` 1D — on flatten les obs 3D avant `push`, on reshape `(B, 3, R, C)` après `sample`. Évite la duplication. Documenté dans `ConvDQNAgent`.

### Arborescence après V2-Z (sections nouvelles uniquement)

```
mw_ia/neural/
└── conv_network.py             # [V2-Z] ConvQNetwork
mw_ia/agents/
└── conv_dqn.py                 # [V2-Z] ConvDQNAgent
scripts/
└── train_cnn_dqn_procedural.py # [V2-Z]
tests/
├── neural/test_conv_network.py            # [V2-Z]
├── envs/test_procedural_env_2d.py         # [V2-Z]
├── agents/test_conv_dqn.py                # [V2-Z]
├── training/test_conv_procedural_runner.py # [V2-Z]
└── test_conv_dqn_config.py                # [V2-Z]
```

---

## 3. Data flow détaillé

Trace complète d'un step `t` dans `ConvProceduralDQNRunner.run()`, shapes pour `max_rows=max_cols=10`, `n_actions=4`, `batch_size=128` :

```
1. env.reset(seed=ep)
   → state = (row, col)            tuple[int, int]
   → info["maze"] = grid           np.ndarray[bool], shape (10, 10)
   → goal = (rows-1, cols-1)       lu depuis env.inner.cfg.goal

2. encode_procedural_observation_2d(state, grid, goal, max_rows=10, max_cols=10)
   → obs                           np.ndarray[float32], shape (3, 10, 10)
     - obs[0]  = position one-hot          (10, 10), un seul 1 en (row, col)
     - obs[1]  = obstacles                  (10, 10), grid.astype(float32)
     - obs[2]  = goal one-hot               (10, 10), un seul 1 en (goal_r, goal_c)
     padding top-left zéros si maze < max_size (cellules libres)

3. agent.act(obs)
   - epsilon-greedy : si random < epsilon → np.random.randint(4)
   - sinon → forward :
     - obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)   shape (1, 3, 10, 10)
     - q     = conv_qnet(obs_t)                                shape (1, 4)
     - action = q.argmax(dim=1).item()                         int

4. env.step(action)
   → next_state, reward, terminated, truncated, info

5. next_obs = encode_procedural_observation_2d(next_state, grid, goal, 10, 10)
   shape (3, 10, 10)

6. agent.observe(obs, action, reward, next_obs, done)
   - replay_buffer.push(obs.flatten(), action, reward, next_obs.flatten(), done)
     ↑ flatten (3*10*10 = 300) pour réutiliser ReplayBuffer V1 inchangé
   - if buffer.size >= max(min_to_learn, batch_size) and global_step % train_every == 0:
       train_step()  → metric loss
   - if global_step % target_sync_steps == 0: copy online → target
   - global_step += 1
   - epsilon = linear_schedule(global_step)

7. train_step() interne :
   - batch = replay_buffer.sample(128)
     → states_flat   torch.float32  (128, 300)
     → actions       torch.int64    (128,)
     → rewards       torch.float32  (128,)
     → next_flat     torch.float32  (128, 300)
     → dones         torch.bool     (128,)
   - states     = states_flat.view(128, 3, 10, 10)
   - next_s     = next_flat.view(128, 3, 10, 10)
   - q_online   = conv_qnet(states)            (128, 4)
   - q_taken    = q_online.gather(1, actions)  (128, 1)
   - with torch.no_grad():
       q_target = target_qnet(next_s).max(1).values   (128,)
       y        = rewards + γ * q_target * (~dones)   (128,)
   - loss = huber(q_taken.squeeze(-1), y)
   - optimizer + grad clip + AMP (cohérent V1)
```

### Shape interne CNN (forward pass)

```
input              (B, 3, 10, 10)
Conv1 (3→32, k=3, pad=1)  →  (B, 32, 10, 10)   ReLU
Conv2 (32→64, k=3, pad=1) →  (B, 64, 10, 10)   ReLU
Flatten            →  (B, 6400)
FC1 (6400→256)     →  (B, 256)                  ReLU
FC2 (256→4)        →  (B, 4)                     Q-values
```

Params count :
- `Conv1 = 3*32*3*3 + 32 = 896`
- `Conv2 = 32*64*3*3 + 64 = 18 496`
- `FC1   = 6400*256 + 256 = 1 638 656`
- `FC2   = 256*4 + 4 = 1 028`
- **Total ≈ 1.66 M paramètres** (comparable au MLP V2-X `(256, 256)` à 120 K — ~14× plus, mais 99 % concentrés dans FC1).

### Notes performance

- Le flatten + reshape entre buffer et forward ajoute 2 ops vectorielles par step de train. Négligeable (300 floats × 128 batch = 38 K floats).
- Conv2d supporte AMP nativement sur Ampere (RTX 3060). Garder `torch.cuda.amp.autocast()` et `GradScaler` cohérent V1.

---

## 4. Error handling, edge cases & invariants Aether

### Edge cases d'encoding 2D

| Cas | Comportement |
|---|---|
| Maze < max_size | Padding zéros top-left (cellules libres). L'agent voit des bordures artificielles qu'il apprend à ignorer (cohérent V2-X). |
| `state` hors grille | `assert 0 <= state[0] < grid.shape[0] and 0 <= state[1] < grid.shape[1]`. Erreur tôt, message explicite. |
| `goal` hors `max_rows × max_cols` | `assert 0 <= goal[0] < max_rows and 0 <= goal[1] < max_cols`. Évite IndexError silencieux. |
| `grid.shape > (max_rows, max_cols)` | `assert grid.shape[0] <= max_rows and grid.shape[1] <= max_cols`. |
| `grid.dtype != bool` | Acceptée si convertible (cohérent V2-X via `.astype(float32)`). |

### Edge cases agent/buffer

| Cas | Comportement |
|---|---|
| `batch_size > buffer.size` | Skip `train_step()` — garde `len(buffer) >= max(min_episodes_to_learn, batch_size)` (fix V2-Y appliqué dès V2-Z). |
| Reshape buffer flat → (B, 3, R, C) | `assert obs.shape == (3, max_rows, max_cols)` dans `ConvDQNAgent.observe()`. |
| AMP NaN/Inf dans la loss | `loss.isfinite()` check avant `backward()`, log warning + skip step (cohérent V1). |
| Target sync au step 0 | Init : `target_qnet.load_state_dict(online_qnet.state_dict())` dans `__init__`. |
| GPU absent | `torch.cuda.is_available()` check, fallback CPU avec log warning (cohérent V1). |
| VRAM OOM | Pas de garde explicite (1.66 M × 128 ≈ 50 MB GPU, marge confortable sur 12 GB). Descendre `batch_size` via CLI si observé. |

### Compatibilité Aether (V2-A) — invariants I1-I8

Les 8 invariants I1-I8 sont **architecture-agnostic** et restent valides sans modification :

| Invariant | Compatibilité V2-Z |
|---|---|
| I1 `gamma_in_open_unit` | γ vient de `ConvDQNConfig.gamma`, même contrainte (0, 1) |
| I2 `bellman_contraction` | Opérateur Bellman = γ × max Q_target, indépendant de la forme du réseau |
| I3 `huber_nonneg` | Huber loss inchangée |
| I4 `winrate_bounds` | `MetricsTracker.winrate()` partagé V1/V2-X/V2-Y/V2-Z |
| I5 `epsilon_schedule_decreasing` | `linear_schedule` partagé, ε ∈ [0, 1] décroissant |
| I6 `replay_buffer_capacity` | `ReplayBuffer` V1 réutilisé sans modif |
| I7 `reward_bounded` | Récompenses `GridWorld` V1 inchangées |
| I8 `episode_termination_exclusive` | `GridWorld.step()` V1 inchangé |

`VariantSpec` V2-A consomme `ConvDQNConfig` **sans aucune extension de schéma**. Les champs nouveaux (`conv_channels`, `kernel_size`, `padding`, `fc_hidden`) ne sont pas dans `VariantSpec` car ils n'affectent pas les invariants formels — uniquement la performance empirique.

Smoke test E2E V2-Z (à inclure dans `test_conv_dqn_config.py`) :

```python
from mw_ia.config import ConvDQNConfig
from mw_ia.guardrails import VariantSpec, verify_formal
cfg = ConvDQNConfig()
spec = VariantSpec(
    gamma=cfg.gamma, lr=cfg.lr,
    epsilon_start=cfg.epsilon_start, epsilon_end=cfg.epsilon_end,
    epsilon_decay_steps=cfg.epsilon_decay_steps,
    batch_size=cfg.batch_size,
    replay_capacity=cfg.replay_capacity,
    target_sync_steps=cfg.target_sync_steps,
)
assert verify_formal(spec).passed
```

---

## 5. Stratégie de testing (TDD bite-sized par phase)

### Pattern hérité V2-A/V2-X/V2-Y

- Un fichier de test par composant
- Tests rouges écrits avant impl
- Fixtures réutilisables (`conftest.py` par dossier)
- Pas de mocks lourds — on teste les vrais tenseurs PyTorch sur CPU
- Reuse des fixtures existantes : `rng` (`tests/envs/conftest.py`), `cpu_device` (`tests/neural/conftest.py`)

### Phases TDD prévisionnelles

| Phase | Composant | Fichier de test | Tests | Cumulés |
|---|---|---|---|---|
| 1 | Setup scaffold (`neural/conv_network.py`, `agents/conv_dqn.py`, skeletons tests) | — | 0 | 183 (baseline) |
| 2 | `encode_procedural_observation_2d` | `tests/envs/test_procedural_env_2d.py` | 6 | 189 |
| 3 | `ConvQNetwork` (forward 1+batch, params count, gradient flow, state_dict compat) | `tests/neural/test_conv_network.py` | 5 | 194 |
| 4 | `ConvDQNConfig` (defaults, validation `__post_init__`, conv_channels variants) | `tests/test_conv_dqn_config.py` | 4 | 198 |
| 5 | `ConvDQNAgent` (init, act eps-greedy, observe push, target sync, train trigger, finite loss, smoke E2E Aether) | `tests/agents/test_conv_dqn.py` | 7 | 205 |
| 6 | `ConvProceduralDQNRunner` (1 épisode E2E, scheduler/bucket tracker hooks, `maze_changed` callback, smoke 10 ép sans NaN) | `tests/training/test_conv_procedural_runner.py` | 3 | 208 |
| 7 | CLI `train_cnn_dqn_procedural.py` + CI smoke + GUI button | (CI workflow) | 0 (CI) | 208 |
| 8 | README V2-Z + DoD + tag `v0.2.0-z` | — | 0 | 208 |

**Total ajouté : 25 tests** (183 → 208 attendu).

### Détail des tests par phase

**Phase 2 — `test_procedural_env_2d.py`** (6 tests) :
- `test_encode_shape_default` : `(3, 10, 10)` float32
- `test_encode_agent_channel` : un seul 1 en `(row, col)`, zéros ailleurs
- `test_encode_obstacles_channel` : matches `grid.astype(float32)`
- `test_encode_goal_channel` : un seul 1 en `(goal_r, goal_c)`
- `test_encode_padding_smaller_maze` : maze 6×6 dans max 10×10, zéros padding top-left
- `test_encode_asserts_invalid_inputs` : state hors grille / grid > max / goal hors max → `AssertionError`

**Phase 3 — `test_conv_network.py`** (5 tests) :
- `test_forward_single_sample` : input `(1, 3, 10, 10)` → output `(1, 4)`
- `test_forward_batch` : input `(32, 3, 10, 10)` → output `(32, 4)`
- `test_params_count` : assert total params ≈ 1.66M (tolerance ±5 %)
- `test_gradient_flow` : `loss.backward()` produit des grads non-nulls sur toutes les couches
- `test_state_dict_compat` : `target.load_state_dict(online.state_dict())` round-trip exact

**Phase 4 — `test_conv_dqn_config.py`** (4 tests) :
- `test_defaults` : `conv_channels=(32, 64)`, `kernel_size=3`, `padding=1`, `fc_hidden=256`
- `test_post_init_validation_positive` : channels/kernel/fc_hidden > 0
- `test_conv_channels_arbitrary_length` : `(16,)` ou `(32, 64, 128)` acceptés
- `test_aether_compat` : `verify_formal(VariantSpec.from(ConvDQNConfig())).passed == True`

**Phase 5 — `test_conv_dqn.py`** (7 tests) :
- `test_init` : online + target nets, buffer empty, optimizer Adam, global_step=0
- `test_act_random_when_eps_high` : seed fixe, eps=1.0 → action ∈ {0,1,2,3}
- `test_act_greedy_when_eps_zero` : eps=0 → action = argmax Q-values
- `test_observe_pushes_buffer` : 1 transition → buffer.size = 1
- `test_target_sync` : après `target_sync_steps` updates, target_qnet == online_qnet
- `test_train_trigger_min_episodes` : pas de train_step avant `len(buffer) >= max(min, batch_size)`
- `test_train_step_returns_finite_loss` : après buffer rempli, train_step retourne loss > 0 et finie

**Phase 6 — `test_conv_procedural_runner.py`** (3 tests) :
- `test_single_episode_runs` : 1 épisode sans crash, metrics récoltés
- `test_callbacks_fired` : `on_maze_changed`, `on_episode`, `on_step`, `on_loss` tous appelés
- `test_smoke_10_episodes_no_nan` : 10 ép, `metrics.losses` tous finis, winrate ∈ [0, 1]

### CI smoke

Ajout à `.github/workflows/aether_verify.yml` :

```yaml
- name: Smoke train_cnn_dqn_procedural
  run: python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode obstacles --device cpu
```

10 épisodes CPU pour rester sous 30s en CI. Pas d'assertion de winrate (10 ép trop court), juste "no crash".

### Property-based via hypothesis

Pas prévu en V2-Z MVP (pattern V2-A déjà couvre les invariants Aether). Possible extension future pour `encode_procedural_observation_2d` (channels well-formed pour tout maze valide).

---

## 6. DoD, pièges anticipés & critères de livraison `v0.2.0-z`

### Definition of Done

1. Tous les composants livrés (`ConvQNetwork`, `encode_procedural_observation_2d`, `ConvDQNConfig`, `ConvDQNAgent`, `ConvProceduralDQNRunner`, CLI, GUI button)
2. `pytest -q` → **208 passed** (183 baseline + 25 V2-Z)
3. `bash aether/verify_all.sh` → 8 OK (inchangé)
4. CI workflow `aether_verify.yml` étendu avec smoke `train_cnn_dqn_procedural.py --episodes 10 --device cpu`, vert sur push
5. Smoke E2E manuel sur RTX 3060 : `python scripts/train_cnn_dqn_procedural.py --episodes 50 --device cuda` → pas de crash, winrate > 0 %, loss finite
6. Smoke GUI manuel : bouton "Démarrer (procedural CNN)" lance le runner, fenêtre se rafraîchit, courbe difficulty oscille
7. **Critère succès empirique** (2 runs reproductibles 5000 ép GPU) :
   - Final winrate ≥ **95 % @ diff=0.05** (match V2-Y)
   - Scheduler atteint et maintient **diff ≥ 0.10** avec winrate bucket ≥ 70 %
8. Section V2-Z complète dans `README.md` (motivation, archi, recette gagnante consolidée, comparaison V2-X / V2-Y / V2-Z)
9. Section V2-Z complète dans `CLAUDE.md` (statut phases, composants, décisions techniques, pièges, recette opérationnelle)
10. Tag `v0.2.0-z` posé sur le commit final
11. Tag `v0.2.0-y` et antérieurs intacts (pas de force-push, pas de modification rétroactive)

**Note** : DoD #1-#10 sont bloquants pour le tag. DoD #7 est l'**objectif scientifique** mais **ne bloque PAS** la livraison du tag — pattern V2-X/V2-Y : on livre le MVP, on itère ensuite sur les hyperparams. Documenter chaque tentative dans `CLAUDE.md`.

### Pièges anticipés (à confirmer empiriquement)

| # | Piège | Anticipation |
|---|---|---|
| 1 | **Padding zéros = bordure artificielle** | L'encoder padde top-left avec zéros sur les 3 canaux (cellules libres, pas d'obstacle, pas de goal). Pour 10×10 fixe c'est sans effet (le maze remplit toute la zone). Si on varie la taille du maze, le CNN voit une zone "vide" en bas-droite — risque qu'il apprenne à confondre "padding" et "couloirs ouverts". Mitigation : ajouter un 4ᵉ canal "valid region mask" si ce piège se matérialise. Pas en MVP. |
| 2 | **`conv_channels=(32, 64)` overkill** | 1.66 M params dont 99 % dans FC1. Possible que `(16, 32)` ou `(8, 16)` suffise. À itérer après MVP. |
| 3 | **`scheduler_update=200` trop patient pour CNN** | Pattern V2-Y : LSTM nécessite update=50. CNN feedforward devrait théoriquement matcher V2-X (200), à vérifier empiriquement. CLI flag exposé. |
| 4 | **`target_sync_steps=1000`** | Possible que CNN nécessite sync plus fréquent. À itérer. |
| 5 | **VRAM si on monte à `max_size=20`** | `FC1 = (20*20*64) * 256 ≈ 6.5 M params` (vs 1.64 M pour 10×10). Reste OK sur 12 GB, mais penser à `nn.AdaptiveAvgPool2d` ou stride=2 si on va plus large (32×32+). Pas en MVP. |
| 6 | **Reshape buffer flat → 3D collision** | Si `obs_dim` du buffer (300) ≠ `3 * R * C` au reshape, erreur runtime peu claire. Garde `assert obs.shape == (3, R, C)` dans `ConvDQNAgent.observe()`. |
| 7 | **Hypothèse : peut-être que ça ne suffit pas** | Bottleneck pourrait être Q-values instables (V2-Y a montré que LSTM bat MLP sans franchir le plafond). Si V2-Z match V2-Y sans franchir diff=0.10 → escalade vers **V2-W Double DQN**. |

### Validation empirique post-livraison

Pattern V2-Y consolidé : **2 runs de 5000 épisodes GPU** avec defaults consolidés. Si la session #1 montre que CNN ne franchit pas diff=0.10, itérer sur :

1. `--conv-channels 16 32` (plus léger)
2. `--scheduler-update-interval 100` (intermédiaire entre V2-X 200 et V2-Y 50)
3. `--epsilon-decay-steps 300000` (plus de temps d'exploration)
4. `--target-sync-steps 500` (sync plus fréquent)

Documenter chaque tentative dans `CLAUDE.md` (pattern section "V2-X — recette opérationnelle").

### Reproductibilité

2 entraînements avec seeds différentes doivent donner des courbes "qualitativement comparables" :
- Final winrate ± 5 pp
- Diff finale ± 0.025

Documenter les 2 runs dans la section CLAUDE.md V2-Z.

### Sous-projets V3+ déblocables

**Si V2-Z atteint le critère succès** :
- Combinaison V2-Z + V2-Y → **CNN-LSTM** (DRQN à perception spatiale)
- Combinaison V2-Z + Double DQN → **V2-W intégré**
- Mazes plus larges (`max_size=15` ou `20`) → vrai test de la translation equivariance

**Si V2-Z plafonne aussi à diff=0.05** :
- **V2-W Double DQN** devient prioritaire (réduction surestimation Q-values, autre levier majeur)
- Indique que le bottleneck n'est pas la représentation mais l'objectif d'apprentissage lui-même

---

## Annexe — récap des décisions clés

| Décision | Choix | Raison |
|---|---|---|
| Scope | MVP CNN feedforward only | Isole l'effet de la représentation spatiale ; comparable directement à V2-X consolidé |
| Canaux input | 3 (agent + obstacles + goal) | Goal explicite → robuste si on varie taille/position ; standard DeepMind |
| Profondeur conv | 2 conv + 2 FC, pas de pooling | Préserve l'info spatiale sur 10×10 ; ~1.66 M params, marge GPU |
| Critère succès | Match V2-Y @ diff=0.05 + franchir diff=0.10 | Réaliste, mesure 2 axes : qualité politique + capacité curriculum |
| Intégration GUI | Bouton "Démarrer (procedural CNN)" | Permet comparaison interactive V2-X vs V2-Z |
| Pattern fichiers | Parallèles aux V2-X/V2-Y existants | V2-X et V2-Y restent intacts (pattern v0.2.0-x/y respecté) |
| Buffer | `ReplayBuffer` V1 réutilisé (flatten/reshape) | Évite duplication d'un `SpatialReplayBuffer` |
| Scheduler defaults | V2-X (`update=200`, `step=0.05`) | CNN feedforward → cohérent V2-X, pas V2-Y LSTM |

---

**Fin de la spec V2-Z.**
