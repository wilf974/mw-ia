# V2-B1a — Policy Snapshot Rehearsal pour V2-ZY+Polyak

**Date** : 2026-05-29
**Statut** : Spec — design validé en brainstorming, en attente review user finale avant writing-plans
**Position programme V2** : Sous-projet B, phase 1a (rehearsal frozen — variante MVP de B1)
**Dépendances livrées** : V2-Z, V2-Y, V2-W, V2-V (best-checkpoint), V2-ZY, V2-U (Polyak), V2-B0 (PER + finding phase-dépendance)
**Cible empirique** : 15×15 V2-ZY+Polyak où V2-B0+PER a régressé (mean 64 % → 46 %)

---

## 1. Contexte & motivation

### 1.1 État du programme V2 post-B0

V2-B0 a livré un finding scientifique non-trivial : **PER trajectory-level présente une dépendance de phase au régime d'apprentissage**.

- 10×10 (régime saturé baseline 92 %) : PER AIDE significativement (mean +6 pp, std /3, worst +20 pp)
- 15×15 (régime en croissance baseline 64 %) : PER DÉGRADE significativement (mean −18 pp, min −20 pp)

Mécanisme conjecturé : PER amplifie ce qui existe déjà dans le signal d'apprentissage. En régime saturé, les bonnes politiques existent presque-stables → consolidation. En régime exploratoire, les Q-values sont bruitées → sur-amplification du bruit.

Implication : **PER n'est pas le bottleneck principal du régime V2-ZY+Polyak en croissance**. Le bottleneck réel reste celui identifié par V2-V : *"le meilleur agent existait avant ep 3500 ; l'entraînement après le détruit"*. C'est un problème de **conservation / forgetting**, pas de sampling.

### 1.2 Architectural invariant central

> **Une fois capturée, une trajectoire snapshot N'EST JAMAIS modifiée, ré-évaluée, ré-encodée, ou ré-rolled-out. Elle reste un témoin frozen de la politique au pic.**

Cet invariant définit formellement ce qu'est le rehearsal dans B1a — distinct de :
- Re-rollout dynamique (B1 option 1, rejetée en brainstorming pour confusion causale)
- Distillation dynamique (B1b, hors-scope MVP)
- KL constraint (B1c, hors-scope MVP)

L'invariant est ancré à 3 niveaux : spec (cette section), code (docstring `SnapshotTrajectoryStore`), tests (`test_immutability_after_capture`).

### 1.3 Décomposition B / position de B1a

| Sous-projet | Statut | Hypothèse testée | Mécanisme |
|---|---|---|---|
| **B0** | ✅ livré (tag `v0.2.0-b0`) | Le replay uniforme est-il un bottleneck ? | PER trajectory-level (Schaul 2015 + R2D2) |
| **B1a** (ce spec) | en cours | Préserver les trajectoires near-frontier suffit-il à éviter le forgetting ? | Snapshot frozen + mix uniforme dans batch |
| B1b (futur) | reportable | Distillation dynamique snapshot → courant | Snapshot policy génère advice trajectories |
| B1c (futur) | reportable | KL constraint policy courante / snapshot | Régularisation TRPO/PPO-style |
| B2 (futur) | reportable | Rappel conditionnel par contexte | Episodic memory + retrieval policy |

B1a est le MVP : variante la plus simple de B1, reste dans paradigme DQN, isole proprement l'hypothèse forgetting-preservation.

---

## 2. Hypothèse & question scientifique

### 2.1 Hypothèse principale

> Si on capture, au moment du best detected par V2-V, 50 trajectoires successful récentes du buffer, et qu'on les conserve dans un stock séparé frozen avec sliding window des 3 derniers bests, puis qu'on mixe 20 % de ces trajectoires dans chaque batch d'entraînement, alors la politique courante préservera l'accès aux régimes utiles découverts antérieurement et le mean best @ diff=0.30 en 15×15 sera au moins maintenu (no-regression) voire amélioré au-delà de la baseline V2-U.

### 2.2 Hypothèse secondaire (interaction PER × B1a)

> Si B1a stabilise la politique courante autour de bons régimes, alors PER (qui dégradait en régime exploratoire) devrait passer de sur-amplificateur de bruit à amplificateur de signal — *phase-dépendance prédite par V2-B0*. Le bras V2-B1a+PER devrait au moins récupérer 10 pp de la régression de V2-B0 PER seul.

### 2.3 Régime cible

V2-ZY+Polyak (CNN + LSTM + Double DQN + Polyak τ=0.005) à 15×15, où V2-B0 PER avait régressé. C'est le régime où la lecture phase-dépendance prédit que B1a et PER+B1a devraient se distinguer le plus.

### 2.4 Critère scientifique de qualité

Pattern V2-U / V2-B0 strict :
- Same-seed n=5 (seeds 0-4)
- Eval rigoureux V2-V (best @ diff=0.30 fixe greedy, 10 seeds held-out)
- Pré-enregistrement des critères d'acceptance (section 9)
- Bench factoriel 2×2 (PER × B1a) pour analyse marginale et interaction

---

## 3. Périmètre

### 3.1 In-scope

- Implémentation `SnapshotTrajectoryStore` (sliding window N=3 × snapshot_size=50, filtre succès strict, sample uniforme, immutable)
- Helper `concat_batchseq` dans `mw_ia/neural/sequence_buffer.py`
- Hook `agent.on_new_best()` dans `RecurrentDQNAgent` (V2-Y) et `ConvRecurrentDQNAgent` (V2-ZY)
- Sample-time mix 80/20 dans `end_episode()` (4 combinaisons PER × B1a)
- Extension `ConvRecurrentProceduralDQNRunner` : appel `agent.on_new_best()` après `best_tracker.update()` retourne True
- 4 nouveaux champs config (`b1a_enabled`, `b1a_snapshot_size`, `b1a_n_windows`, `b1a_mix_ratio`) × DRQNConfig + ConvRecurrentDQNConfig
- 4 flags CLI × 2 scripts
- 2 smoke CI (`--episodes 30` pour exercer la capture)
- Bench n=5 same-seed bras 3 (B1a seul) + bras 4 (B1a+PER) sur 15×15
- Documentation README + CLAUDE.md + tag `v0.2.0-b1a`

### 3.2 Out-of-scope

- **B1b distillation** : snapshot policy génère advice trajectories
- **B1c KL constraint** : régularisation policy courante / snapshot
- **B2 episodic memory** : retrieval conditionnel par contexte
- **PER sur le snapshot store** : sample uniforme strict (degré de liberté supplémentaire en B2 ou B0.1)
- **Backport V2-V au runner V2-Y** : hors-scope, asymétrie acceptée (cf. section 5.5)
- **Curriculum-conditioned rehearsal** : pas de filtrage par difficulté de capture, sample uniforme
- **Snapshot refresh hors best detection** : timer-based, episode-based, reward-based — toutes rejetées
- **Re-rollout policy snapshot** : viole l'invariant 1.2

---

## 4. Architecture

### 4.1 Vue d'ensemble

```
                  ┌──────────────────────────────────────────────┐
                  │  ConvRecurrentDQNAgent (V2-ZY+Polyak)        │
                  │  ┌────────────────────────────────────────┐  │
                  │  │ self.buffer (Sequence or PER buffer)   │  │
                  │  │ self.snapshot_store (if b1a_enabled)   │──┼─→ SnapshotTrajectoryStore
                  │  │       └─ sliding window N=3 × 50 traj  │  │     - immutable after capture
                  │  │       └─ filter: terminated AND r>0    │  │     - uniform sample
                  │  │ self._beta_scheduler (if per_enabled)  │  │
                  │  └────────────────────────────────────────┘  │
                  │                                              │
                  │  end_episode() :                             │
                  │    main_B = B × (1 − mix_ratio)              │
                  │    snapshot_B = B × mix_ratio                │
                  │    main_batch = self.buffer.sample(...)      │
                  │    if b1a_active:                            │
                  │      snap_batch = snapshot_store.sample(...) │
                  │      batch = concat(main, snap)              │
                  │      weights = concat(per_w, ones) (if PER)  │
                  │    trainer.step_with_priorities(batch, w)    │
                  │    buffer.update_priorities(idx, td[:main_B])│ ◀── only main portion
                  │                                              │
                  │  on_new_best() :  ◀──── called by runner     │
                  │    if b1a_enabled :                          │
                  │      n = snapshot_store.capture_from(buffer) │
                  └──────────────────────────────────────────────┘
                                         ▲
                                         │
                  ┌──────────────────────┴───────────────────────┐
                  │  ConvRecurrentProceduralDQNRunner            │
                  │    eval @ eval_every_episodes :              │
                  │      m = evaluator.evaluate(agent)           │
                  │      if best_tracker.update(m, agent, ep) :  │
                  │        agent.on_new_best()  ◀── NEW B1a hook │
                  └──────────────────────────────────────────────┘
```

### 4.2 Inventaire fichiers

| Type | Fichier | LOC nettes estimées |
|---|---|---|
| Nouveau | `mw_ia/training/snapshot_store.py` | ~120 |
| Modifié | `mw_ia/neural/sequence_buffer.py` | +20 (`concat_batchseq`) |
| Modifié | `mw_ia/config.py` | +24 (4 champs × 2 dataclasses + validation) |
| Modifié | `mw_ia/agents/recurrent_dqn.py` | +60 |
| Modifié | `mw_ia/agents/conv_recurrent_dqn.py` | +60 |
| Modifié | `mw_ia/training/runner.py` | +6 (hook `on_new_best()`) |
| Modifié | `scripts/train_drqn_procedural.py` | +15 |
| Modifié | `scripts/train_cnn_lstm_dqn_procedural.py` | +15 |
| Modifié | `.github/workflows/aether_verify.yml` | +20 (2 smoke jobs `--episodes 30`) |
| Nouveau | `tests/training/test_snapshot_store.py` | ~120 (12 tests) |
| Nouveau | `tests/agents/test_b1a_recurrent_agents.py` | ~120 (14 cases parametrized) |
| Modifié | `tests/neural/test_prioritized_sequence_buffer.py` | +30 (4 tests validation) |

**Total** : ~340 LOC code + ~270 LOC tests = ~610 LOC, plus 30 nouveaux tests pytest (323 → 353).

### 4.3 Principes directeurs

1. **Stock séparé strictement** : le snapshot store n'écrit JAMAIS dans le main buffer. Séparation physique, pas logique.
2. **Frozen entre captures** : aucune ré-évaluation, ré-encodage, ré-rollout des trajectoires snapshot (invariant 1.2).
3. **Filtre succès = `terminated AND total_reward > 0`** : épuré, pas heuristique.
4. **Capture déclenchée par BestCheckpointTracker** : événement de capacité validée, pas timer.
5. **`per_enabled` × `b1a_enabled` orthogonaux** : 4 combinaisons supportées sans condition spéciale.

---

## 5. Composants détaillés

### 5.1 `SnapshotTrajectoryStore`

Interface complète :

```python
class SnapshotTrajectoryStore:
    """Sliding window N captures × snapshot_size trajectoires. Immutable after capture.

    Invariant architectural :
        Une fois captures, une trajectoire snapshot N'EST JAMAIS modifiée,
        re-evaluee, re-encodee, ou re-rolled-out. Elle reste un temoin frozen
        de la politique au pic.

    Sliding window : à la (N+1)e capture, la fenêtre la plus ancienne est écrasée
    (FIFO). Pour N=3 et snapshot_size=50, max 150 trajectoires en stock total.
    """

    def __init__(
        self,
        obs_dim: int,
        max_steps: int = 200,
        *,
        n_windows: int = 3,
        snapshot_size: int = 50,
        seed: int = 0,
    ) -> None: ...

    def __len__(self) -> int:
        """Nombre de trajectoires actuellement stockées (≤ n_windows × snapshot_size)."""

    @property
    def n_captures(self) -> int:
        """Nombre total de captures effectuées (informational, peut dépasser n_windows)."""

    def capture_from(
        self,
        source_buffer: SequenceReplayBuffer | PrioritizedSequenceReplayBuffer,
    ) -> int:
        """Extrait jusqu'à snapshot_size trajectoires successful récentes.

        Filtre succès strict :
            terminated_last_step AND sum(rewards) > 0

        Itère le source_buffer en arrière depuis current_idx. Collecte les premières
        snapshot_size trajectoires qui passent le filtre.

        Storage : remplit la prochaine window slot, ou écrase la plus ancienne si
        sliding window plein (FIFO).

        Returns: nombre de trajectoires effectivement capturées.
        """

    def sample(self, batch_size: int, seq_len: int) -> BatchSeq:
        """Sample uniforme parmi toutes les trajectoires snapshot stockées.

        Returns BatchSeq compatible V2-Y. Raises ValueError si len(self) < batch_size.
        """
```

**Storage interne** : arrays pre-allocated `(n_windows × snapshot_size, max_steps, ...)` pour `_states`, `_actions`, `_rewards`, `_next_states`, `_dones`. `_lengths` shape `(n_windows × snapshot_size,)`. `_window_sizes[w]` int array shape `(n_windows,)`. `_oldest_window_idx` int. `_n_captures` int.

**Memory footprint** : ~50 MB pour 10×10 (`obs_dim=200`), ~110 MB pour 15×15 (`obs_dim=450`). Négligeable.

**Filtre succès — implémentation** :

```python
def _is_successful(self, source, slot: int) -> bool:
    length = int(source._lengths[slot])
    if length == 0:
        return False
    terminated_at_end = source._dones[slot, length - 1] == 1.0
    total_reward = float(np.sum(source._rewards[slot, :length]))
    return terminated_at_end and total_reward > 0.0
```

Pour GridWorld 10×10 (`goal_reward=1.0`, `step_penalty=-0.01`, `obstacle_penalty=-1.0`) : succès → `total_reward = 1.0 − 0.01 × length > 0 ⟺ length < 100`. Échec sur obstacle → `≤ −1.0`. Filtre cleanly bimodal.

### 5.2 Helper `concat_batchseq`

Dans `mw_ia/neural/sequence_buffer.py` à côté du dataclass `BatchSeq` :

```python
def concat_batchseq(a: BatchSeq, b: BatchSeq) -> BatchSeq:
    """Concat 2 BatchSeq le long de la dimension batch (axis=1).

    Préconditions : seq_len identique, obs_dim identique.
    """
    return BatchSeq(
        states=np.concatenate([a.states, b.states], axis=1),
        actions=np.concatenate([a.actions, b.actions], axis=1),
        rewards=np.concatenate([a.rewards, b.rewards], axis=1),
        next_states=np.concatenate([a.next_states, b.next_states], axis=1),
        dones=np.concatenate([a.dones, b.dones], axis=1),
        mask=np.concatenate([a.mask, b.mask], axis=1),
    )
```

### 5.3 Agent extension — constructor

Ajouté à `RecurrentDQNAgent` (V2-Y) ET `ConvRecurrentDQNAgent` (V2-ZY) :

```python
if cfg.b1a_enabled:
    self.snapshot_store: SnapshotTrajectoryStore | None = SnapshotTrajectoryStore(
        obs_dim=obs_dim_flat,  # V2-ZY: in_channels × rows × cols ; V2-Y: obs_dim
        max_steps=cfg.max_steps_per_episode,
        n_windows=cfg.b1a_n_windows,
        snapshot_size=cfg.b1a_snapshot_size,
        seed=seed,
    )
else:
    self.snapshot_store = None
```

### 5.4 Agent extension — `on_new_best()` hook

```python
def on_new_best(self) -> int:
    """Hook appelé par le runner quand BestCheckpointTracker détecte un nouveau peak.

    Si B1a activé : capture jusqu'à snapshot_size trajectoires successful récentes
    depuis self.buffer dans self.snapshot_store (sliding window N=3).
    Si B1a désactivé : no-op.

    Returns: nombre de trajectoires effectivement capturées (0 si B1a off
    ou pas de trajectoires successful dans le buffer).
    """
    if not self.cfg.b1a_enabled:
        return 0
    return self.snapshot_store.capture_from(self.buffer)
```

### 5.5 Runner extension — hook B1a

Modification minimale dans `ConvRecurrentProceduralDQNRunner.run()` (V2-ZY) — UNE ligne après `best_tracker.update()` :

```python
# (V2-V existant, inchangé)
eval_metrics = self.evaluator.evaluate(self.agent, self.dqn_cfg.eval_target_difficulty)
improved = self.best_tracker.update(eval_metrics, self.agent, episode=ep)

# === NOUVEAU B1a hook ===
if improved:
    n_captured = self.agent.on_new_best()
    if n_captured > 0:
        self.callbacks.on_log(
            "info",
            f"B1a snapshot capture : {n_captured} trajectoires (window {self.agent.snapshot_store.n_captures})",
        )

# (V2-V callback existant, inchangé)
self.callbacks.fire_evaluation(...)
```

**V2-Y runner asymmetry** : V2-V `PeriodicEvaluator` + `BestCheckpointTracker` n'ont PAS été ajoutés à `RecurrentProceduralDQNRunner` (V2-Y) lors de la livraison V2-V. Conséquence pour B1a :
- **V2-ZY** : B1a fully usable via CLI flags et runner ConvRecurrent
- **V2-Y** : `RecurrentDQNAgent` reçoit le code B1a mais `on_new_best()` ne sera jamais appelé via le pipeline V2-Y standard. Backport V2-V hors-scope B1a.

Pour le bench cible (V2-ZY 15×15), cette asymétrie est sans impact.

### 5.6 Sample-time mix (4 combinaisons PER × B1a)

Helper privé `_sample_training_batch()` :

```python
@dataclass
class _TrainingBatch:
    batch: BatchSeq
    weights: np.ndarray | None       # PER IS weights ou None
    tree_indices: np.ndarray | None  # leaf indices pour update_priorities (PER only)


def _sample_training_batch(self) -> _TrainingBatch:
    B = self.cfg.batch_size
    L = self.cfg.sequence_length

    # B1a actif si activé ET snapshot suffisamment rempli
    snapshot_B = int(B * self.cfg.b1a_mix_ratio) if self.cfg.b1a_enabled else 0
    b1a_active = (
        self.cfg.b1a_enabled
        and snapshot_B > 0
        and len(self.snapshot_store) >= snapshot_B
    )
    main_B = B - snapshot_B if b1a_active else B

    # --- Sample main portion ---
    if self.cfg.per_enabled:
        beta = self._beta_scheduler.beta(self._episode_count)
        prio = self.buffer.sample(main_B, L, beta=beta)
        main_batch = prio.batch
        main_weights = prio.weights
        tree_indices = prio.tree_indices
    else:
        main_batch = self.buffer.sample(main_B, L)
        main_weights = None
        tree_indices = None

    # --- Sample snapshot portion (si actif) ---
    if not b1a_active:
        return _TrainingBatch(batch=main_batch, weights=main_weights, tree_indices=tree_indices)

    snapshot_batch = self.snapshot_store.sample(snapshot_B, L)
    combined_batch = concat_batchseq(main_batch, snapshot_batch)

    if self.cfg.per_enabled:
        # Snapshot portion : IS weight = 1.0 (jamais re-pondérée par PER)
        snapshot_weights = np.ones(snapshot_B, dtype=np.float32)
        combined_weights = np.concatenate([main_weights, snapshot_weights])
        return _TrainingBatch(batch=combined_batch, weights=combined_weights, tree_indices=tree_indices)
    else:
        return _TrainingBatch(batch=combined_batch, weights=None, tree_indices=None)
```

### 5.7 Boucle train dans `end_episode()`

Remplace la boucle PER actuelle (V2-B0 Task 7) par :

```python
losses: list[float] = []
for _ in range(self.cfg.train_steps_per_episode):
    tb = self._sample_training_batch()
    if tb.weights is not None:
        loss, td_errors = self.trainer.step_with_priorities(
            tb.batch, tb.weights, eta=self.cfg.per_eta,
        )
        # Update priorities UNIQUEMENT sur la portion main (snapshot store frozen)
        if tb.tree_indices is not None:
            main_B = len(tb.tree_indices)
            self.buffer.update_priorities(tb.tree_indices, td_errors[:main_B])
    else:
        loss = self.trainer.step(tb.batch)
    losses.append(loss)
```

### 5.8 Matrice de comportement 2×2

| `per_enabled` | `b1a_enabled` | Sampling | Loss | Priority updates |
|:-:|:-:|---|---|---|
| OFF | OFF | Pure main, uniform | Mean Huber sur mask | Aucune |
| ON | OFF | Pure main, PER stratified | IS-weighted Huber | Sur main batch |
| OFF | ON | 80% main uniform + 20% snapshot uniform | Mean Huber sur mask | Aucune |
| ON | ON | 80% main PER + 20% snapshot uniform | IS-weighted (PER pour main, 1.0 pour snapshot) | Sur portion main seulement (`td_errors[:main_B]`) |

---

## 6. Configuration & CLI

### 6.1 Nouveaux champs config (× 2 dataclasses)

Ajoutés à `DRQNConfig` (V2-Y) ET `ConvRecurrentDQNConfig` (V2-ZY) :

```python
# V2-B1a : Policy Snapshot Rehearsal (sliding window N captures × snapshot_size traj)
b1a_enabled: bool = False
b1a_snapshot_size: int = 50      # nombre de trajectoires capturées par best
b1a_n_windows: int = 3           # sliding window FIFO (max snapshot_size × n_windows en stock)
b1a_mix_ratio: float = 0.2       # fraction du batch venant du snapshot (0.2 = 20%)
```

**Validation `__post_init__`** :

```python
if self.b1a_snapshot_size <= 0:
    raise ValueError(f"b1a_snapshot_size doit etre > 0, recu {self.b1a_snapshot_size}")
if self.b1a_n_windows <= 0:
    raise ValueError(f"b1a_n_windows doit etre > 0, recu {self.b1a_n_windows}")
if not (0.0 < self.b1a_mix_ratio < 1.0):
    raise ValueError(
        f"b1a_mix_ratio doit etre dans ]0, 1[, recu {self.b1a_mix_ratio}"
    )
```

### 6.2 Flags CLI (× 2 scripts)

```python
parser.add_argument(
    "--b1a",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="V2-B1a : Policy Snapshot Rehearsal — capture des trajectoires successful "
         "depuis le buffer au moment du best eval (V2-V), inject 20%% dans chaque batch. "
         "Default False = pas de rehearsal (V2-U / V2-B0 baseline).",
)
parser.add_argument("--b1a-snapshot-size", type=int, default=50)
parser.add_argument("--b1a-n-windows", type=int, default=3)
parser.add_argument("--b1a-mix-ratio", type=float, default=0.2)
```

### 6.3 Recettes CLI bench 4-bras

```bash
# Bras 3 — B1a seul
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --seed {N} --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --b1a --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b1a_15x15_seed{N}.pt

# Bras 4 — B1a + PER (test interaction)
python scripts/train_cnn_lstm_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda \
    --seed {N} --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --per --b1a --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b1a_per_15x15_seed{N}.pt
```

---

## 7. Tests

### 7.1 Inventaire

| Fichier | Tests | Pytest cumul |
|---|---|---|
| `tests/training/test_snapshot_store.py` (12 tests) | snapshot_store interface + invariant | 323 → 335 |
| Extensions `tests/neural/test_prioritized_sequence_buffer.py` (4 tests) | validation 4 nouveaux champs config | 335 → 339 |
| `tests/agents/test_b1a_recurrent_agents.py` (7 parametrized × 2 agents = 14 cases) | agent integration B1a | 339 → 353 |

**Total V2-B1a : 30 nouveaux tests, pytest 323 → 353**.

### 7.2 `test_snapshot_store.py` — 12 tests

1. `test_init_validates_args` — `n_windows ≥ 1`, `snapshot_size ≥ 1`, `obs_dim ≥ 1`, `max_steps ≥ 1`
2. `test_empty_store_has_zero_length`
3. `test_capture_from_empty_buffer_returns_zero`
4. `test_capture_filters_terminated_with_positive_reward`
5. `test_capture_takes_recent_first`
6. `test_sliding_window_evicts_oldest_after_n_captures`
7. `test_n_captures_tracks_total_unbounded`
8. `test_sample_returns_batchseq_with_correct_shape`
9. `test_sample_uniform_distribution` (10000 samples, ±2% convergence)
10. `test_sample_raises_if_too_few_trajectories`
11. **`test_immutability_after_capture`** — INVARIANT central : modifier `source._states[i] = 999` post-capture → store unchanged
12. `test_reproducibility_with_seed`

### 7.3 `test_b1a_recurrent_agents.py` — 7 parametrized × 2 = 14 cases

Parametrization sur `[V2-Y RecurrentDQNAgent, V2-ZY ConvRecurrentDQNAgent]` :

1. `test_b1a_disabled_no_snapshot_store` — `snapshot_store is None` quand `b1a_enabled=False`, `on_new_best()` retourne 0
2. `test_b1a_enabled_instantiates_snapshot_store` — `SnapshotTrajectoryStore` instancié, `len == 0` init
3. `test_on_new_best_triggers_capture` — push trajectoires successful, `on_new_best()` → `len > 0`, `n_captures == 1`
4. `test_b1a_threshold_strict_pure_main_below_threshold` — snapshot avec `snapshot_B − 1` traj → batch tiré purement main
5. `test_b1a_active_mixes_batch_shape` — snapshot rempli → batch shape `(seq, B, ...)` avec `B*0.2` derniers = snapshot
6. `test_b1a_per_on_weights_concat_correct` — `weights[main_B:] == 1.0` strict, `weights[:main_B]` = PER IS weights
7. `test_b1a_per_on_priorities_updated_main_only` — mock `buffer.update_priorities`, vérifie appelé avec `td_errors[:main_B]`

### 7.4 Validation config (4 tests) — extension `test_prioritized_sequence_buffer.py`

- `test_drqn_config_b1a_snapshot_size_zero_raises`
- `test_drqn_config_b1a_n_windows_zero_raises`
- `test_drqn_config_b1a_mix_ratio_out_of_range_raises` (couvre 0.0, 1.0, 1.5, -0.1)
- `test_conv_recurrent_config_b1a_validation_parallel`

### 7.5 Backwards compat

Garantie de non-régression : `pytest -q` doit afficher **353 passed** (323 baseline V2-B0 + 30) après livraison B1a. Les 323 tests baseline V2-B0 doivent tous rester verts, grâce au `b1a_enabled=False` default.

---

## 8. CI smoke

2 nouveaux jobs dans `.github/workflows/aether_verify.yml` après les smokes V2-B0 :

```yaml
- name: Smoke test V2-B1a (snapshot rehearsal seul) on V2-ZY
  run: |
    mkdir -p checkpoints
    python scripts/train_cnn_lstm_dqn_procedural.py \
      --episodes 30 --mode obstacles --device cpu \
      --b1a --eval-every-episodes 5 \
      --best-checkpoint-path checkpoints/ci_v2b1a_best.pt
    test -f checkpoints/ci_v2b1a_best.pt

- name: Smoke test V2-B1a + PER cohabit (4e bras factoriel)
  run: |
    mkdir -p checkpoints
    python scripts/train_cnn_lstm_dqn_procedural.py \
      --episodes 30 --mode obstacles --device cpu \
      --b1a --per --polyak-tau 0.005 --eval-every-episodes 5 \
      --best-checkpoint-path checkpoints/ci_v2b1a_per_best.pt
    test -f checkpoints/ci_v2b1a_per_best.pt
```

**Note `--episodes 30`** (vs `--episodes 10` pour V2-B0) : il faut ≥ 2 evaluations (à `eval_every_episodes=5`) pour qu'un best soit captured et exercer le pipeline complet de capture. Sinon `on_new_best()` n'est jamais déclenché et la voie de code B1a critique n'est pas testée.

---

## 9. Protocole de bench scientifique

### 9.1 Design factoriel 2×2

| | **B1a OFF** | **B1a ON** |
|---|---|---|
| **PER OFF** | Bras 1 — V2-U baseline (déjà collecté) | **Bras 3** — B1a seul (NOUVEAU) |
| **PER ON** | Bras 2 — V2-B0 PER seul (déjà collecté) | **Bras 4** — interaction PER × B1a (NOUVEAU) |

**Compute** : 10 nouveaux runs (bras 3 + 4) × ~1h GPU RTX 3060 = ~10h GPU total.

### 9.2 Baselines de référence

| Métrique | V2-U baseline 15×15 (Bras 1) | V2-B0 PER 15×15 (Bras 2) |
|---|---|---|
| Mean best @ diff=0.30 | 64 % | 46 % |
| Std (n−1) | 13.4 pp | 15.17 pp |
| Min | 50 % | 30 % |
| Max | 80 % | 70 % |
| Diff_max mean | 0.36 | 0.30 |
| Late-stage collapse | 0/5 | 0/5 |

### 9.3 Critères acceptance pré-enregistrés

**Phase 1 — Bras 3 (B1a seul) vs V2-U baseline** :

| Critère | Seuil | Type |
|---|---|---|
| Mean ≥ 55 % | tolérance −9 pp pour variance | mandatory |
| Min ≥ 30 % | pas en dessous de V2-B0+PER | mandatory |
| 0/5 late-stage collapse | pattern V2-U / V2-B0 | mandatory |
| Diff_max mean ≥ 0.30 | scheduler progresse | mandatory |
| Mean > 64 % | improvement sur baseline | bonus |
| Min > 50 % | worst-case sauvé | bonus |
| Std < 10 pp | variance écrasée | bonus |

**Phase 2 — Bras 4 (B1a + PER) vs V2-B0+PER seul** :

| Critère | Seuil | Type |
|---|---|---|
| Mean > 56 % | recovery PER regression (≥ +10 pp vs Bras 2) | mandatory |
| Mean ≥ 64 % | full recovery jusqu'à baseline V2-U | bonus |
| Mean > 64 % AND > Bras 3 mean | positive interaction (PER amplifies signal après B1a stabilization) | bonus fort |

### 9.4 Analyse marginale factorielle 2×2

```text
PER main effect    = (Bras 2 + Bras 4) / 2 − (Bras 1 + Bras 3) / 2
B1a main effect    = (Bras 3 + Bras 4) / 2 − (Bras 1 + Bras 2) / 2
Interaction        = (Bras 4 + Bras 1) − (Bras 2 + Bras 3)
```

**Interaction significative** si `|Interaction| > 10 pp` :
- `Interaction > 0` : synergie (phase-dépendance confirmée)
- `Interaction < 0` : antagonisme (B1a et PER se neutralisent)

### 9.5 Matrice de décision post-bench

| Bras 3 (B1a seul) | Bras 4 (B1a+PER) | Verdict | Action tag |
|---|---|---|---|
| < 55 % (échec phase 1) | n'importe | B1a insuffisant comme rehearsal | `v0.2.0-b1a` finding négatif, brainstorm B2 |
| ≥ 55 % mais < 64 % | < 56 % | B1a stabilise mais PER reste antagoniste | Tag équilibré |
| ≥ 64 % | < 56 % | B1a améliore standalone, PER non récupéré | Tag avec "rehearsal seul suffit" |
| ≥ 55 % | ≥ 56 % (recovery partielle) | Interaction synergique partielle | Tag équilibré |
| ≥ 64 % | ≥ 64 % | **Phase-dépendance confirmée** | **Tag finding fort publishable** |
| ≥ 64 % | > 64 % strong | **Synergie positive PER × B1a** | **Tag finding très fort** |

---

## 10. Pièges & mitigations

### 10.1 Pièges d'implémentation (à attraper par TDD)

| # | Piège | Impact | Détection |
|---|---|---|---|
| 1 | Snapshot store muté post-capture (re-encode, re-rollout) | Invariant 1.2 cassé, finding invalidé | `test_immutability_after_capture` |
| 2 | `td_errors[:main_B]` oublié dans `update_priorities` (PER + B1a) | IndexError / priorities corrompues | `test_b1a_per_on_priorities_updated_main_only` (mock) |
| 3 | `concat_batchseq` axis=0 au lieu de axis=1 | Crash forward (shape brisée) | `test_b1a_active_mixes_batch_shape` |
| 4 | Filtre succès accepte `total_reward = 0` | Snapshot dilué par "ni succès ni échec" | `test_capture_filters_terminated_with_positive_reward` (test cas `reward = 0` exclu) |
| 5 | Threshold "snapshot ready" off-by-one (`>` vs `>=`) | B1a active/désactive à mauvais moment | `test_b1a_threshold_strict_pure_main_below_threshold` |
| 6 | `snapshot_B = int(B × ratio) == 0` silencieusement | B1a "activé" mais réellement inactif | check `snapshot_B > 0` dans `b1a_active` |
| 7 | IS weights snapshot portion ≠ 1.0 (PER over-corrects) | Sémantique frozen rehearsal cassée | `test_b1a_per_on_weights_concat_correct` |
| 8 | `on_new_best()` appelé après `fire_evaluation()` callback (race avec post-eval state) | Snapshot capture sur état modifié | Hook strict avant `fire_evaluation()` |
| 9 | Sliding window FIFO eviction off-by-one | Trajectoires fraîches perdues | `test_sliding_window_evicts_oldest_after_n_captures` N+1 captures |
| 10 | `agent.snapshot_store` accédé alors que `b1a_enabled=False` | AttributeError | `on_new_best()` no-op safe by design |
| 11 | `_episode_count` (PER β) et `n_captures` (B1a) confusion | β annealing désaligné | `n_captures` informational only, jamais lu par β |
| 12 | Smoke CI `--episodes 10` jamais déclenche capture | Pipeline B1a non testé en CI | `--episodes 30` minimum |

### 10.2 Pièges de bench

| # | Piège | Probabilité | Mitigation |
|---|---|---|---|
| B1 | B1a capture trop tôt (peu de successful trajectoires en buffer early training) | basse | Threshold `>= snapshot_B` strict → pure main jusqu'à snapshot rempli |
| B2 | Seul 1-2 captures sur 5000 ép (best plafonné tôt) | moyenne | C'est une donnée intéressante (early saturation), pas un bug. Documenter dans logs |
| B3 | Snapshot trajectories biais vers diff bas | basse | Filtre + buffer training spans toute la diff range. À monitorer |
| B4 | B1a recouvre mean mais variance reste élevée | moyenne | Rapporter std même si critère mean atteint, pas de cherry-pick |
| B5 | Seed 1 crash `density=0.43` (piège #10 V2-U) | moyenne | `--max-attempts-bfs 500` exposé (déjà testé V2-B0 phase 2) |

---

## 11. Definition of Done

### 11.1 Code & tests

- [ ] 30 nouveaux tests pytest verts (snapshot_store 12 + agents B1a parametrized 14 + config validation 4)
- [ ] `pytest -q` total : **353 passed** (323 baseline V2-B0 + 30)
- [ ] `bash aether/verify_all.sh` → 8 OK (inchangé)
- [ ] V2-U baseline reproductible strict sans `--b1a` (smoke 5 ép sans flag)
- [ ] V2-B0 PER baseline reproductible strict sans `--b1a` (smoke 5 ép `--per`)
- [ ] V2-Y et V2-ZY baselines préservées (existing tests verts)
- [ ] CI : 2 nouveaux smoke jobs passent sur CPU
- [ ] `git log` : commits TDD bite-sized par phase

### 11.2 Bench scientifique

- [ ] Bras 3 — V2-B1a seul 15×15 n=5 collecté
- [ ] Bras 4 — V2-B1a + PER 15×15 n=5 collecté
- [ ] Tableau résultats par seed + agrégats dans CLAUDE.md
- [ ] Analyse marginale 2×2 factorielle (PER effect, B1a effect, interaction)
- [ ] Verdict selon matrice de décision (section 9.5), documenté honnêtement
- [ ] Pas de goalpost moving — critères Phase 1 et Phase 2 pré-enregistrés respectés

### 11.3 Doc

- [ ] Spec dans `docs/superpowers/specs/2026-05-29-mw-ia-policy-snapshot-rehearsal-design.md` + commit
- [ ] Plan TDD via writing-plans dans `docs/superpowers/plans/`
- [ ] Section V2-B1a dans CLAUDE.md (phases + composants + décisions + pièges + recettes + bench)
- [ ] README mis à jour avec recette B1a
- [ ] Cartographie bottlenecks RL mise à jour dans `memory/projet_mw_ia_phase_dependence_finding.md` (row B1a)
- [ ] Architectural invariant 1.2 formalisé spec + docstring + tests

### 11.4 Tag

- [ ] `git tag v0.2.0-b1a` après DoD ci-dessus

---

## 12. Décisions reportées post-bench

| Outcome | Décision suivante |
|---|---|
| Phase 1 ❌ (B1a alone fails) | Tag négatif. Brainstorm B2 (episodic memory) — retrieval par contexte peut-être nécessaire, pas juste mix uniforme |
| Phase 1 ✅, Phase 2 ❌ (B1a aide, PER reste antagoniste) | Tag équilibré. B2 prioritaire |
| Phase 1 ✅, Phase 2 ✅ recovery partielle | Tag "interaction synergique partielle". B2 reste pertinent mais B1a déjà victoire stratégique |
| Phase 1 ✅, Phase 2 ✅ recovery complète OU positive interaction | **Tag finding fort publishable — validation expérimentale phase-dépendance.** B2 devient "amplification" plutôt que "fix" |

**Bras 3 alone est le test critique** : détermine si "rehearsal frozen" est viable indépendamment du sampling strategy. Sans victoire bras 3, le finding V2-V *"le meilleur agent existait déjà avant ep 3500"* ne se traduit pas en capacité opérationnelle.

---

## 13. Référence littéraire

| Concept | Source |
|---|---|
| Experience Replay original | Lin 1992, "Self-improving reactive agents based on reinforcement learning" |
| Rehearsal vs distillation | Hinton et al. 2015, "Distilling the Knowledge in a Neural Network" |
| Catastrophic forgetting RL | Kirkpatrick et al. 2017, "Overcoming catastrophic forgetting in neural networks" (EWC) |
| Episodic memory RL | Pritzel et al. 2017, "Neural Episodic Control" |
| Best policy snapshots | Espeholt et al. 2018, "IMPALA" (V-trace + multi-actor) |

B1a est plus simple que tous ces papers — c'est un MVP de rehearsal frozen, conçu pour tester proprement l'hypothèse forgetting-preservation. Si validé, ouvre la voie vers B1b (distillation) ou B2 (retrieval) qui se rapprochent de l'état de l'art.
