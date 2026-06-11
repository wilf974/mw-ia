# Spec V2-ZY — CNN + LSTM + Double DQN combiné

> **Sous-projet** : V2-ZY (MW_IA — Reinforcement Learning éducatif)
> **Date** : 2026-05-23
> **Statut** : Spec validée — implémentation à dérouler via `superpowers:writing-plans` puis `superpowers:subagent-driven-development`
> **Tag livraison cible** : `v0.2.0-zy`
> **Sous-projets prérequis livrés** : V1, V2-A, V2-X, V2-Y, V2-Z, V2-W, V2-V

---

## 1. Vue d'ensemble & hypothèse

### Motivation empirique consolidée (post-V2-V validation, 2026-05-23)

L'évaluation rigoureuse V2-V (greedy strict, 10 seeds eval held-out, diff fixe) a établi le nouveau benchmark de référence :

| Variante | Mean best @ diff=0.30 (eval rigoureux) | Best ≥ 70 % strict |
|---|---|---|
| V2-W CNN + Double DQN (n=5) | 58 % | 2/5 |
| V2-W @ diff=0.20 (n=5) | 74 % | 4/5 |
| **V2-ZY cible @ diff=0.30** | **≥ 70 %** | **≥ 4/5** |

**Phrase clé V2-V** :
> La capacité réelle se situe autour de **diff=0.20 robuste**, avec un **plafond actuel vers diff=0.25-0.30**. Tous les benchmarks futurs DOIVENT être rapportés en eval V2-V rigoureux.

### Hypothèse V2-ZY (combo des 3 leviers livrés)

> Conv2D (V2-Z) gère la perception spatiale, LSTM (V2-Y) ajoute la mémoire temporelle (utile pour se souvenir des dead-ends explorés intra-épisode), Double DQN (V2-W) stabilise l'estimation Q. Les trois sont théoriquement orthogonaux. Si l'hypothèse tient, V2-ZY franchit la cible V2-V @ diff=0.20 (74 % moyen, 4/5 ≥ 70 %) transposée à **diff=0.30** : best ≥ 70 % sur 4/5 seeds en eval rigoureux.

### Question scientifique

> Est-ce que la mémoire temporelle (V2-Y LSTM) ajoutée à la perception spatiale (V2-Z) et à la stabilité Q (V2-W) permet de pousser le plafond de diff=0.25-0.30 vers diff=0.40+ en eval rigoureux ?

### Critère succès chiffré

- **4/5 seeds** avec `best_checkpoint_winrate @ diff=0.30 ≥ 70 %`
- Mesuré en eval V2-V greedy strict (10 seeds eval 10000-10009, `eval_target_difficulty=0.30`)
- Benchmark complémentaire @ diff=0.40 si critère principal atteint (caractérisation du nouveau plafond)

### Pattern d'intégration

Nouveau réseau combiné `ConvRecurrentQNetwork`, réutilise le buffer V2-Y `SequenceReplayBuffer` et le trainer V2-Y `RecurrentDQNTrainer` (étendu avec flag `double_dqn` compatible V2-W pattern). Nouveau runner parallèle qui intègre V2-V eval+best-checkpoint dès l'origine. Tag cible : `v0.2.0-zy` (posé même si critère non atteint — livraison code ≠ validation scientifique).

---

## 2. Architecture & composants

### Réseau combiné `ConvRecurrentQNetwork`

Architecture : `Conv → Flatten → LSTM → FC` (Hausknecht-style DRQN sur observations spatiales).

```
input              (B, seq_len, 3, R, C)        ← obs 3D par step
                   reshape → (B*seq_len, 3, R, C)
Conv1 (3→32, k=3, pad=1)  → (B*seq_len, 32, R, C) + ReLU
Conv2 (32→64, k=3, pad=1) → (B*seq_len, 64, R, C) + ReLU
Flatten            → (B*seq_len, 64*R*C)
                   reshape → (B, seq_len, 64*R*C)   ← retour à 3D pour LSTM

# Default pour 10×10 : 64*10*10 = 6400 features par step

LSTM(input=6400, hidden=128, batch_first=True)  → (B, seq_len, 128)
FC(128 → n_actions)                              → (B, seq_len, 4)
```

- Conv block réutilise les hyperparams V2-Z (`conv_channels=(32, 64), kernel_size=3, padding=1`)
- LSTM hyperparams V2-Y (`lstm_hidden=128`)
- Forward 1-step (eval/act runtime) accepte aussi `(1, 1, 3, R, C)` avec gestion hidden state externe (compat V2-Y agent pattern)
- **Params estimés** : Conv ≈ 19 K + LSTM 6400→128 ≈ 3.3 M + FC ≈ 500 ≈ **3.3 M total** (vs V2-Z CNN seul 1.66 M, V2-Y LSTM seul ~1 M)

### Nouveaux fichiers code

| Fichier | Rôle |
|---|---|
| `mw_ia/neural/conv_recurrent.py` | `ConvRecurrentQNetwork` : Conv block + LSTM + FC. Forward 1-step et batch sequence. Hidden state externe (compat V2-Y agent pattern). |
| `mw_ia/agents/conv_recurrent_dqn.py` | `ConvRecurrentDQNAgent` : équivalent V2-Y `RecurrentDQNAgent` mais sur ConvRecurrentQNetwork. Hidden state runtime maintenu, reset par épisode, train via SequenceReplayBuffer. |
| `scripts/train_cnn_lstm_dqn_procedural.py` | CLI parallèle aux scripts V2-Y/V2-Z. Flags combinés V2-Z (conv) + V2-Y (lstm, seq) + V2-W (double_dqn) + V2-V (eval, best-checkpoint). |
| `tests/neural/test_conv_recurrent.py` | 5 tests réseau |
| `tests/agents/test_conv_recurrent_dqn.py` | 7 tests agent |
| `tests/training/test_conv_recurrent_procedural_runner.py` | 2 tests runner intégration |
| `tests/test_conv_recurrent_dqn_config.py` | 5 tests config |

### Extensions de fichiers existants

| Fichier | Extension |
|---|---|
| `mw_ia/config.py` | + `ConvRecurrentDQNConfig` frozen dataclass : champs combinés V2-Z (conv) + V2-Y (lstm, seq, replay_capacity en trajectoires) + V2-W (`double_dqn=True` par défaut V2-ZY) + V2-V (eval_enabled, eval_seeds, eval_target_difficulty, best_checkpoint_path). |
| `mw_ia/neural/recurrent_trainer.py` | + paramètre `double_dqn: bool = False` au `__init__` (default `False` préserve V2-Y baseline) + branche conditionnelle dans `step()` (pattern V2-W). |
| `mw_ia/training/runner.py` | + `ConvRecurrentProceduralDQNRunner` (parallèle à `ConvProceduralDQNRunner` et `RecurrentProceduralDQNRunner`). Intègre `PeriodicEvaluator` + `BestCheckpointTracker` (pattern V2-V). |
| `mw_ia/training/evaluator.py` | + dans `PeriodicEvaluator.evaluate()` : appel `agent.begin_episode()` au début de chaque rollout SI la méthode existe (via `getattr`). Duck-typing : no-op pour ConvDQNAgent V2-Z/W, reset hidden pour ConvRecurrentDQNAgent V2-ZY. |
| `tests/neural/test_recurrent_trainer.py` (existant V2-Y) | + 1 test pour la branche Double DQN dans le trainer |
| `tests/training/test_evaluator.py` (existant V2-V) | + 1 test pour `begin_episode` duck-typing |

### Décisions d'API

**`ConvRecurrentQNetwork.forward(x, hidden=None) -> tuple[Tensor, tuple]`** :
- `x` shape `(B, seq_len, in_channels, rows, cols)` (training BPTT) OU `(1, 1, in_channels, rows, cols)` (runtime act)
- `hidden` : tuple `(h, c)` LSTM ou `None` (auto-init zéros)
- Returns: `(q_values, new_hidden)` avec `q_values` shape `(B, seq_len, n_actions)`

**`ConvRecurrentDQNAgent.act(state, *, greedy=False) -> int`** :
- `state` shape `(in_channels, rows, cols)` (single obs, pas séquence)
- Maintient `self._hidden_state` runtime (LSTM continuity intra-épisode)
- Forward LSTM MAINTENU même en exploration eps-greedy (pattern V2-Y)
- Bypass eps-greedy + `torch.no_grad()` si `greedy=True` (compat V2-V eval)

**`ConvRecurrentDQNAgent.begin_episode() -> None`** :
- Reset `self._hidden_state` à `None` (LSTM redémarre fresh)
- Démarre une nouvelle trajectoire dans `SequenceReplayBuffer`
- Appelé par `PeriodicEvaluator` au début de chaque rollout eval

**`RecurrentDQNTrainer.__init__(..., double_dqn: bool = False)`** (extension V2-Y) :
- Nouveau kwarg `double_dqn` stocké comme attribut
- Default `False` préserve V2-Y baseline (35 tests V2-Y existants restent verts)

**`RecurrentDQNTrainer.step(batch) -> float`** (extension branche V2-W) :
- Si `self.double_dqn=True` :
  - `next_actions = online(next_states).argmax(dim=-1)` (sélection : online)
  - `q_next = target(next_states).gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)` (eval : target)
- Sinon : `q_next = target(next_states).max(dim=-1).values` (V2-Y baseline DQN classique)
- Loss Huber masquée + AMP + grad clip identiques V2-Y

**`PeriodicEvaluator.evaluate(agent, difficulty)`** (extension V2-V duck-typing) :
- Au début de chaque rollout (avant `eval_env.reset(seed=seed)`) :
  - `begin = getattr(agent, "begin_episode", None)`
  - `if begin is not None: begin()`
- No-op pour ConvDQNAgent V2-Z/W (pas de méthode), reset hidden pour ConvRecurrentDQNAgent V2-ZY

### Pattern d'isolation

- V2-Y `RecurrentDQNAgent`, `RecurrentProceduralDQNRunner` restent intacts (zéro modif)
- V2-Z `ConvDQNAgent`, `ConvProceduralDQNRunner` restent intacts (zéro modif)
- V2-W (flag `double_dqn` dans `ConvDQNConfig`) reste intact
- V2-V (`PeriodicEvaluator`, `BestCheckpointTracker`) reçoit 1 extension backwards-compat (`begin_episode` duck-typing)
- V2-Y trainer reçoit 1 nouveau kwarg backwards-compat (default `False`)

---

## 3. Data flow détaillé

### Initialisation `ConvRecurrentProceduralDQNRunner.__init__`

```python
super().__init__(train_cfg, callbacks)
self.scheduler = AdaptiveDifficultyScheduler(sched_cfg)
self.bucket_tracker = DifficultyBucketTracker(train_cfg)
self.agent = ConvRecurrentDQNAgent(
    in_channels=3, rows=proc_cfg.max_rows, cols=proc_cfg.max_cols,
    n_actions=4, cfg=dqn_cfg, device=device, seed=seed,
)

if dqn_cfg.eval_enabled:
    eval_gen = type(env.generator).__new__(type(env.generator))
    eval_gen.__dict__.update(env.generator.__dict__)
    eval_env = ProceduralGridWorld(cfg=proc_cfg, generator=eval_gen)
    self.evaluator = PeriodicEvaluator(
        eval_env=eval_env, eval_seeds=dqn_cfg.eval_seeds,
        max_steps=dqn_cfg.eval_max_steps,
        observation_encoder=encode_procedural_observation_2d,
        proc_cfg=proc_cfg,
    )
    self.best_tracker = BestCheckpointTracker(path=dqn_cfg.best_checkpoint_path)
else:
    self.evaluator = None
    self.best_tracker = None
```

### Boucle d'épisode dans `ConvRecurrentProceduralDQNRunner.run()`

```python
for ep in range(self.dqn_cfg.episodes):
    self.env.set_difficulty(self.scheduler.current)
    state, info = self.env.reset(seed=ep)
    maze = info["maze"]
    goal = self.env.inner.cfg.goal

    # V2-Y pattern : reset hidden + begin_episode trajectoire
    self.agent.reset_hidden()
    self.agent.begin_episode()

    ep_reward, ep_len = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated) and ep_len < self.dqn_cfg.max_steps_per_episode:
        obs = encode_procedural_observation_2d(
            state=state, grid=maze, goal=goal,
            max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
        )
        a = self.agent.act(obs)
        s2, r, terminated, truncated, _ = self.env.step(a)
        next_obs = encode_procedural_observation_2d(
            state=s2, grid=maze, goal=goal,
            max_rows=self.proc_cfg.max_rows, max_cols=self.proc_cfg.max_cols,
        )
        self.agent.observe(obs, a, r, next_obs, terminated or truncated)
        state, ep_reward, ep_len = s2, ep_reward + r, ep_len + 1

    # Train à la fin d'épisode (pattern V2-Y DRQN)
    m = self.agent.end_episode()
    if "loss" in m:
        self.metrics.record_loss(m["loss"])
        self.callbacks.fire_loss(self.agent.global_step, m["loss"])

    self.metrics.record_episode(ep_reward, ep_len, success=terminated)
    self.bucket_tracker.record_episode(...)
    self.callbacks.fire_episode(...)

    if (ep + 1) % self.sched_cfg.update_interval == 0:
        new_diff = self.scheduler.update(winrate=self.metrics.winrate())
        self.callbacks.fire_difficulty_updated(...)

    # V2-V eval périodique
    if self.evaluator is not None and (ep + 1) % self.dqn_cfg.eval_every_episodes == 0:
        eval_metrics = self.evaluator.evaluate(self.agent, self.dqn_cfg.eval_target_difficulty)
        improved = self.best_tracker.update(eval_metrics, self.agent, episode=ep)
        self.callbacks.fire_evaluation(...)
        self.callbacks.fire_log("info", f"eval ep {ep} : winrate={eval_metrics['winrate']:.2%} ...")
```

### Trace `ConvRecurrentDQNAgent.act(state)` (runtime, 1-step forward)

```python
def act(self, state: np.ndarray, *, greedy: bool = False) -> int:
    assert state.shape == (self.in_channels, self.rows, self.cols)
    if (not greedy) and self._rng.random() < self.epsilon:
        # IMPORTANT : forward LSTM MAINTENU même en exploration eps-greedy
        # (cohérent V2-Y pattern — la mémoire doit suivre la trajectoire)
        action = int(self._rng.integers(0, self.n_actions))
        with torch.no_grad():
            x = torch.from_numpy(state).float().to(self.device)
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, R, C)
            _, self._hidden_state = self.online(x, self._hidden_state)
        return action
    # Greedy : forward + argmax
    with torch.no_grad():
        x = torch.from_numpy(state).float().to(self.device)
        x = x.unsqueeze(0).unsqueeze(0)
        q, self._hidden_state = self.online(x, self._hidden_state)
        return int(q.argmax(dim=-1).item())
```

### Trace `PeriodicEvaluator.evaluate(agent, difficulty)` (avec begin_episode duck-typing)

```python
def evaluate(self, agent: _ActableAgent, difficulty: float) -> dict[str, float | int]:
    self.eval_env.set_difficulty(difficulty)
    n_success = 0
    total_reward = 0.0
    total_length = 0
    for seed in self.eval_seeds:
        # V2-V extension : duck-typing begin_episode
        begin = getattr(agent, "begin_episode", None)
        if begin is not None:
            begin()  # reset hidden state pour LSTM agents

        state, info = self.eval_env.reset(seed=seed)
        maze = info["maze"]
        goal = self.eval_env.inner.cfg.goal
        ep_reward = 0.0
        ep_len = 0
        terminated = truncated = False
        while not (terminated or truncated) and ep_len < self.max_steps:
            obs = self.observation_encoder(
                state=state, grid=maze, goal=goal,
                max_rows=self.proc_cfg.max_rows,
                max_cols=self.proc_cfg.max_cols,
            )
            action = agent.act(obs, greedy=True)
            state, reward, terminated, truncated, _ = self.eval_env.step(action)
            ep_reward += reward
            ep_len += 1
        if terminated:
            n_success += 1
        total_reward += ep_reward
        total_length += ep_len

    n = len(self.eval_seeds)
    return {
        "winrate": n_success / n,
        "mean_reward": total_reward / n,
        "mean_length": total_length / n,
        "n_episodes": n,
        "difficulty": float(difficulty),
    }
```

### Trace `RecurrentDQNTrainer.step(batch)` (BPTT recurrent + Double DQN optionnel)

```python
def step(self, batch: BatchSeq) -> float:
    # batch contient trajectoires sample (B, seq_len, ...) + mask padding
    states = batch.states         # (B, seq_len, obs_dim_flat) → reshape (B, seq_len, 3, R, C) si V2-ZY
    actions = batch.actions       # (B, seq_len)
    rewards = batch.rewards       # (B, seq_len)
    next_states = batch.next_states
    dones = batch.dones           # (B, seq_len)
    mask = batch.mask             # (B, seq_len) — 1=vrai step, 0=padding

    with torch.amp.autocast(...):
        q_online, _ = self.online(states, None)  # (B, seq_len, n_actions)
        q_pred = q_online.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            if self.double_dqn:  # V2-W branche appliquée au BPTT recurrent
                q_online_next, _ = self.online(next_states, None)
                next_actions = q_online_next.argmax(dim=-1)  # (B, seq_len)
                q_target_next, _ = self.target(next_states, None)
                q_next = q_target_next.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            else:  # V2-Y baseline
                q_target_next, _ = self.target(next_states, None)
                q_next = q_target_next.max(dim=-1).values

            target_q = rewards + self.gamma * q_next * (1.0 - dones)

        loss_per = self.loss_fn(q_pred, target_q)  # reduction='none'
        loss = (loss_per * mask).sum() / mask.sum().clamp(min=1.0)

    # AMP scaler + grad clip + optimizer step (identique V2-Y)
    self.optimizer.zero_grad(set_to_none=True)
    if self.use_amp:
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self._scaler.step(self.optimizer)
        self._scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
    return float(loss.detach().item())
```

**Note Double DQN avec hidden state** : le forward `self.online(next_states, None)` reset l'hidden state au début de chaque séquence d'entraînement (DRQN simple V2-Y, pas burn-in R2D2). Le `None` initialise zéros.

### Coût mémoire

- `ConvRecurrentQNetwork` ≈ 3.3 M params × 2 (online + target) + AMP optimizer state ≈ 80 MB GPU
- `SequenceReplayBuffer` V2-Y : capacity=5000 traj × max_steps=200 × obs_dim=300 × 4 bytes × 2 (state + next) ≈ **2.4 GB** sur RTX 3060
- Si OOM observé : descendre `replay_capacity=2000` (encore 1000+ trajectoires = ample pour stats)

### Overhead training

- Conv block ~0.3 ms / step CPU side, négligeable GPU
- LSTM unchanged from V2-Y
- BPTT 32 steps coût identique V2-Y
- Eval V2-V : ~10 % overhead (idem V2-V sur ConvProceduralDQNRunner)
- **Total estimé** : ~10-15 % plus lent que V2-W standalone, ~25 % plus lent que V2-Y standalone

---

## 4. Testing + DoD

### Phases TDD prévisionnelles

| Phase | Composant | Fichier de test | Tests | Cumul |
|---|---|---|---|---|
| 1 | Setup scaffold (`conv_recurrent.py`, `conv_recurrent_dqn.py`, fichiers tests vides) | — | 0 | 231 (baseline) |
| 2 | `ConvRecurrentQNetwork` (forward 1-step, batch sequence, hidden state, gradient flow, params count) | `tests/neural/test_conv_recurrent.py` | 5 | 236 |
| 3 | `RecurrentDQNTrainer` extension `double_dqn` flag | `tests/neural/test_recurrent_trainer.py` (existant V2-Y) | +1 | 237 |
| 4 | `ConvRecurrentDQNConfig` (champs combinés + validation + Aether compat) | `tests/test_conv_recurrent_dqn_config.py` | 5 | 242 |
| 5 | `ConvRecurrentDQNAgent` (init, reset_hidden, begin_episode, act hidden maintenu eps-random, act greedy déterministe, end_episode train trigger, Aether smoke) | `tests/agents/test_conv_recurrent_dqn.py` | 7 | 249 |
| 6 | `ConvRecurrentProceduralDQNRunner` (smoke E2E 200 ép avec V2-V eval activé, best-checkpoint sauvegardé) | `tests/training/test_conv_recurrent_procedural_runner.py` | 2 | 251 |
| 7 | V2-V extension `PeriodicEvaluator.evaluate()` appelle `agent.begin_episode()` si existe | `tests/training/test_evaluator.py` (existant V2-V) | +1 | 252 |
| 8 | CLI `train_cnn_lstm_dqn_procedural.py` + CI smoke | (CI workflow) | 0 | 252 |
| 9 | README + CLAUDE.md V2-ZY + smoke E2E GPU + tag `v0.2.0-zy` | — | 0 | 252 |

**Total ajouté : 21 tests** (231 → 252).

### Détail des tests critiques

**`test_conv_recurrent.py`** (5 tests) :

1. `test_forward_single_step_with_hidden` — input `(1, 1, 3, 10, 10)` + `hidden=None` → q `(1, 1, 4)` + new hidden tuple `(h, c)`
2. `test_forward_batch_sequence` — input `(8, 32, 3, 10, 10)` → q `(8, 32, 4)`
3. `test_hidden_state_propagation` — 2 forwards consécutifs avec passage du hidden, outputs diffèrent du forward `hidden=None`
4. `test_gradient_flow` — `loss.backward()` produit des grads non-nuls sur Conv1, Conv2, LSTM, FC
5. `test_params_count` — total params ≈ 3.3 M (tolerance ±5 %)

**`test_recurrent_trainer.py`** (+1 test V2-Y existant) :

6. `test_double_dqn_branch_differs_from_standard` — pattern V2-W appliqué au trainer recurrent. Avec online ≠ target, q_next Double DQN ≠ q_next DQN classique sur batch sequence

**`test_conv_recurrent_dqn_config.py`** (5 tests) :

1. `test_defaults` — `conv_channels=(32,64)`, `lstm_hidden=128`, `double_dqn=True` (V2-ZY = combo), `eval_enabled=True`, `eval_target_difficulty=0.30`
2. `test_validation_conv_channels_positive`
3. `test_validation_lstm_hidden_positive`
4. `test_validation_eval_target_difficulty_bounds` (∈ [0, 1])
5. `test_aether_compat` — `verify_formal(VariantSpec.from(ConvRecurrentDQNConfig())).passed`

**`test_conv_recurrent_dqn.py`** (7 tests, pattern V2-Y `test_recurrent_dqn.py`) :

1. `test_init` — online + target nets, sequence buffer empty, global_step=0, hidden None
2. `test_reset_hidden` — `_hidden_state` redevient None
3. `test_begin_episode_resets_hidden_and_starts_trajectory` — reset + trajectory démarrée dans sequence buffer
4. `test_act_maintains_hidden_state_even_in_exploration` — eps=1.0 → action random MAIS `_hidden_state` updated par forward
5. `test_act_greedy_deterministic` — eps=0 + même hidden → mêmes actions sur même obs
6. `test_end_episode_trains_when_buffer_full` — `max(min_episodes_to_learn, batch_size)` trigger (fix V2-Y appliqué dès le début)
7. `test_aether_smoke` — `verify_formal(VariantSpec.from(cfg)).passed`

**`test_conv_recurrent_procedural_runner.py`** (2 tests, pattern V2-V) :

1. `test_runner_full_episode_with_eval_and_best_checkpoint` — smoke 200 ép `eval_every_episodes=50`, `eval_target_difficulty=0.30`, vérifier (a) eval ≥ 3 fois, (b) `best_path.exists()`, (c) `best_tracker.best_winrate >= 0.0`
2. `test_runner_eval_disabled_no_evaluator` — `eval_enabled=False` → `evaluator is None` et `best_tracker is None`, run() sans crash

**`test_evaluator.py`** (+1 test V2-V) :

9. `test_evaluator_calls_begin_episode_if_exists` — duck-typing : agent avec `begin_episode` attribute appelé une fois par seed eval ; agent sans cette méthode = no-op (pas d'AttributeError)

### Definition of Done

**DoD bloquante (livraison code)** :

1. ✅ `mw_ia/neural/conv_recurrent.py` : `ConvRecurrentQNetwork` complète
2. ✅ `mw_ia/neural/recurrent_trainer.py` : kwarg `double_dqn` + branche conditionnelle dans `step()`
3. ✅ `mw_ia/agents/conv_recurrent_dqn.py` : `ConvRecurrentDQNAgent` complète (act, observe, end_episode, save/load, reset_hidden, begin_episode)
4. ✅ `mw_ia/config.py` : `ConvRecurrentDQNConfig` complète (combo V2-Z + V2-Y + V2-W + V2-V)
5. ✅ `mw_ia/training/runner.py` : `ConvRecurrentProceduralDQNRunner` intégré avec V2-V
6. ✅ `mw_ia/training/evaluator.py` : extension `evaluate()` appelle `agent.begin_episode()` duck-typing
7. ✅ `scripts/train_cnn_lstm_dqn_procedural.py` : CLI complète (flags V2-Z + V2-Y + V2-W + V2-V combinés)
8. ✅ `pytest -q` → **252 passed** (231 baseline + 21 V2-ZY)
9. ✅ `bash aether/verify_all.sh` → 8 OK inchangé
10. ✅ Smoke E2E GPU 500 ép avec `--eval --best-checkpoint-path` : best-checkpoint .pt créé, eval logs présents, pas de crash NaN
11. ✅ Section V2-ZY dans README.md (parallèle V2-V)
12. ✅ Section V2-ZY dans CLAUDE.md (phases, composants, décisions, pièges)
13. ✅ Tag `v0.2.0-zy` posé
14. ✅ Tags antérieurs (v0.1.0, v0.2.0-a/x/y/z/w/v) intacts
15. ✅ V2-Y baseline (35 tests existants) reste vert avec `double_dqn=False` default (zero régression)

**DoD non-bloquante (validation scientifique)** :

16. ⏳ Benchmark same-seed n=5 V2-ZY ep=5000 GPU avec `--eval-target-difficulty 0.30 --best-checkpoint-path checkpoints/v2zy_best_seed{N}.pt`
17. ⏳ Section "V2-ZY — validation empirique" dans CLAUDE.md : tableau comparatif vs V2-W (mêmes seeds, seule différence = ajout LSTM + flag double_dqn dans config)
18. ⏳ **Critère succès** : 4/5 seeds avec best ≥ 70 % @ diff=0.30
19. ⏳ Si critère atteint → benchmark bonus @ diff=0.40 pour caractériser nouveau plafond
20. ⏳ Si critère non atteint → documenter honnêtement, candidats restants : Polyak soft target, mazes larges, refonte LSTM avec burn-in R2D2

### Pièges anticipés

| # | Piège | Mitigation |
|---|---|---|
| 1 | **`SequenceReplayBuffer` stocke obs en flat (300 dim)** | `ConvRecurrentDQNAgent.observe()` flatten 3D→1D avant push. Trainer reshape `(B, seq, 3, R, C)` avant forward. Identique pattern V2-Z buffer V1. |
| 2 | **LSTM forward avec `hidden=None` au début de chaque séquence training** | DRQN simple (V2-Y) — hidden zero-init pour chaque trajectoire sample. Pas de burn-in R2D2 (hors-scope V2-ZY MVP). Documenter dans docstring. |
| 3 | **V2-Y trainer modification touche V2-Y livré** | `double_dqn: bool = False` par défaut — préserve V2-Y baseline. Test régression : les 35 tests V2-Y existants doivent rester verts. |
| 4 | **V2-V `evaluate()` ne reset pas hidden state entre seeds eval** | Extension V2-V : appel `agent.begin_episode()` au début de chaque rollout via `getattr(agent, 'begin_episode', None)`. Compatible avec ConvDQNAgent V2-Z (no-op) ET ConvRecurrentDQNAgent V2-ZY (reset). |
| 5 | **Forward conv block sur grand batch×seq (B=128 × seq=32 = 4096 frames)** | Reshape `(B*seq, 3, R, C)` pour conv, puis `(B, seq, features)` pour LSTM. ~50 MB VRAM par forward, ample marge RTX 3060. |
| 6 | **Eval avec hidden state runtime contamine entre rollouts seeds** | `PeriodicEvaluator` doit reset hidden au début de CHAQUE seed eval (10 rollouts indépendants). Via `agent.begin_episode()` extension (Piège 4). |
| 7 | **Replay buffer 2.4 GB peut OOM si autres processus** | Configurable via `replay_capacity` CLI flag. Default 5000 trajectoires (V2-Y) reste sûr. Si OOM, descendre à 2000. |
| 8 | **Hidden state save/load** : `agent.save(path)` actuel V2-Y sauve online+target+global_step+cfg. Pas le hidden state. | Acceptable : best-checkpoint est sauvegardé en début d'eval (juste après `begin_episode`), donc avec un hidden zero-init au load. Best-model performance dépend uniquement des poids réseau. |
| 9 | **Hidden state forward en eps-random** | Volontaire (pattern V2-Y) : la mémoire LSTM doit suivre la trajectoire complète indépendamment des choix d'action. Documenter dans docstring `act()`. |

---

## 5. Annexe — récap des décisions clés

| Décision | Choix | Raison |
|---|---|---|
| Architecture réseau | `Conv → Flatten → LSTM → FC` (Hausknecht-style) | Standard pattern DRQN+CNN, compatible BPTT avec SequenceReplayBuffer V2-Y |
| Réseau dim params | ~3.3 M (Conv 19 K + LSTM 6400→128 ≈ 3.3 M + FC 0.5 K) | Conv block V2-Z + LSTM hidden V2-Y inchangé |
| Buffer | `SequenceReplayBuffer` V2-Y réutilisé (obs flatten 1D → reshape `(B, seq, 3, R, C)` au train) | Zéro duplication, pattern V2-Z buffer V1 |
| Trainer | `RecurrentDQNTrainer` V2-Y étendu avec flag `double_dqn` (default `False`) | Backwards-compat V2-Y, branche conditionnelle V2-W pattern |
| Default `double_dqn` dans `ConvRecurrentDQNConfig` | `True` (V2-W activé par défaut V2-ZY) | V2-ZY = combo des 3 leviers |
| Agent | Nouveau `ConvRecurrentDQNAgent` (parallèle `RecurrentDQNAgent` V2-Y) | Hidden state runtime maintenu, pattern V2-Y |
| Runner | Nouveau `ConvRecurrentProceduralDQNRunner` (parallèle ConvProceduralDQNRunner V2-Z/W) | Intègre V2-V eval+best-checkpoint dès l'origine |
| V2-V extension | `PeriodicEvaluator.evaluate()` appelle `agent.begin_episode()` si méthode existe | Duck-typing, no-op pour V2-Z, reset hidden pour V2-ZY |
| Critère succès | 4/5 seeds best ≥ 70 % @ diff=0.30 en eval rigoureux | Transposition V2-V @ diff=0.20 (74 % moyen, 4/5 ≥ 70 %) à diff=0.30 |
| Tag | `v0.2.0-zy` posé même si critère non atteint | Livraison code ≠ validation scientifique (pattern V2-V) |

### Story scientifique cible (post-V2-ZY)

Si critère atteint (4/5 ≥ 70 % @ diff=0.30) :

| Levier(s) | Mean best @ diff=0.30 | Best ≥ 70 % strict |
|---|---|---|
| V2-W CNN + Double DQN (n=5) | 58 % | 2/5 |
| **V2-ZY CNN + LSTM + Double DQN (n=5, cible)** | **≥ 70 %** | **≥ 4/5** |
| Gain attribuable au LSTM (V2-Y) | **+12 pp** | **+2 seeds** |

→ Story : "**Perception spatiale + stabilité Q + mémoire temporelle = combo nécessaire et suffisant pour bucket 1 robuste**". 3 leviers orthogonaux validés indépendamment et en combinaison.

### Sous-projets V3+ déblocables après V2-ZY

- **V2-ZY benchmark @ diff=0.40** : caractériser le nouveau plafond (peut-être bucket 2 0.4-0.6 partiellement rempli ?)
- **Soft target Polyak τ=0.005** : appliqué à V2-ZY pour éliminer le résidu de collapse tardif
- **R2D2 burn-in** : remplacer DRQN simple par burn-in pour stabiliser LSTM training
- **Mazes larges (max_size=15/20)** : tester translation equivariance CNN sur grilles plus grandes
- **Sous-projet B (mémoire persistante cross-session)** : maintenant viable, infra V2-V protège du collapse, V2-ZY donne un agent stable

### Pourquoi le sous-projet est moyen-gros mais bien borné

L'infrastructure existe déjà :
- Conv block (V2-Z), LSTM (V2-Y), Double DQN logic (V2-W), buffer trajectoires (V2-Y), trainer BPTT (V2-Y), eval+best-checkpoint (V2-V)

V2-ZY est essentiellement **un nouveau réseau combinant deux extracteurs existants + 1 flag déjà éprouvé** + plomberie runner/CLI. Plus gros que V2-W (~50 LOC) ou V2-V (~150-200 LOC), mais beaucoup plus petit qu'une refonte. ROI scientifique : très élevé si le combo théorique fonctionne empiriquement.

---

**Fin de la spec V2-ZY.**
