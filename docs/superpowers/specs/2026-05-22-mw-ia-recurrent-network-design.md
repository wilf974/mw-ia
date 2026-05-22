# MW_IA — Recurrent DQN (LSTM) — Design Spec

> **Sous-projet V2-Y** (roadmap #1, successeur logique de V2-X). À implémenter via le pattern Subagent-Driven Development utilisé pour V1, V2-A et V2-X.

---

## 1. Objectif

Franchir le plafond architectural du DQN feedforward identifié en V2-X (winrate plafonne à ~80 % à difficulté ~0.10) en introduisant une **mémoire neuronale temporelle (LSTM)** dans le QNetwork.

L'agent récurrent peut désormais conditionner ses décisions sur l'historique de trajectoire dans le maze courant — c'est-à-dire se souvenir des dead-ends récents, des cellules déjà visitées, et de l'orientation générale vers le goal.

### Critère de succès vérifiable

- Env de test : V2-X procedural obstacles avec recette gagnante V2-X (`min_density=0.0`, `max_density=0.50`, `--hidden 256 256`, `--epsilon-decay-steps 200000`).
- 5000 épisodes sur RTX 3060.
- **L'agent atteint un winrate ≥ 70 % au bucket 1 du tracker (difficulté 0.20-0.40).**
- Baseline V2-X (DQN feedforward) : plafonne au bucket 0 (0.0-0.20), bucket 1 vide.
- Critère équivalent : le scheduler doit franchir et maintenir une difficulté ≥ 0.20 de manière stable.

### Critères de succès secondaires

- **170+ tests pytest verts** à la livraison (148 V1+V2-A+V2-X + ~22 nouveaux V2-Y).
- V1 fix-map et V2-X procedural continuent de fonctionner sans modification (rétro-compat stricte).
- Tous les invariants Aether V2-A (I1-I8) continuent de tenir.

### Hors-scope explicites

- **GRU, Transformer, autres archis récurrentes** → roadmap V3+
- **Burn-in style R2D2** (DRQN simple suffit pour MW_IA) → optimisation future
- **Stored hidden state en buffer** (memory-heavy) → optimisation future
- **Double DQN, Dueling DQN, Prioritized Experience Replay** → sous-projets séparés
- **Intégration GUI** spécifique au mode récurrent (la GUI V2-X continue de marcher avec V1 et V2-X)
- **Hyperparameter tuning automatique** (manuel pour l'instant)
- **Modification de V1, V2-A, V2-X** : rétro-compat stricte sur tous les modules existants

### Contraintes non-négociables

1. Tous les invariants Aether V2-A continuent de tenir. Aucune modification de `mw_ia/guardrails/` ni `aether/invariants/`.
2. V1 et V2-X restent intacts. Nouveau code en parallèle.
3. Pattern Subagent-Driven Development utilisé pour V1, V2-A, V2-X.

---

## 2. Architecture

V1 et V2-X restent intacts. V2-Y est une **3ème ligne** d'agent/runner parallèle, comme V2-X était la 2ème.

```
mw_ia/
├── config.py                          # [MODIFIÉ] + DRQNConfig (frozen dataclass)
├── neural/
│   ├── network.py                     # V1 inchangé
│   ├── recurrent.py                   # [NOUVEAU] RecurrentQNetwork (LSTM)
│   ├── replay_buffer.py               # V1 inchangé
│   ├── sequence_buffer.py             # [NOUVEAU] SequenceReplayBuffer
│   ├── trainer.py                     # V1 inchangé
│   └── recurrent_trainer.py           # [NOUVEAU] RecurrentDQNTrainer
├── agents/
│   ├── dqn.py                         # V1 inchangé
│   └── recurrent_dqn.py               # [NOUVEAU] RecurrentDQNAgent
├── training/
│   └── runner.py                      # [MODIFIÉ] + RecurrentProceduralDQNRunner
└── scripts/
    └── train_drqn_procedural.py       # [NOUVEAU] CLI DRQN
```

### 2.1 Nouveaux composants

#### `RecurrentQNetwork`

Fichier : `mw_ia/neural/recurrent.py`

Structure :
```
input → Linear(input_dim → fc_hidden) → ReLU → LSTM(fc_hidden, lstm_hidden) → Linear(lstm_hidden → n_actions)
```

```python
class RecurrentQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_actions: int = 4,
        fc_hidden: int = 256,
        lstm_hidden: int = 128,
    ) -> None:
        ...
        self.fc_in = nn.Linear(input_dim, fc_hidden)
        self.lstm = nn.LSTM(fc_hidden, lstm_hidden, batch_first=False)
        self.fc_out = nn.Linear(lstm_hidden, n_actions)

    def forward(
        self,
        obs_seq: torch.Tensor,                              # (seq, batch, input_dim)
        hidden: tuple[torch.Tensor, torch.Tensor] | None,   # ((1, batch, lstm_hidden), idem) | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Returns: (q_values (seq, batch, n_actions), new_hidden tuple)
```

- Hidden = `None` → auto-init zéros (pattern PyTorch LSTM).
- Accepte aussi bien un seul timestep (`seq=1, batch=1`, utilisé par `act()`) qu'une séquence complète (`seq=32, batch=128`, utilisé par le trainer).
- `batch_first=False` (convention RL canonique : seq en premier).

#### `SequenceReplayBuffer`

Fichier : `mw_ia/neural/sequence_buffer.py`

Stocke des **trajectoires complètes par épisode** dans un buffer circulaire.

```python
@dataclass
class BatchSeq:
    states: np.ndarray       # (seq, batch, obs_dim) float32
    actions: np.ndarray      # (seq, batch) int64
    rewards: np.ndarray      # (seq, batch) float32
    next_states: np.ndarray  # (seq, batch, obs_dim) float32
    dones: np.ndarray        # (seq, batch) float32
    mask: np.ndarray         # (seq, batch) float32 — 1 pour vrais steps, 0 pour padding


class SequenceReplayBuffer:
    """Buffer circulaire de trajectoires complètes.

    Capacity = nombre de trajectoires (PAS transitions, contrairement à V1 ReplayBuffer).
    Chaque trajectoire = jusqu'à max_steps transitions de l'épisode.

    Sample : tire B trajectoires aléatoires, pour chacune tire une fenêtre aléatoire de
    seq_len steps consécutifs, padding zéros si trajectoire plus courte.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        max_steps: int = 200,
        *,
        seed: int = 0,
    ) -> None:
        ...

    def push_trajectory(self, trajectory: list[tuple]) -> None:
        """trajectory = liste de (state, action, reward, next_state, done).

        Raises ValueError si len(trajectory) ∉ [1, max_steps].
        """

    def sample(self, batch_size: int, seq_len: int) -> BatchSeq:
        """Tire batch_size trajectoires avec remise, fenêtre aléatoire seq_len par trajectoire."""
```

- `replay_capacity` default 5000 trajectoires (ordre de grandeur ~1.6 GB sur RTX 3060 12 GB).
- À ne pas confondre avec `replay_capacity` V1 (qui compte des transitions).

#### `RecurrentDQNTrainer`

Fichier : `mw_ia/neural/recurrent_trainer.py`

Identique au `DQNTrainer` V1 mais opère sur séquences :

```python
class RecurrentDQNTrainer:
    def __init__(
        self,
        online: RecurrentQNetwork,
        target: RecurrentQNetwork,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        device: str = "cuda",
        use_amp: bool = True,
    ) -> None:
        ...

    def step(self, batch: BatchSeq) -> float:
        # Forward online : q_pred (seq, batch, n_actions) avec hidden=None (zero init)
        # Forward target : q_next (seq, batch, n_actions) sur next_states, hidden=None
        # target_q = rewards + gamma * q_next.max(dim=-1) * (1 - dones)
        # huber_loss = smooth_l1(q_pred[gather(actions)], target_q) * mask
        # loss = huber_loss.sum() / mask.sum()  (moyenne sur vrais steps uniquement)
        # backward + clip_grad_norm 10.0 + Adam + AMP
```

- **Hidden state zero-init au début de chaque séquence d'entraînement** (DRQN simple, pas burn-in).
- **Huber loss masquée** : ignore les steps paddés (mask=0).
- AMP, grad clip, Adam : pattern V1.
- `sync_target()` identique au V1.

#### `RecurrentDQNAgent`

Fichier : `mw_ia/agents/recurrent_dqn.py`

Interface compatible avec `RecurrentProceduralDQNRunner`.

```python
class RecurrentDQNAgent(Agent):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        cfg: DRQNConfig,
        *,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        ...
        self.online = RecurrentQNetwork(obs_dim, n_actions, cfg.fc_hidden, cfg.lstm_hidden).to(device)
        self.target = RecurrentQNetwork(obs_dim, n_actions, cfg.fc_hidden, cfg.lstm_hidden).to(device)
        self.trainer = RecurrentDQNTrainer(self.online, self.target, ...)
        self.buffer = SequenceReplayBuffer(cfg.replay_capacity, obs_dim, cfg.max_steps_per_episode, seed=seed)
        self._hidden_state: tuple[torch.Tensor, torch.Tensor] | None = None
        self._episode_trajectory: list[tuple] = []

    def reset_hidden(self) -> None:
        """Appelé par le runner au début de chaque épisode."""
        self._hidden_state = None

    def begin_episode(self) -> None:
        """Vide la trajectoire de l'épisode courant."""
        self._episode_trajectory = []

    def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
        # 1. ε-greedy random ?
        # 2. Forward 1 timestep, met à jour self._hidden_state
        # 3. Retourne argmax

    def observe(self, obs, action, reward, next_obs, done) -> dict[str, float]:
        self._episode_trajectory.append((obs, action, reward, next_obs, done))
        self.global_step += 1
        # PAS de train step ici. Tout se passe à end_episode().
        return {"epsilon": self.epsilon}

    def end_episode(self) -> dict[str, float]:
        """Appelé par le runner après terminated/truncated.

        Push la trajectoire dans le buffer, déclenche train steps si seuil atteint.
        """
        self.buffer.push_trajectory(self._episode_trajectory)
        metrics: dict[str, float] = {"epsilon": self.epsilon}
        if len(self.buffer) >= self.cfg.min_episodes_to_learn:
            losses: list[float] = []
            for _ in range(self.cfg.train_steps_per_episode):
                batch = self.buffer.sample(self.cfg.batch_size, self.cfg.sequence_length)
                losses.append(self.trainer.step(batch))
            if losses:
                self.last_loss = sum(losses) / len(losses)
                metrics["loss"] = self.last_loss
        if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
            self.trainer.sync_target()
            self.target_syncs += 1
        return metrics
```

- **Hidden state runtime maintenu entre `act()` consécutifs** dans un épisode.
- **Train step à la fin de l'épisode** (pas par step comme V1).
- Pattern save/load identique à V1.

#### `RecurrentProceduralDQNRunner`

Extension dans `mw_ia/training/runner.py` (ajout en bas, V2-X `ProceduralDQNRunner` inchangé).

Quasi-identique à `ProceduralDQNRunner` mais :

- Utilise `RecurrentDQNAgent` au lieu de `DQNAgent`.
- Appelle `agent.reset_hidden()` et `agent.begin_episode()` après `env.reset()`.
- Appelle `agent.end_episode()` après la boucle step (au lieu de `observe()` à chaque step).
- Métriques `loss`, `epsilon` traitées identiquement.
- Tout le reste (scheduler, bucket tracker, callbacks GUI) inchangé.

```python
class RecurrentProceduralDQNRunner(_BaseRunner):
    def __init__(
        self,
        *,
        env: ProceduralGridWorld,
        proc_cfg: ProceduralEnvConfig,
        drqn_cfg: DRQNConfig,
        sched_cfg: SchedulerConfig,
        train_cfg: TrainingConfig,
        callbacks: RunnerCallbacks,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        super().__init__(train_cfg, callbacks)
        ...
        obs_dim = 2 * proc_cfg.max_rows * proc_cfg.max_cols
        self.agent = RecurrentDQNAgent(obs_dim, n_actions=4, cfg=drqn_cfg, device=device, seed=seed)

    def run(self) -> None:
        for ep in range(self.drqn_cfg.episodes):
            ...
            self.env.set_difficulty(self.scheduler.current)
            state, info = self.env.reset(seed=ep)
            maze = info["maze"]
            difficulty = info["difficulty"]
            self.callbacks.fire_maze_changed(maze=maze, episode_id=ep, difficulty=difficulty)

            self.agent.reset_hidden()
            self.agent.begin_episode()

            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated) and ep_len < self.drqn_cfg.max_steps_per_episode:
                ...
                obs = encode_procedural_observation(state, maze, ...)
                a = self.agent.act(obs)
                s2, r, terminated, truncated, _ = self.env.step(a)
                next_obs = encode_procedural_observation(s2, maze, ...)
                self.agent.observe(obs, a, r, next_obs, terminated or truncated)
                state = s2
                ep_reward += r
                ep_len += 1

            m = self.agent.end_episode()
            if "loss" in m:
                self.metrics.record_loss(m["loss"])
                self.callbacks.fire_loss(self.agent.global_step, m["loss"])
            self.callbacks.fire_epsilon(self.agent.global_step, m["epsilon"])

            self.metrics.record_episode(ep_reward, ep_len, success=terminated)
            self.bucket_tracker.record_episode(success=terminated, reward=ep_reward, length=ep_len, difficulty=difficulty)
            ...
```

### 2.2 Composants modifiés

#### `DRQNConfig`

Nouveau frozen dataclass dans `mw_ia/config.py`, structurellement similaire à `DQNConfig` mais avec champs LSTM-specific :

```python
@dataclass(frozen=True)
class DRQNConfig:
    """Deep Recurrent Q-Network (LSTM). Successeur V2-Y de DQNConfig."""

    # Réseau
    fc_hidden: int = 256                # couche FC avant LSTM
    lstm_hidden: int = 128              # taille du hidden state LSTM

    # Sequence
    sequence_length: int = 32

    # Replay
    replay_capacity: int = 5000         # NOMBRE DE TRAJECTOIRES (pas transitions !)
    min_episodes_to_learn: int = 100
    train_steps_per_episode: int = 4

    # Optimisation (identique V1)
    batch_size: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000  # default V2-X gagnant
    target_sync_steps: int = 1_000
    use_amp: bool = True

    # Training
    episodes: int = 5_000
    max_steps_per_episode: int = 200

    def __post_init__(self) -> None:
        if not (1 <= self.sequence_length <= self.max_steps_per_episode):
            raise ValueError(...)
        if self.replay_capacity <= 0:
            raise ValueError(...)
        if not (0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0):
            raise ValueError(...)
        # ... etc.
```

### 2.3 Composants explicitement non modifiés

- `mw_ia/agents/dqn.py` (V1 `DQNAgent`)
- `mw_ia/neural/network.py` (V1 `QNetwork`)
- `mw_ia/neural/replay_buffer.py` (V1 `ReplayBuffer`)
- `mw_ia/neural/trainer.py` (V1 `DQNTrainer`)
- `mw_ia/training/runner.py` classes `DQNRunner`, `TabularRunner`, `ProceduralDQNRunner` (extensions seulement, pas modifications)
- `mw_ia/guardrails/` (V2-A intact)
- `aether/` (V2-A intact)
- `mw_ia/gui/` (V2-X intact)
- Aucun test V1/V2-A/V2-X modifié

---

## 3. Data flow

Flow d'un épisode dans le mode RecurrentDQN procedural :

```
[RecurrentProceduralDQNRunner.run_episode(episode_id)]
   │
   ├─► env.set_difficulty(scheduler.current)
   ├─► state, info = env.reset(seed=episode_id)
   │       └─► info = {"maze": grid, "difficulty": diff, "episode_id": ep}
   │
   ├─► agent.reset_hidden()       # (h, c) ← None (zero-init lazy au prochain forward)
   ├─► agent.begin_episode()      # trajectoire courante vidée
   │
   ├─► boucle step :
   │      ├─► obs = encode_procedural_observation(state, maze, max_rows, max_cols)
   │      │       # dim 200 sur 10×10
   │      │
   │      ├─► action = agent.act(obs)
   │      │      ├─► x = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0)   # (seq=1, batch=1, 200)
   │      │      ├─► (q, new_hidden) = self.online(x, self._hidden_state)
   │      │      ├─► self._hidden_state = new_hidden    # maintenu entre steps
   │      │      └─► return ε-greedy(q.argmax())
   │      │
   │      ├─► next_state, r, terminated, truncated, _ = env.step(action)
   │      ├─► next_obs = encode_procedural_observation(next_state, maze, ...)
   │      │
   │      ├─► agent.observe(obs, action, r, next_obs, done)
   │      │      └─► self._episode_trajectory.append((obs, action, r, next_obs, done))
   │      │
   │      └─► state = next_state ; ep_reward += r ; ep_len += 1
   │
   ├─► metrics = agent.end_episode()
   │      ├─► self.buffer.push_trajectory(self._episode_trajectory)
   │      ├─► if len(buffer) >= min_episodes_to_learn :
   │      │      └─► boucle train_steps_per_episode (4 par défaut) :
   │      │             ├─► batch = self.buffer.sample(batch_size=128, seq_len=32)
   │      │             │       # B trajectoires aléatoires, fenêtres aléatoires 32 steps,
   │      │             │       # padding zéros + mask
   │      │             └─► loss = self.trainer.step(batch)
   │      │                    # forward online sur seq complète, hidden=None
   │      │                    # forward target sur next_states, hidden=None
   │      │                    # huber masquée (ignore padding)
   │      │                    # BPTT 32 steps + Adam + AMP + grad clip 10
   │      └─► if global_step // target_sync_steps > syncs : sync_target()
   │
   ├─► bucket_tracker.record_episode(...)
   ├─► metrics_global.record_episode(...)
   ├─► si (ep+1) % update_interval == 0 : scheduler.update(winrate=...)
   │
   └─► callbacks GUI : on_maze_changed, on_episode_done, on_difficulty_updated
```

### Points clés

- **Hidden state runtime vs train state** : pendant un épisode, l'agent maintient `self._hidden_state` (utile pour `act()`). Pendant le training, le forward part toujours d'un hidden state zéro-initialisé au début de la séquence de 32 steps (bias DRQN classique accepté).

- **Train step à la fin d'épisode** : DRQN nécessite trajectoires complètes ; entraîner sur transitions individuelles ne marche pas avec LSTM. Conséquence : `train_every` du V1 ignoré ; remplacé par `train_steps_per_episode` (4 train batches par épisode).

- **Mask pour épisodes courts** : agent atteint goal en 18 steps → trajectoire stockée a longueur 18. Échantillonnage retourne `obs[0:18] + 14 zéros padding` avec `mask = [1]*18 + [0]*14`. La huber loss multiplie par mask avant moyennage.

- **Échantillonnage** : tire `batch_size=128` trajectoires distinctes (avec remise), puis pour chacune tire un offset aléatoire `start ∈ [0, max(0, len(traj) - seq_len)]`. Pas de chevauchement entre trajectoires (chaque batch element a son hidden init à zéro).

- **Memory budget** : `replay_capacity=5000` trajectoires × 200 steps max × (obs_dim=200 × 4 bytes × 2 + overhead) ≈ 1.6 GB sur RTX 3060 12 GB. Acceptable.

---

## 4. Error handling

| Cas | Source | Stratégie MVP | Visibilité |
|---|---|---|---|
| Hidden state non reset entre épisodes | Bug runner | Test unitaire : 2 épisodes consécutifs avec mêmes obs → même action si reset_hidden() OK | Test |
| Trajectoire poussée avec mauvaise longueur | Bug agent | `push_trajectory` valide `1 ≤ len ≤ max_steps`, `raise ValueError` | Crash explicite |
| Sample avant min_episodes_to_learn | Logique runner | `sample()` lève `ValueError("buffer trop petit")`, runner garde la garde `if len(buffer) >= min` | Pattern V1 |
| Séquence demandée > max_steps | Mauvaise config | `__post_init__` valide `1 ≤ sequence_length ≤ max_steps_per_episode` | Validation construction |
| Hidden state mauvaise shape | Bug LSTM | `forward` accepte `None` ou tuple `(h, c)` avec h.shape == c.shape == `(1, batch, lstm_hidden)`. Assert explicite | AssertionError |
| Padding mask incorrect | Bug buffer | Test unitaire : 18 steps → mask `[1]*18 + [0]*14`, loss masquée doit ignorer | Test |
| NaN/Inf dans loss BPTT | Gradient explosion | `clip_grad_norm_(max_norm=10.0)` + AMP. Si NaN persistent, baisser seq_len de 32 à 16 (manuel) | Crash si non clamp |
| Replay buffer overflow | Trop d'épisodes | Buffer circulaire : écrase à `idx`, `idx = (idx+1) % capacity` (pattern V1) | Silent, capacity respectée |
| OOM VRAM | Batch×seq×obs grosse | `__post_init__` valide bornes raisonnables. Doc : baisser batch_size ou replay_capacity | Doc only |
| BPTT trop lent | seq_len grand | Pas une erreur ; tradeoff documenté (~2-3x V1 par train step) | Observable via log |
| Agent V1 utilise par mégarde RecurrentDQNAgent | Mauvais wiring | `RecurrentProceduralDQNRunner` seul l'instancie. V1/V2-X inchangés. | Pattern projet |
| Tests Aether V2-A cassent | Modif invariants par effet de bord | Aucune modif `mw_ia/guardrails/` ni `aether/invariants/` en V2-Y. Test sync continue de garantir. | Test V2-A intact |

### Décisions de design issues du tableau

1. **Crashes explicites > silent fallbacks** (pattern projet V1/V2-A/V2-X).
2. **Hidden state init à zéro = bias accepté** (DRQN simple, pas burn-in en MVP).
3. **Train step à la fin d'épisode** (granularité différente de V1). Compteur `global_step` continue d'avancer step-par-step pour epsilon decay.
4. **`replay_capacity` = nombre de TRAJECTOIRES** (pas transitions, contrairement à V1). Documenté dans la docstring.
5. **Pas de récupération sur NaN loss** : utilisateur stoppe et ajuste hyperparam. Grad clipping V1 suffit dans 99 % des cas.

---

## 5. Configuration

### `DRQNConfig` (nouveau dataclass frozen)

Voir §2.2. Defaults :

- `fc_hidden=256`, `lstm_hidden=128`
- `sequence_length=32`
- `replay_capacity=5000` (trajectoires)
- `min_episodes_to_learn=100`
- `train_steps_per_episode=4`
- `batch_size=128`, `lr=1e-3`, `gamma=0.99`
- `epsilon_start=1.0`, `epsilon_end=0.05`, `epsilon_decay_steps=200_000` (V2-X gagnant)
- `target_sync_steps=1_000`
- `use_amp=True`
- `episodes=5_000`, `max_steps_per_episode=200`

---

## 6. Testing strategy

### 6.1 Tests unitaires (~22 tests)

**`tests/neural/test_recurrent.py`** — ~7 tests
- Instanciation, forward 1 step shape correct, forward seq `(32, 4, 200)` → Q `(32, 4, 4)` + hidden
- `None` hidden → auto-init
- Hidden passé entre 2 forward change le résultat (sanity LSTM)
- 2 forward mêmes obs + mêmes hidden → mêmes Q (déterminisme)
- Backward pass propage gradient sur tous params
- Mauvaise shape hidden → AssertionError

**`tests/neural/test_sequence_buffer.py`** — ~6 tests
- `push_trajectory` valide longueur [1, max_steps]
- `push_trajectory` len > max → ValueError
- `sample` avant min épisodes → ValueError
- Shapes `(seq_len, batch, obs_dim)` correctes + mask `(seq_len, batch)`
- Padding correct : 18 steps → 18 mask=1 + 14 mask=0
- Capacity circulaire

**`tests/neural/test_recurrent_trainer.py`** — ~3 tests
- 1 train step batch faux ne crash pas
- Mask masque effectivement la huber loss
- `sync_target()` copie les poids

**`tests/agents/test_recurrent_dqn.py`** — ~6 tests
- `act()` retourne action ∈ [0, n_actions)
- `act()` mêmes obs 2x avec hidden reset → même action (greedy=True)
- `reset_hidden()` zero-init effectivement
- `begin_episode()` vide trajectoire
- `end_episode()` push trajectoire + déclenche train step si seuil
- save/load round-trip

### 6.2 Tests intégration (~2 tests, lents)

**`tests/training/test_recurrent_procedural_runner.py`**
- 30 épisodes sur env V2-X obstacles CPU, recette gagnante : pas d'erreur, mazes différents, loss > 0
- Smoke pytest total : **≥ 170 passed** (148 + 22)

### 6.3 Test E2E smoke (1, CI)

**`scripts/train_drqn_procedural.py`**
- `python scripts/train_drqn_procedural.py --episodes 20 --device cpu`
- Vérifie : exit 0, output "winrate", "difficulty", "Per-bucket"
- Step CI ajouté dans `.github/workflows/aether_verify.yml`

### 6.4 Validation finale (manuelle)

`python scripts/train_drqn_procedural.py --episodes 5000 --mode obstacles --device cuda --hidden 256 256 --epsilon-decay-steps 200000` (ou flags simplifiés si `DRQNConfig` defaults sont déjà optimaux).

**Critère** : bucket 1 (0.2-0.4) winrate ≥ 70 % en fin d'entraînement. Comparaison directe avec baseline V2-X DQN feedforward 5000 ép qui plafonne au bucket 0.

### 6.5 Tests V2-A (intacts)

- `pytest tests/guardrails/` continue de passer (43 tests).
- `bash aether/verify_all.sh` continue d'afficher 8 OK.

### 6.6 Hors-scope testing

- Pas de benchmark performance automatisé.
- Pas de test GUI (V2-X garantit rétro-compat).
- Pas de test OOM VRAM.
- Pas de test NaN/Inf (grad clipping suffit dans 99 %).

---

## 7. Stratégie d'implémentation

Pattern **Subagent-Driven Development**, comme V1, V2-A et V2-X. Phases prévues (à détailler dans le plan suivant) :

1. **Phase 1 — Setup** : scaffold `mw_ia/neural/recurrent.py`, `sequence_buffer.py`, `recurrent_trainer.py`, dossier `tests/neural/`.
2. **Phase 2 — `RecurrentQNetwork`** (Linear+LSTM+Linear avec hidden tuple).
3. **Phase 3 — `SequenceReplayBuffer`** (push_trajectory + sample seq_len avec mask + padding).
4. **Phase 4 — `RecurrentDQNTrainer`** (BPTT + huber masquée).
5. **Phase 5 — `DRQNConfig`** (frozen dataclass + `__post_init__`).
6. **Phase 6 — `RecurrentDQNAgent`** (orchestration : reset_hidden / begin/end_episode / save/load).
7. **Phase 7 — `RecurrentProceduralDQNRunner`** (extension de runner.py).
8. **Phase 8 — `scripts/train_drqn_procedural.py`** + intégration CI.
9. **Phase 9 — README V2-Y + DoD smoke test + tag `v0.2.0-y`** (ou nom à décider).

Chaque phase suit le triple gate : implementer subagent → spec compliance review → code quality review.

---

## 8. Références

- État projet : `CLAUDE.md` (V1 `v0.1.0`, V2-A `v0.2.0-a`, V2-X `v0.2.0-x`, 148 tests verts + recette V2-X opérationnelle).
- Recette V2-X gagnante : `--hidden 256 256 --epsilon-decay-steps 200000` (cf. CLAUDE.md §"V2-X — recette opérationnelle").
- DRQN : Hausknecht & Stone (2015), "Deep Recurrent Q-Learning for Partially Observable MDPs".
- R2D2 (référence pour burn-in, hors-scope V2-Y MVP) : Kapturowski et al. (2019), "Recurrent Experience Replay in Distributed Reinforcement Learning".
- Spec V2-X (motivation du plafond identifié) : `docs/superpowers/specs/2026-05-22-mw-ia-procedural-env-design.md`.
