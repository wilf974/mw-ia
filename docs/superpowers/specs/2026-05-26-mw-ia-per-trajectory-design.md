# V2-B0 — Trajectory-level Prioritized Experience Replay (PER)

**Date** : 2026-05-26
**Statut** : Spec — design validé en brainstorming, en attente review user finale avant writing-plans
**Position programme V2** : Sous-projet B, phase 0 (contrôle scientifique avant B1/B2)
**Dépendances livrées** : V2-Z (tag `v0.2.0-z`), V2-Y (`v0.2.0-y`), V2-W (`v0.2.0-w`), V2-V (`v0.2.0-v`), V2-ZY (`v0.2.0-zy`), V2-U (`v0.2.0-u`)
**Cible empirique** : régime stable V2-ZY+Polyak (mean 92 % @ diff=0.30 sur 10×10, mean 64 % sur 15×15, std 13 pp, 0/5 collapse)

---

## 1. Contexte & motivation

### 1.1 État du programme V2

La branche RL "capacité × stabilité × scaling" a été fermée proprement (cf. CLAUDE.md sections V2-Z à V2-U) avec la cascade additive :

1. **V2-Z** : représentation spatiale CNN débloque la généralisation curriculum
2. **V2-W** : Double DQN double le mean (variance restait haute)
3. **V2-V** : eval rigoureux + best-checkpoint = mesure honnête
4. **V2-ZY** : combo CNN + LSTM + Double DQN = potentiel maximal mais instable (std 39.6 pp)
5. **V2-U** : Polyak soft target stabilise V2-ZY → mean 92 %, std 13 pp, 0/5 collapse

Le finding consolidé V2-U n'est pas "Polyak résout tout" mais "**LSTM × Polyak = double lissage temporel cohérent**" : LSTM construit une représentation lente, Polyak stabilise la target conformément à cette lenteur. Sans LSTM, Polyak n'apporte rien (V2-W+Polyak +2 pp marginal).

### 1.2 Le signal qui motive B

Le finding pratique le plus important de V2-W n=5 :

> Le meilleur agent V2-W existait avant ep 3500. L'entraînement après ep 3500 le détruit sur certains seeds.

V2-V (best-checkpoint tracking) a capté ce pic, mais le pattern indique un **bottleneck structurel** : le système découvre parfois une politique utile, ne sait pas la préserver, et le replay uniforme la dilue. C'est exactement le régime où la mémoire devient le bottleneck.

### 1.3 Décomposition B0/B1/B2 (validée brainstorming 2026-05-26)

| Sous-projet | Hypothèse testée | Mécanisme | LOC estimées |
|---|---|---|---|
| **B0** (ce design) | Le replay uniforme est-il un bottleneck ? | Prioritized Experience Replay trajectoire-level | ~800 |
| B1 (futur) | Le comportement découvert tôt peut-il être préservé/réinjecté ? | Policy snapshot rehearsal au pic eval V2-V | TBD |
| B2 (futur) | Le rappel conditionnel par contexte apporte-t-il plus que PER + rehearsal ? | Episodic memory store + retrieval policy | TBD |

B0 est le contrôle scientifique : si PER améliore V2-ZY+Polyak, on confirme que le sampling mémoire est causalement impliqué. Si PER n'aide pas, B1/B2 devront viser autre chose que la priorité TD-error simple.

---

## 2. Hypothèse & question scientifique

**Question primaire** : Le replay uniforme sur trajectoires dans `SequenceReplayBuffer` est-il un bottleneck du régime stable V2-ZY+Polyak ?

**Hypothèse H0 (nulle)** : Le sampling uniforme ne pénalise pas l'apprentissage. PER trajectory-level avec hyperparams canoniques (Schaul 2015 α=0.6, β annealé 0.4→1.0, R2D2 2019 η=0.9) n'améliore pas significativement mean/min/convergence/diff_max sur 15×15 n=5.

**Hypothèse H1** : Le sampling uniforme dilue les trajectoires informatives (succès near-frontier, TD-error élevé). PER reproduit ces trajectoires plus souvent, accélère la convergence ou améliore le plafond.

**Régime cible** : V2-ZY+Polyak (CNN + LSTM + Double DQN + Polyak τ=0.005). C'est le régime stable validé empiriquement V2-U n=5. Toute modification doit préserver ce régime ou l'améliorer.

**Critère scientifique de qualité** : protocole same-seed n=5 (seeds 0-4), eval rigoureux V2-V (best @ diff=0.30 fixe greedy sur 10 seeds held-out), pré-enregistrement des critères d'acceptance (cf. section 9).

---

## 3. Périmètre

### 3.1 In-scope

- Implémentation `PrioritizedSequenceReplayBuffer` (trajectory-level, sum tree, IS correction)
- Extension `RecurrentDQNTrainer` avec méthode `step_with_priorities` (loss IS-weighted, R2D2 aggregation des TD-errors)
- Extension `RecurrentDQNAgent` (V2-Y) et `ConvRecurrentDQNAgent` (V2-ZY) avec branche conditionnelle PER
- Helper `BetaScheduler` (annealing linéaire β_start → β_end)
- 6 nouveaux champs config (`per_enabled`, `per_alpha`, `per_beta_start`, `per_beta_end`, `per_eta`, `per_epsilon`) sur `DRQNConfig` ET `ConvRecurrentDQNConfig`
- 6 flags CLI sur `train_drqn_procedural.py` ET `train_cnn_lstm_dqn_procedural.py`
- 2 smoke CI (PER seul + PER + Polyak cohabit)
- Bench n=5 same-seed V2-ZY+Polyak+PER vs baseline V2-U sur 10×10 ET 15×15
- Flag CLI `--max-attempts-bfs` (pre-mitigation piège #6, cf. section 10)
- Documentation : README + CLAUDE.md + tag `v0.2.0-b0`

### 3.2 Out-of-scope (renvoyé à plus tard)

- **PER sur `ReplayBuffer` flat V1** (utilisé par V2-W et baseline DQN) : pas dans le régime cible
- **Step-within-trajectory PER** (R2D2 strict) : variante explorée en B0.1 si B0 null + V2-W PER positif
- **Bucket-aware / frontier-aware priorities** : appartient à B2 (conflate non-uniforme et frontier-aware sinon)
- **Policy snapshot rehearsal** : appartient à B1
- **Episodic memory store + retrieval** : appartient à B2
- **GUI bouton "procedural CNN+LSTM+PER"** : extensible post-livraison si pertinent
- **Grid search hyperparams** : defaults littéraires figés. Grid search uniquement si bench null sur 15×15

---

## 4. Architecture

### 4.1 Vue d'ensemble

```
                  ┌──────────────────────────────────────────┐
                  │  ConvRecurrentDQNAgent  (V2-ZY+Polyak)   │
                  │  ┌─────────────────────────────────────┐ │
                  │  │ if cfg.per_enabled:                 │ │
                  │  │   PrioritizedSequenceReplayBuffer ──┼─┼──→ SumTree
                  │  │   BetaScheduler                     │ │      (capacity, alpha, eps)
                  │  │ else:                               │ │
                  │  │   SequenceReplayBuffer (V2-Y/ZY)   ←┼─┼──── baseline strict
                  │  └─────────────────────────────────────┘ │
                  │  end_episode():                          │
                  │    push_trajectory(priority=max_prio)    │
                  │    if PER:                               │
                  │      beta = scheduler.beta(ep_count)     │
                  │      prio_batch = buffer.sample(B,L,beta)│
                  │      loss, td = trainer.step_with_prio…  │──→ R2D2 aggregation
                  │      buffer.update_priorities(idx, td)   │
                  └──────────────────────────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │  RecurrentDQNTrainer (V2-Y/W/U/ZY)       │
                  │   _step_impl(batch, weights, eta):       │
                  │     forward + target (Double DQN si on)  │
                  │     loss = (huber × mask × IS_w).sum()   │
                  │            / mask.sum()                  │
                  │     backward + clip + opt step + Polyak  │
                  │     return (loss, eta×max + (1-eta)×mean)│
                  └──────────────────────────────────────────┘
```

### 4.2 Inventaire fichiers

| Type | Fichier | LOC nettes estimées |
|---|---|---|
| Nouveau | `mw_ia/neural/sum_tree.py` | ~80 |
| Nouveau | `mw_ia/neural/prioritized_sequence_buffer.py` (incl. `BetaScheduler`) | ~150 |
| Modifié | `mw_ia/neural/recurrent_trainer.py` | +60 (refactor `_step_impl`) |
| Modifié | `mw_ia/agents/recurrent_dqn.py` | +30 |
| Modifié | `mw_ia/agents/conv_recurrent_dqn.py` | +30 |
| Modifié | `mw_ia/config.py` | +12 (6 champs × 2 dataclasses) |
| Modifié | `scripts/train_drqn_procedural.py` | +20 |
| Modifié | `scripts/train_cnn_lstm_dqn_procedural.py` | +20 |
| Modifié | `.github/workflows/aether_verify.yml` | +16 (2 smoke jobs) |
| Nouveau | `tests/neural/test_sum_tree.py` | ~80 |
| Nouveau | `tests/neural/test_prioritized_sequence_buffer.py` | ~120 |
| Nouveau | `tests/neural/test_beta_scheduler.py` | ~30 |
| Nouveau | `tests/neural/test_per_trainer.py` | ~80 |
| Nouveau | `tests/agents/test_per_recurrent_agents.py` | ~80 |

**Total** : ~420 LOC code (modifs + additions) + ~390 LOC tests (5 fichiers) = ~810 LOC, plus ~48 nouveaux tests pytest (incluant parametrization V2-Y / V2-ZY).

**Principe directeur** : extension additive stricte. Defaults `per_enabled=False` partout. 265 tests baseline passent inchangés. V2-Y et V2-ZY baselines reproductibles strict sans `--per`. Pattern V2-W/V2-U.

---

## 5. Composants détaillés

### 5.1 `SumTree`

Structure de données pour O(log N) sample + update.

**Convention indexation** (acceptée pour toute capacity ≥ 1) :
- Array de taille `2 × capacity − 1`
- Feuilles aux indices `[capacity − 1, 2 × capacity − 2]`
- Nœuds internes `[0, capacity − 2]`
- `parent(i) = (i − 1) // 2`, `left(i) = 2i + 1`, `right(i) = 2i + 2`

**Interface** :

```python
class SumTree:
    def __init__(self, capacity: int) -> None: ...
    def update(self, leaf_idx: int, priority: float) -> None: ...  # propage sum vers racine
    def total(self) -> float: ...                                  # somme à la racine
    def find(self, value: float) -> tuple[int, float]: ...         # descend, retourne (leaf_idx, priority)
    
    @property
    def capacity(self) -> int: ...
```

**Validation** : `capacity > 0` requis. `leaf_idx ∈ [0, capacity − 1]` (interface utilisateur exprime en termes de slots, pas d'indices internes du tableau).

**Invariants testés** :
- `total()` == somme des priorités leaves après N updates
- `find(value)` retourne le bon leaf pour distribution proportionnelle
- Edge cases : capacity=1, capacity=5000 (default V2-ZY 10×10), capacity=2500 (V2-ZY 15×15)
- Distribution convergente : 10000 samples, fréquence empirique converge vers priorité normalisée à ±3 %

### 5.2 `PrioritizedSequenceReplayBuffer`

**Interface** :

```python
from dataclasses import dataclass
from mw_ia.neural.sequence_buffer import BatchSeq

@dataclass
class PrioritizedBatchSeq:
    batch: BatchSeq              # BatchSeq standard V2-Y (states/actions/rewards/.../mask)
    weights: np.ndarray          # (B,) float32 — IS weights, normalisés par max(w)
    tree_indices: np.ndarray     # (B,) int64 — leaf indices pour update_priorities

class PrioritizedSequenceReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        max_steps: int = 200,
        *,
        alpha: float = 0.6,
        epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None: ...
    
    def __len__(self) -> int: ...
    
    def push_trajectory(self, trajectory: list[tuple]) -> None:
        """Stocke trajectoire (idem SequenceReplayBuffer V2-Y).
        Priorité initiale = self._max_priority (greedy init pour nouvelle trajectoire).
        """
    
    def sample(self, batch_size: int, seq_len: int, beta: float) -> PrioritizedBatchSeq:
        """Sampling stratifié sum tree + extraction fenêtre seq_len + IS weights normalisés."""
    
    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """td_errors: (B,) — agrégat R2D2 par trajectoire calculé côté trainer.
        new_priority = (|td_error| + epsilon) ** alpha
        Met à jour sum tree et _max_priority si dépassé.
        """
```

**Storage interne** : arrays identiques à `SequenceReplayBuffer` V2-Y (states/actions/rewards/next_states/dones/lengths shape `(capacity, max_steps, ...)`) + `SumTree(capacity)` + `_max_priority: float = 1.0`.

**Sampling stratifié** (réduit variance vs proportional pur — implémentation Schaul reference) :
1. `total = sum_tree.total()`
2. Pour `b in range(batch_size)` :
   - `segment_low = b × total / batch_size`
   - `segment_high = (b + 1) × total / batch_size`
   - `value ~ U(segment_low, segment_high)`
   - `(leaf_idx, priority) = sum_tree.find(value)`
3. Construction `BatchSeq` (offset aléatoire dans trajectoire + padding + mask, idem V2-Y)
4. IS weights : `P_i = priority_i / total`, `w_i = (1/N × 1/P_i)^beta`, normalisés `w_i / max(w_j)`

**Décisions clés** :
- α baked at update time (priority stockée = `(|td| + ε)^α`) ; évite recompute à chaque sample
- IS normalisation par `max(w)` (stabilité numérique > correction théorique parfaite)
- ε > 0 garantit `priority > 0` pour toute trajectoire (sinon trajectoires deviennent non-samplables)
- `_max_priority` tracker (nouvelle trajectoire reçoit max courant ; sinon priority 0 → jamais samplée)
- β passé en argument de `sample()` (pas stocké) ; l'agent l'anneal au fil des épisodes
- Pas d'agrégation R2D2 dans le buffer ; le trainer a accès mask + per-step TD-errors. Buffer reste générique
- `PrioritizedBatchSeq` wrapper plutôt qu'extension `BatchSeq` (préserve strictement le type V2-Y baseline)

**Tests** (~12 tests dans `test_prioritized_sequence_buffer.py`) :
- Push trajectoire → priorité = max courant
- Sample stratifié : distribution proportionnelle aux priorités (10000 samples)
- IS weights normalisés : `weights.max() == 1.0 ± 1e-6`
- IS weights décroissent quand β augmente
- `update_priorities` met à jour correctement le sum tree
- Première trajectoire est sampleable (greedy init)
- Buffer circulaire : nouvelle trajectoire écrase l'ancienne, sum tree cohérent
- Edge case : sample avec `batch_size > len(buffer)` → ValueError
- Edge case : sample avec `seq_len > max_steps` → ValueError
- Reproductibilité : même seed → même séquence de samples
- Buffer à capacity 5000 ET 2500 (V2-ZY 10×10 vs 15×15)
- Backwards compat : interface ne casse pas si appelée avec arg manquant

### 5.3 `BetaScheduler`

```python
class BetaScheduler:
    """Annealing linéaire β_start → β_end sur total_episodes."""
    def __init__(self, beta_start: float, beta_end: float, total_episodes: int) -> None:
        # Validation : β ∈ [0,1], total_episodes > 0
        ...
    
    def beta(self, episode: int) -> float:
        if episode <= 0:
            return self.beta_start
        if episode >= self.total_episodes:
            return self.beta_end
        return self.beta_start + (self.beta_end - self.beta_start) * (episode / self.total_episodes)
```

**Placé dans** `prioritized_sequence_buffer.py` (cohésion PER, évite prolifération de fichiers).

**Tests** (~4 tests) :
- `beta(0) == beta_start`
- `beta(total_episodes) == beta_end`
- `beta(total_episodes + N) == beta_end` (clamp)
- Validation `beta_start ∈ [0,1]`, `beta_end ∈ [0,1]`, `total_episodes > 0`

### 5.4 Extension `RecurrentDQNTrainer`

**Stratégie** : factor le pipeline `step()` existant en `_step_impl(batch, weights=None, eta=0.9) → (loss, td_errors|None)`. Puis :
- `step(batch) → float` — appel sans IS, signature V2-Y inchangée (35+35 tests préservés)
- `step_with_priorities(batch, weights, eta=0.9) → (loss, td_errors)` — appel avec IS et aggrégation

**Pipeline interne** (modifications par rapport à `step()` actuel) :

```python
def _step_impl(self, batch, weights=None, eta=0.9):
    # === Lignes 64-91 INCHANGÉES : tensors to device, forward online, target Q (Double DQN ou V2-Y) ===
    
    with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
        # ... q_pred, target_q calculés comme actuellement ...
        
        elem_loss = self.loss_fn(q_pred, target_q)  # (seq, batch)
        
        # === NOUVEAU : IS weights appliqués au numérateur ===
        if weights is None:
            masked_loss = elem_loss * mask
        else:
            w = torch.from_numpy(weights).to(self.device, non_blocking=True).unsqueeze(0)  # (1, batch)
            masked_loss = elem_loss * mask * w
        
        n_valid = mask.sum().clamp(min=1.0)
        loss = masked_loss.sum() / n_valid
    
    # === Backward + grad clip + optimizer step + Polyak : INCHANGÉ ===
    
    # === NOUVEAU : aggrégation R2D2 si PER ===
    if weights is None:
        return float(loss.detach().item()), None
    
    with torch.no_grad():
        td_step = (target_q - q_pred).detach().abs()        # (seq, batch)
        masked_td = td_step * mask                          # (seq, batch)
        max_per_traj = masked_td.max(dim=0).values          # (batch,)
        sum_per_traj = masked_td.sum(dim=0)                 # (batch,)
        length_per_traj = mask.sum(dim=0).clamp(min=1.0)    # (batch,)
        mean_per_traj = sum_per_traj / length_per_traj
        priorities = eta * max_per_traj + (1 - eta) * mean_per_traj
    
    return float(loss.detach().item()), priorities.cpu().numpy().astype(np.float32)


def step(self, batch: BatchSeq) -> float:
    loss, _ = self._step_impl(batch, weights=None)
    return loss

def step_with_priorities(
    self,
    batch: BatchSeq,
    weights: np.ndarray,
    eta: float = 0.9,
) -> tuple[float, np.ndarray]:
    loss, td_errors = self._step_impl(batch, weights=weights, eta=eta)
    return loss, td_errors
```

**Décisions clés** :
- Pas de duplication de logique : `_step_impl` unique, wrappers de 1 ligne
- IS appliqué AU NUMÉRATEUR uniquement (formule Schaul standard)
- Aggregation hors autocast (`torch.no_grad()`) ; pas de gradient à propager
- TD-error = `target_q − q_pred` ; `target_q` déjà détaché (ligne 79-91 existante), q_pred détaché via `.detach()` pour aggregation
- Polyak update inchangé (ligne 112-113 actuelle) ; s'applique aux deux chemins (PER ou non)
- Retour `.astype(np.float32)` explicite (évite ambiguïté dtype)

**Tests `test_per_trainer.py`** (~8 tests, ~80 LOC) :
- `step_unchanged_signature_returns_float` — V2-Y compat
- `step_with_priorities_returns_tuple` — type retour `(float, ndarray)` shape `(batch,)`
- `is_weights_change_loss` — loss avec `weights=[2,1,1,1]` ≠ loss avec `[1,1,1,1]`
- `uniform_weights_match_step` — `step_with_priorities(batch, ones, eta).loss ≈ step(batch).loss` (tolerance AMP)
- `td_errors_r2d2_aggregation` — vérifie `priority = 0.9 × max + 0.1 × mean` sur batch synthétique connu
- `mask_excludes_padded_steps_from_aggregation` — trajectoire courte ne contamine pas max/mean
- `double_dqn_path_with_per` — PER + Double DQN cohabitent
- `polyak_with_per` — PER + Polyak cohabitent (target update post-backward inchangé)

### 5.5 Extension agents (V2-Y `RecurrentDQNAgent` + V2-ZY `ConvRecurrentDQNAgent`)

Pattern parallèle strict (cohérent V2-W flag + V2-U Polyak).

**Constructor** (modification après ligne 59 V2-Y, parallèle V2-ZY) :

```python
if cfg.per_enabled:
    self.buffer = PrioritizedSequenceReplayBuffer(
        cfg.replay_capacity, obs_dim, cfg.max_steps_per_episode,
        alpha=cfg.per_alpha, epsilon=cfg.per_epsilon, seed=seed,
    )
    self._beta_scheduler: BetaScheduler | None = BetaScheduler(
        cfg.per_beta_start, cfg.per_beta_end, cfg.episodes,
    )
else:
    self.buffer = SequenceReplayBuffer(
        cfg.replay_capacity, obs_dim, cfg.max_steps_per_episode, seed=seed,
    )
    self._beta_scheduler = None
self._episode_count: int = 0
```

**`end_episode()`** — branche conditionnelle après seuil buffer :

```python
def end_episode(self) -> dict[str, float]:
    if self._episode_trajectory:
        self.buffer.push_trajectory(self._episode_trajectory)
    self._episode_count += 1
    
    metrics: dict[str, float] = {"epsilon": self.epsilon}
    if len(self.buffer) >= max(self.cfg.min_episodes_to_learn, self.cfg.batch_size):
        losses: list[float] = []
        if self.cfg.per_enabled:
            beta = self._beta_scheduler.beta(self._episode_count)
            for _ in range(self.cfg.train_steps_per_episode):
                prio_batch = self.buffer.sample(
                    self.cfg.batch_size, self.cfg.sequence_length, beta=beta,
                )
                loss, td_errors = self.trainer.step_with_priorities(
                    prio_batch.batch, prio_batch.weights, eta=self.cfg.per_eta,
                )
                self.buffer.update_priorities(prio_batch.tree_indices, td_errors)
                losses.append(loss)
            metrics["per_beta"] = beta
        else:
            # === Boucle V2-Y existante (lignes 117-122) INCHANGÉE ===
            for _ in range(self.cfg.train_steps_per_episode):
                batch = self.buffer.sample(self.cfg.batch_size, self.cfg.sequence_length)
                losses.append(self.trainer.step(batch))
        if losses:
            self.last_loss = sum(losses) / len(losses)
            metrics["loss"] = self.last_loss
    
    # === V2-U Polyak skip hard sync : lignes 128-131 INCHANGÉES ===
    if self.cfg.polyak_tau == 0.0:
        if self.global_step // self.cfg.target_sync_steps > self.target_syncs:
            self.trainer.sync_target()
            self.target_syncs += 1
    return metrics
```

**Décisions clés** :
- `_episode_count` séparé de `len(buffer)` : le buffer plafonne à `replay_capacity` mais β doit anneal sur `cfg.episodes` strictement
- β publié dans `metrics["per_beta"]` (debug + extension GUI future possible)
- `save()` / `load()` inchangés ; priorité PER vit dans le buffer, jamais sauvegardée (cohérent V1/V2-Y pattern)
- V2-V `PeriodicEvaluator` + `BestCheckpointTracker` compat strict (eval n'utilise pas buffer)
- Polyak + PER orthogonaux : `trainer.step_with_priorities` appelle `polyak_update` post-backward exactement comme `step()`

**Tests `test_per_recurrent_agents.py`** (~8 tests pour V2-Y, parallèle 8 pour V2-ZY) :
- `per_disabled_instantiates_sequence_buffer` — comportement V2-Y/ZY strict
- `per_enabled_instantiates_prioritized_buffer` + `_beta_scheduler` non-None
- `end_episode_per_path_calls_step_with_priorities` (mocked)
- `end_episode_per_path_calls_update_priorities` (mocked)
- `episode_count_increments_independently_of_buffer_len` (au-delà capacity)
- `beta_decreases_with_episode_count`
- `polyak_and_per_cohabit` (target update post chaque train_step)
- `save_load_unchanged_with_per` (PER state non sauvegardé, intentionnel)

---

## 6. Configuration & CLI

### 6.1 Nouveaux champs config

Ajoutés à `DRQNConfig` (V2-Y) ET `ConvRecurrentDQNConfig` (V2-ZY), defaults identiques :

```python
per_enabled: bool = False
per_alpha: float = 0.6           # priority exponent (Schaul 2015)
per_beta_start: float = 0.4      # IS exponent initial (Schaul 2015)
per_beta_end: float = 1.0        # IS exponent final (annealing complet)
per_eta: float = 0.9             # R2D2 aggregation eta (Kapturowski 2019)
per_epsilon: float = 1e-6        # small constant pour garantir priority > 0
```

**Validation** dans `__post_init__` :
- `per_alpha ∈ [0, 1]`
- `per_beta_start ∈ [0, 1]`, `per_beta_end ∈ [0, 1]`
- `per_eta ∈ [0, 1]`
- `per_epsilon > 0` strict

### 6.2 Flags CLI

Ajoutés à `scripts/train_drqn_procedural.py` ET `scripts/train_cnn_lstm_dqn_procedural.py` :

```python
parser.add_argument(
    "--per",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="V2-B0 : Prioritized Experience Replay trajectoire-level (Schaul 2015 + R2D2). "
         "Default False = SequenceReplayBuffer uniforme baseline V2-Y/V2-ZY.",
)
parser.add_argument("--per-alpha", type=float, default=0.6,
                    help="V2-B0 : priority exponent alpha (default 0.6, Schaul 2015).")
parser.add_argument("--per-beta-start", type=float, default=0.4,
                    help="V2-B0 : IS exponent beta initial (default 0.4, Schaul 2015).")
parser.add_argument("--per-beta-end", type=float, default=1.0,
                    help="V2-B0 : IS exponent beta final (default 1.0, annealing complete).")
parser.add_argument("--per-eta", type=float, default=0.9,
                    help="V2-B0 : R2D2 priority aggregation eta (default 0.9).")
parser.add_argument("--per-epsilon", type=float, default=1e-6,
                    help="V2-B0 : small constant epsilon (default 1e-6) garantit priority > 0.")
parser.add_argument(
    "--max-attempts-bfs",
    type=int,
    default=100,
    help="ProceduralEnvConfig max_attempts_bfs (default 100). Recommande bench B0 : 500 "
         "pour eviter crash density >= 0.43 (cf. CLAUDE.md piege #10).",
)
```

**Help-text ASCII strict** (Windows cp1252 compat) : "alpha", "beta", "epsilon", "tau" en mots ; pas de glyphes Unicode.

**Propagation au config** : ~7 lignes additionnelles dans le `main()` de chaque script.

### 6.3 Recette CLI bench B0 (V2-ZY+Polyak+PER)

```bash
# 10x10 (sanity / no regression)
python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed {N} \
    --polyak-tau 0.005 --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b0_10x10_seed{N}.pt

# 15x15 (test scientifique)
python scripts/train_cnn_lstm_dqn_procedural.py \
    --episodes 5000 --mode obstacles --device cuda --seed {N} \
    --max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500 \
    --polyak-tau 0.005 --per --max-attempts-bfs 500 \
    --best-checkpoint-path checkpoints/v2b0_15x15_seed{N}.pt
```

---

## 7. Tests

**48 nouveaux tests pytest** distribués sur 5 fichiers (les 16 tests agents sont 8 parametrized × 2 agents V2-Y + V2-ZY) :

| Fichier | Tests | Couverture |
|---|---|---|
| `tests/neural/test_sum_tree.py` | 8 | init, update, total, find, distribution convergente, edge cases (capacity=1, 5000, 2500) |
| `tests/neural/test_prioritized_sequence_buffer.py` | 12 | push (greedy init), sample (stratifié + IS), update_priorities, reproductibilité seed, buffer circulaire, edge cases |
| `tests/neural/test_beta_scheduler.py` | 4 | annealing linéaire, clamps, validation params |
| `tests/neural/test_per_trainer.py` | 8 | step backward compat, step_with_priorities tuple, IS effect, R2D2 aggregation, mask, Double DQN + PER, Polyak + PER |
| `tests/agents/test_per_recurrent_agents.py` | 16 (8 V2-Y + 8 V2-ZY) | buffer instantiation conditionnelle, end_episode PER path, episode_count, beta annealing, save/load |

**Backwards compat** : `pytest -q` total attendu **313 passed** (265 baseline + 48).

**`bash aether/verify_all.sh`** inchangé (toujours 8 OK).

---

## 8. CI smoke

Ajout à `.github/workflows/aether_verify.yml` :

```yaml
- name: Smoke test V2-B0 PER (trajectory-level) on V2-ZY
  run: |
    mkdir -p checkpoints
    python scripts/train_cnn_lstm_dqn_procedural.py \
      --episodes 10 --mode obstacles --device cpu \
      --per --eval-every-episodes 5 \
      --best-checkpoint-path checkpoints/ci_v2b0_best.pt
    test -f checkpoints/ci_v2b0_best.pt

- name: Smoke test V2-B0 PER + Polyak cohabit (sanity check)
  run: |
    mkdir -p checkpoints
    python scripts/train_cnn_lstm_dqn_procedural.py \
      --episodes 10 --mode obstacles --device cpu \
      --per --polyak-tau 0.005 --eval-every-episodes 5 \
      --best-checkpoint-path checkpoints/ci_v2b0_polyak_best.pt
    test -f checkpoints/ci_v2b0_polyak_best.pt
```

**Pas de smoke V2-Y PER** : pertinence scientifique faible (V2-Y baseline plafonné à diff=0.05), économise ~30 s CI. Test unitaire `test_per_recurrent_agents.py::test_per_enabled_path_v2y` couvre la voie code V2-Y.

---

## 9. Protocole de bench scientifique

### 9.1 Baselines de référence (déjà collectées V2-U)

| Métrique | V2-ZY+Polyak 10×10 (n=5) | V2-ZY+Polyak 15×15 (n=5) |
|---|---|---|
| Mean best @ diff=0.30 | 92 % | 64 % |
| Std (n−1) | 13.0 pp | 13.4 pp |
| Min | 70 % | 50 % |
| Max | 100 % | 80 % |
| Late-stage collapse | 0/5 | 0/5 |
| Diff_max training (mean) | ~0.65 | ~0.36 |

### 9.2 Protocole same-seed n=5

Pattern V2-U reproduit strictement : seeds 0-4, eval rigoureux V2-V (best @ diff=0.30 fixe greedy sur 10 seeds held-out 10000-10009), variable unique changée = `--per`.

**Phase 1 — 10×10 sanity** : 5 runs V2-ZY+Polyak+PER, comparaison directe vs baseline V2-U 10×10.

**Phase 2 — 15×15 test réel** : 5 runs V2-ZY+Polyak+PER avec `--max-rows 15 --max-cols 15 --max-steps 400 --replay-capacity 2500`, comparaison vs baseline V2-U 15×15.

**Compute estimé** : 10 runs × ~30-45 min RTX 3060 = 5-7.5 h GPU total.

### 9.3 Critères d'acceptance (pré-enregistrés, pas de goalpost moving)

**Phase 1 — 10×10 = no-regression check** :
- Mean ≥ **85 %** (tolère −7 pp vs baseline 92 %, dans la variance d'échantillonnage)
- Std ≤ **20 pp** (variance ne doit pas exploser)
- Late-stage collapse = **0/5** (must match baseline)
- Diff_max training mean ≥ **0.5** (capacité préservée)

Si ❌ phase 1 → **B0 rejeté** ("PER casse le régime stable validé"). Pas de phase 2. Investigation requise.

**Phase 2 — 15×15 = test scientifique réel** :

Au moins UN des critères suivants pour valider "PER aide" :

| Critère | Seuil | Interprétation |
|---|---|---|
| Mean amélioré | > 64 % | Gain global |
| Min amélioré | > 50 % | Worst-case réduit |
| Convergence accélérée | Médiane `ep_to_best` < baseline médiane | PER converge plus vite |
| Diff_max training | Mean > 0.36 | Scheduler franchit plus haut |

**Outcomes possibles** :

| Critères 15×15 atteints | Verdict B0 | Décision suivante |
|---|---|---|
| 0/4 | PER ne franchit pas le plafond 15×15 | Tag `v0.2.0-b0` (livraison code valide), brainstorm B1 directement |
| ≥ 1/4 | PER aide | Documenter critère + magnitude + hypothèse mécaniste, brainstorm B1 |
| ≥ 3/4 | Finding publishable | Cascade Conv + LSTM + Double DQN + Polyak + PER confirmée |

---

## 10. Pièges & mitigations

### 10.1 Pièges d'implémentation (à attraper par TDD)

| # | Piège | Impact si raté | Détection |
|---|---|---|---|
| 1 | `abs(td_error)` oublié avant `^α` | Crash sur td négatif | `test_td_errors_r2d2_aggregation` avec td négatif |
| 2 | Sum tree off-by-one (convention indexation) | Sampling biaisé silencieux | `test_sum_tree_distribution` (10000 samples) |
| 3 | `_max_priority` non initialisé à 1.0 | Première trajectoire priority=0 → jamais samplée | `test_first_trajectory_sampleable` |
| 4 | TD-error stocké non détaché du graphe autograd | Memory leak progressif, OOM ~500 ép | `test_step_with_priorities_no_grad_through_priorities` |
| 5 | IS weights pas normalisés par max(w) | Gradients explosifs sur trajectoires rares | Assert `weights.max() == 1.0 ± 1e-6` post-normalisation buffer |
| 6 | β annealing off-by-one | β_end pas atteint, ou dépassement | `test_beta_at_zero`, `test_beta_at_total`, `test_beta_overshoot_clamped` |
| 7 | `_episode_count` confondu avec `len(buffer)` | β n'avance plus une fois buffer plein | `test_episode_count_independent_of_buffer_len` |
| 8 | Sum tree update pendant sample | Distribution change mid-batch | Pattern Schaul : sample → step → update sequentiel |
| 9 | AMP float16 cast des IS weights | Précision perdue, IS bruyant | Test sous CPU (no AMP) ET CUDA (AMP) |
| 10 | `PrioritizedBatchSeq.batch` vs raw `BatchSeq` au trainer | TypeError ou résultats faux | Type hints stricts dans signatures |
| 11 | `update_priorities` après `push_trajectory` qui a écrasé le slot | Met à jour priorité de la nouvelle trajectoire (écraseur) au lieu de l'ancienne (écrasée) | Contrat sample → train → update atomique au sein d'un train_step. Documenté docstring buffer |
| 12 | `replay_capacity` 5000 vs 2500 (10×10 vs 15×15) | Sum tree dimensionné mal | Constructeur lit `capacity` arg explicit, pas hardcode |
| 13 | `--per-epsilon` user passe 0 → priorities peuvent être 0 | Trajectoires jamais re-samplées | Validation `per_epsilon > 0` strict dans config + assert buffer constructor |

### 10.2 Pièges de bench (à anticiper en exécution)

| # | Piège | Probabilité | Mitigation |
|---|---|---|---|
| B1 | Hyperparams Schaul (α=0.6) sous-optimaux pour ce régime | moyenne | Defaults littéraires. Si null result, grid search α ∈ {0.4, 0.8} sur 1 seed post-bench |
| B2 | Granularité trajectoire trop grossière | moyenne | Si null result ET PER aide V2-W (test side) → revisiter en B0.1 step-within-trajectory |
| B3 | β annealing 0.4→1.0 over-corrects fin d'entraînement | basse-moyenne | Documenté. Si null result borderline, retest β fixe 0.5 sur 1 seed |
| B4 | 5000 ép trop court pour signal PER | basse | V2-ZY+Polyak baseline déjà saturée à 5000 ép sur 10×10. 15×15 a headroom |
| B5 | `replay_capacity=2500` à 15×15 réduit le bénéfice | basse | Contrainte VRAM. Documenté. Pas de blocker |
| B6 | `RandomObstaclesGenerator` crash density=0.43+ (piège #10 V2-U seed 1) | moyenne-haute | **Pre-mitigation** : flag CLI `--max-attempts-bfs 500` exposé, recette B0 l'utilise |
| B7 | PER + Polyak antagonistes (analogue V2-Y scheduler) | basse-moyenne | Smoke CI vérifie cohabitation no-crash. Bench tranche empiriquement |
| B8 | n=5 insuffisant pour variance reduction modeste | moyenne | Documenté. Pré-enregistrement des critères évite tentation de re-narrer. Si finding ambigu, n=10 possible |

---

## 11. Definition of Done

### 11.1 Code & tests

- [ ] 48 nouveaux tests pytest verts (sum tree 8 + prioritized buffer 12 + beta scheduler 4 + trainer 8 + agents 16 dont 8 parametrized × V2-Y + V2-ZY)
- [ ] `pytest -q` total : **313 passed** (265 baseline + 48)
- [ ] `bash aether/verify_all.sh` → 8 OK (inchangé)
- [ ] V2-Y baseline reproductible strict sans `--per` (smoke 5 ép, loss/winrate cohérents avec baseline existante)
- [ ] V2-ZY baseline idem
- [ ] CI : 2 nouveaux smoke jobs (PER + PER+Polyak) passent sur CPU
- [ ] `git log` : commits TDD bite-sized par phase (pattern V2-U)

### 11.2 Bench scientifique

- [ ] Phase 1 — 10×10 n=5 : **acceptance no-regression PASSED** (mean ≥ 85 %, std ≤ 20 pp, 0/5 collapse, diff_max ≥ 0.5)
- [ ] Phase 2 — 15×15 n=5 : bench exécuté et **documenté** (verdict positif OU négatif honnête)
- [ ] Tableaux résultats par seed + agrégats (n=5) intégrés à CLAUDE.md section V2-B0
- [ ] Story scientifique consolidée 3-5 lignes (pattern V2-U)

### 11.3 Doc

- [ ] Spec dans `docs/superpowers/specs/2026-05-26-mw-ia-per-trajectory-design.md` + commit
- [ ] Plan TDD via writing-plans dans `docs/superpowers/plans/2026-05-26-mw-ia-per-trajectory.md`
- [ ] Section V2-B0 dans CLAUDE.md (état des phases + composants + décisions techniques + pièges + recette CLI + bench results)
- [ ] README mis à jour avec recette B0

### 11.4 Tag

- [ ] `git tag v0.2.0-b0` après DoD ci-dessus

---

## 12. Décisions reportées post-bench

| Outcome 15×15 | Décision suivante |
|---|---|
| ≥ 1 critère atteint | Tag `v0.2.0-b0`, brainstorm **B1 (Policy snapshot rehearsal)** |
| 0 critère atteint, V2-W PER positif (test side optionnel) | B0.1 step-within-trajectory PER avant B1 |
| 0 critère atteint, V2-W PER négatif | Tag `v0.2.0-b0` avec finding négatif documenté, brainstorm **B1 directement** (le replay non-uniforme n'est pas le bottleneck — la mémoire structurelle l'est probablement) |

---

## 13. Hyperparams canoniques (référence littéraire)

| Param | Valeur B0 | Source |
|---|---|---|
| α (priority exponent) | 0.6 | Schaul et al. 2015, "Prioritized Experience Replay", §3.3 (sweep 0.4-0.7, 0.6 optimal Atari) |
| β_start (IS exponent) | 0.4 | Schaul et al. 2015, §3.3 |
| β_end (IS exponent) | 1.0 | Standard annealing (full IS correction at end of training) |
| η (R2D2 aggregation) | 0.9 | Kapturowski et al. 2019, "Recurrent Experience Replay in Distributed Reinforcement Learning" (Appendix A) |
| ε (priority floor) | 1e-6 | Schaul et al. 2015, §3.3 (small positive constant) |

---

## Appendix A — Formules clés

**Priority (Schaul)** :
```
p_i = (|delta_i| + epsilon)^alpha   where alpha in [0, 1]
P_i = p_i / sum_j p_j
```

**Importance Sampling weight (Schaul)** :
```
w_i = (1/N * 1/P_i)^beta   where beta in [0, 1]
w_i_normalized = w_i / max_j w_j
```

**Loss with IS (sequence-level adapté)** :
```
loss = sum_{t,b} ( huber(delta_{t,b}) * mask_{t,b} * w_b ) / sum_{t,b} mask_{t,b}
```

**Priority aggregation R2D2** :
```
priority_b = eta * max_t |delta_{t,b} * mask_{t,b}|
           + (1 - eta) * mean_t |delta_{t,b} * mask_{t,b}|
```

**Beta annealing** :
```
beta(episode) = beta_start + (beta_end - beta_start) * (episode / total_episodes)
              clamped to [beta_start, beta_end]
```
