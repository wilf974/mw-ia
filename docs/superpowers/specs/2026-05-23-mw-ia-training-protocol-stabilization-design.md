# Spec V2-V — Training Protocol Stabilization (MVP)

> **Sous-projet** : V2-V (MW_IA — Reinforcement Learning éducatif)
> **Date** : 2026-05-23
> **Statut** : Spec validée — implémentation à dérouler via `superpowers:writing-plans` puis `superpowers:subagent-driven-development`
> **Tag livraison cible** : `v0.2.0-v`
> **Sous-projets prérequis livrés** : V1, V2-A, V2-X, V2-Y, V2-Z, V2-W

---

## 1. Vue d'ensemble & motivation

### Motivation empirique consolidée (H1 confirmée 2026-05-23)

Test `--episodes 3000` vs `--episodes 5000` sur V2-W seeds 0-4 same-seed :

| Seed | V2-W ep=5000 | V2-W ep=3000 | Lecture |
|---|---|---|---|
| 0 | 62 % @ diff=0.40 | 67 % @ diff=0.30 | léger gain à ep=3000 |
| 1 | 68 % @ diff=0.35 | 72 % @ diff=0.30 | léger gain |
| 2 | 81 % @ diff=0.40 | 87 % @ diff=0.30 | léger gain |
| 3 | 49 % @ diff=0.15 (crash partiel) | 58 % @ diff=0.30 | gain notable |
| **4** | **1 % @ diff=0.10** (collapse total) | **71 % @ diff=0.30** | **sauvé** (+70 pp) |

**Statistique agrégée V2-W ep=3000 (n=5)** : tous les seeds convergent à diff=0.30, std=0.000, bucket 1 moyen 71 %.

### Le problème identifié

Le pipeline RL "train until end" est cassé pour ce setup procédural. **Seed 4 V2-W** est la preuve causale : même seed, même init, même env, seule différence = stopper l'entraînement à ep 3000 au lieu de ep 5000. Résultat : `1 %` → `71 %`. L'entraînement après ep 3000 détruit littéralement l'agent.

Sans best-checkpoint tracking + évaluation périodique :
- Tous les benchmarks RL futurs sont biaisés par le timing arbitraire d'arrêt
- Le meilleur agent entraîné est jeté
- L'effet vrai des futurs tuning (Polyak, lr, archi) reste mesuré sur des artefacts de timing

### Objectif V2-V

Passer de **"train until end"** à **"train while tracking, save best, judge by best"**.

### Hypothèse V2-V

> En évaluant périodiquement l'agent en mode greedy sur un set de seeds eval séparé du training, et en sauvegardant automatiquement le modèle au pic d'eval_winrate, on récupère le meilleur agent atteint pendant l'entraînement **sans connaître a priori le bon ep_max**. Le re-benchmark V2-W same-seed avec V2-V devrait montrer que seed 4 sauvegarde un best-checkpoint avec winrate ~71 % (proche de la mesure ep=3000).

### Scope MVP — 2 features critiques

1. **Évaluation périodique greedy** : toutes les 100 ép, exécuter 10 rollouts greedy (`agent.act(..., greedy=True)`, pas de `observe()`, pas de pollution du buffer) sur seeds eval séparés du training (seeds 10000-10009), à la difficulty courante du scheduler.

2. **Best-checkpoint tracking** : maintenir `best_eval_winrate`. Quand un nouvel eval bat le meilleur, sauvegarder le modèle (`agent.save(best_path)`).

### Hors scope MVP (reportés post-livraison)

- Early stopping (arrêt si pas d'amélioration sur N éval consécutives)
- Moving average metrics (lissage de la décision de save)
- Rollback automatique (restaurer best-model si collapse détecté)
- Évaluation multi-difficulty (caractérisation de la courbe de capacité)
- Brancher sur V2-X/V2-Y runners (le composant `evaluator.py` sera réutilisable)

### Cible scientifique

Re-benchmark V2-W n=5 à ep=5000 AVEC eval+best-checkpoint activés. Si seed 4 → `best_checkpoint_winrate` ≈ 71 % (proche de la mesure ep=3000), V2-V valide son utilité immédiatement.

---

## 2. Architecture & composants

### Pattern d'intégration

Nouveau composant `PeriodicEvaluator` séparé, branché sur `ConvProceduralDQNRunner` uniquement (MVP). V2-X et V2-Y runners restent intacts — réutilisation future facile.

### Nouveaux fichiers code

| Fichier | Rôle |
|---|---|
| `mw_ia/training/evaluator.py` | `PeriodicEvaluator` class : env eval séparé, eval seeds fixes, méthode `evaluate(agent, difficulty) -> dict[str, float]`. Garantit zéro pollution du buffer training. |
| `mw_ia/training/checkpoint_tracker.py` | `BestCheckpointTracker` class : maintient `best_eval_winrate` + path, méthode `update(eval_metrics, agent) -> bool` qui sauvegarde si nouveau meilleur. |
| `tests/training/test_evaluator.py` | 8 tests TDD pour `PeriodicEvaluator` |
| `tests/training/test_checkpoint_tracker.py` | 6 tests TDD pour `BestCheckpointTracker` |

### Extensions de fichiers existants

| Fichier | Extension |
|---|---|
| `mw_ia/config.py` | + 5 champs dans `ConvDQNConfig` : `eval_enabled: bool = True`, `eval_every_episodes: int = 100`, `eval_seeds: tuple[int, ...] = tuple(range(10_000, 10_010))`, `eval_max_steps: int = 200`, `best_checkpoint_path: str \| None = None`. Validation `__post_init__` pour `eval_every_episodes > 0`, `eval_seeds` non-vide. |
| `mw_ia/training/runner.py` | + extension `ConvProceduralDQNRunner.__init__` : instancie `PeriodicEvaluator` + `BestCheckpointTracker` si `dqn_cfg.eval_enabled`. + dans `run()` : appel `evaluator.evaluate(...)` + `tracker.update(...)` tous les `eval_every_episodes` épisodes. + `RunnerCallbacks.on_eval`. |
| `scripts/train_cnn_dqn_procedural.py` | + 3 flags CLI : `--eval / --no-eval` (default `--eval`), `--eval-every-episodes 100`, `--best-checkpoint-path checkpoints/v2v_best_<timestamp>.pt`. |
| `tests/test_conv_dqn_config.py` | + 3 tests pour nouveaux champs eval (defaults + validation + Aether compat inchangée) |
| `tests/training/test_conv_procedural_runner.py` | + 1 test smoke : runner avec eval activé pendant 200 ép → eval ≥ 3 fois, best-checkpoint sauvegardé |

### Décisions d'API

**`PeriodicEvaluator(eval_env, eval_seeds, max_steps, observation_encoder, proc_cfg)`** : constructeur keyword-only. `observation_encoder` est la fonction `encode_procedural_observation_2d` (V2-Z) — injectée pour découpler du runner.

**`PeriodicEvaluator.evaluate(agent, difficulty) -> dict[str, float]`** :

```python
{
    "winrate": float,        # in [0, 1], = n_success / len(eval_seeds)
    "mean_reward": float,
    "mean_length": float,
    "n_episodes": int,       # = len(eval_seeds), typiquement 10
    "difficulty": float,     # difficulty utilisée pour cet eval
}
```

**`BestCheckpointTracker(path: str | Path | None)`** : constructeur prend juste le chemin de sauvegarde (None = tracking en mémoire seulement, pas d'IO).

**`BestCheckpointTracker.update(eval_metrics: dict, agent: ConvDQNAgent, episode: int) -> bool`** : retourne `True` si nouveau best sauvegardé, `False` sinon. Idempotent (égalité ne triggers pas save).

**`BestCheckpointTracker.best_winrate: float`** : propriété, lit le meilleur winrate observé (-inf si aucun update).
**`BestCheckpointTracker.best_episode: int | None`** : épisode du meilleur eval.
**`BestCheckpointTracker.best_difficulty: float | None`** : difficulty au moment du meilleur eval.

### Pas de modification de l'agent

L'API existante `agent.act(obs, greedy=True)` suffit. L'agent ne sait pas qu'il est évalué — c'est l'externe (PeriodicEvaluator) qui pilote. Pas de nouveau format checkpoint : réutilise `agent.save(path)` V1 hérité.

### Séparation training/eval stricte

- Eval env est une instance distincte de `ProceduralGridWorld` créée à l'init de `PeriodicEvaluator`
- Eval n'appelle JAMAIS `agent.observe()` — pas de buffer push, pas de global_step increment, pas de scheduler update
- Eval utilise `agent.act(obs, greedy=True)` qui bypass l'eps-greedy et le rng training (vérifié dans le code V2-Z : `if (not greedy) and self._rng.random() < ...`)
- `torch.no_grad()` autour du forward (pas de grad accumulation) — déjà géré par `agent.act()` interne
- Eval seeds 10000-10009 ne chevauchent JAMAIS les seeds training 0..episodes-1

---

## 3. Data flow détaillé

### Initialisation `ConvProceduralDQNRunner.__init__`

```python
super().__init__(train_cfg, callbacks)
# ... scheduler, bucket_tracker, agent (V2-Z/W existant, inchangé)

if dqn_cfg.eval_enabled:
    # Eval env séparé avec même proc_cfg que training
    eval_gen = _build_fresh_generator(proc_cfg)  # même type que training, instance fraîche
    eval_env = ProceduralGridWorld(cfg=proc_cfg, generator=eval_gen)
    self.evaluator = PeriodicEvaluator(
        eval_env=eval_env,
        eval_seeds=dqn_cfg.eval_seeds,
        max_steps=dqn_cfg.eval_max_steps,
        observation_encoder=encode_procedural_observation_2d,
        proc_cfg=proc_cfg,
    )
    self.best_tracker = BestCheckpointTracker(path=dqn_cfg.best_checkpoint_path)
else:
    self.evaluator = None
    self.best_tracker = None
```

### Boucle eval périodique dans `ConvProceduralDQNRunner.run()`

```python
for ep in range(self.dqn_cfg.episodes):
    # ... training step (V2-Z/W existant, inchangé)
    self.metrics.record_episode(ep_reward, ep_len, success=terminated)
    self.bucket_tracker.record_episode(...)
    self.callbacks.fire_episode(...)

    if (ep + 1) % self.sched_cfg.update_interval == 0:
        # ... scheduler update (V2-Z/W existant)

    # === NOUVEAU V2-V ===
    if (
        self.evaluator is not None
        and (ep + 1) % self.dqn_cfg.eval_every_episodes == 0
    ):
        eval_metrics = self.evaluator.evaluate(self.agent, self.scheduler.current)
        improved = self.best_tracker.update(eval_metrics, self.agent, episode=ep)
        self.callbacks.fire_eval(
            ep=ep,
            eval_winrate=eval_metrics["winrate"],
            eval_diff=eval_metrics["difficulty"],
            best_winrate=self.best_tracker.best_winrate,
            best_episode=self.best_tracker.best_episode,
            improved=improved,
        )
        self.callbacks.fire_log(
            "info",
            f"eval ep {ep:>4} : winrate={eval_metrics['winrate']:.2%} "
            f"@ diff={eval_metrics['difficulty']:.2f}  "
            f"best={self.best_tracker.best_winrate:.2%} @ ep {self.best_tracker.best_episode}"
            + ("  NEW BEST" if improved else "")
        )
```

### Trace d'un eval (`PeriodicEvaluator.evaluate(agent, difficulty)`)

```python
def evaluate(self, agent: ConvDQNAgent, difficulty: float) -> dict[str, float]:
    self.eval_env.set_difficulty(difficulty)
    n_success = 0
    total_reward = 0.0
    total_length = 0
    for seed in self.eval_seeds:
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
            action = agent.act(obs, greedy=True)  # greedy=True : pas de pollution rng
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
        "difficulty": difficulty,
    }
```

### Trace d'un best-checkpoint update

```python
def update(self, eval_metrics: dict, agent: ConvDQNAgent, episode: int) -> bool:
    wr = float(eval_metrics["winrate"])
    if wr > self._best_winrate:
        self._best_winrate = wr
        self._best_episode = episode
        self._best_difficulty = float(eval_metrics["difficulty"])
        if self._path is not None:
            agent.save(self._path)
        return True
    return False
```

### Overhead estimé

- Training ep : ~1 ms sur RTX 3060 (CNN forward + backward AMP)
- Eval ep : `max_steps * ~1 ms forward only` ≈ 200 ms / ep × 10 seeds = ~2 sec / eval
- Frequency : 1 eval / 100 ép training ≈ 100 ms / ép training
- **Overhead training time : ~10 %**

### Coût mémoire

- 1 ProceduralGridWorld eval instance (négligeable)
- 1 best-checkpoint .pt sur disque (~7 MB pour ConvQNetwork 1.66M params)
- Pas de buffer eval (pas de stockage de trajectoires)

---

## 4. Testing + DoD

### Phases TDD prévisionnelles

| Phase | Composant | Fichier de test | Tests | Cumul |
|---|---|---|---|---|
| 1 | Setup scaffold | — | 0 | 211 (baseline) |
| 2 | `PeriodicEvaluator` | `tests/training/test_evaluator.py` | 8 | 219 |
| 3 | `BestCheckpointTracker` | `tests/training/test_checkpoint_tracker.py` | 6 | 225 |
| 4 | `ConvDQNConfig` extension | `tests/test_conv_dqn_config.py` | 3 | 228 |
| 5 | `ConvProceduralDQNRunner` intégration | `tests/training/test_conv_procedural_runner.py` | 1 | 229 |
| 6 | CLI flags + CI smoke | (CI workflow) | 0 | 229 |
| 7 | README + CLAUDE.md + smoke E2E GPU + tag `v0.2.0-v` | — | 0 | 229 |

**Total ajouté : 18 tests** (211 → 229).

### Détail des tests critiques

**`test_evaluator.py`** (8 tests) :

1. `test_init_builds_eval_env` — env eval séparé, seeds eval distincts du training
2. `test_evaluate_returns_proper_metrics` — dict avec `winrate`, `mean_reward`, `mean_length`, `n_episodes`, `difficulty`
3. `test_evaluate_does_not_pollute_buffer` — **critique** : `len(agent.buffer)` inchangé avant/après eval
4. `test_evaluate_does_not_increment_global_step` — **critique** : `agent.global_step` inchangé
5. `test_evaluate_uses_greedy` — eval déterministe sur même obs avec ε haut (action identique sur 2 appels successifs)
6. `test_evaluate_runs_all_seeds` — vérifie que `n_episodes` retourné = `len(eval_seeds)`
7. `test_evaluate_winrate_bounds` — winrate dans [0, 1] (compatible invariant Aether I4)
8. `test_evaluate_respects_difficulty` — eval avec diff différentes → `eval_env._difficulty` set correctement

**`test_checkpoint_tracker.py`** (6 tests) :

1. `test_init_no_best` — `best_winrate == -inf`, `best_episode is None`
2. `test_first_update_always_saves` — premier eval triggers save (improvement vs `-inf`)
3. `test_lower_winrate_does_not_save` — `eval_winrate < best` → skip, file inchangé
4. `test_higher_winrate_saves_and_updates_best` — `eval_winrate > best` → save + update state
5. `test_path_none_no_save_attempted` — `path=None` → no IO, juste tracking en mémoire
6. `test_equal_winrate_does_not_save` — idempotence (égalité ne triggers pas save)

**`test_conv_dqn_config.py`** (+3) :

1. `test_eval_enabled_default_true` + `test_eval_every_episodes_default_100`
2. `test_eval_seeds_default_10000_range` (vérifie `tuple(range(10000, 10010))`)
3. `test_eval_every_episodes_validation` (`0` ou négatif → ValueError)

**`test_conv_procedural_runner.py`** (+1) :

1. `test_runner_eval_enabled_saves_best` — smoke 200 ép, `eval_every_episodes=50` (donc 4 évals), `best_checkpoint_path=tmp_path/"best.pt"`. Vérifier : (a) eval appelé ≥ 3 fois, (b) `tmp_path/"best.pt"` existe sur disque, (c) `runner.best_tracker.best_winrate >= 0`.

### Definition of Done

**DoD bloquante (livraison code)** :

1. ✅ `mw_ia/training/evaluator.py` : `PeriodicEvaluator` class complète
2. ✅ `mw_ia/training/checkpoint_tracker.py` : `BestCheckpointTracker` class complète
3. ✅ `mw_ia/config.py` : 5 nouveaux champs dans `ConvDQNConfig` + validation
4. ✅ `mw_ia/training/runner.py` : intégration dans `ConvProceduralDQNRunner`, `RunnerCallbacks.on_eval` ajouté
5. ✅ `scripts/train_cnn_dqn_procedural.py` : 3 nouveaux flags CLI
6. ✅ `pytest -q` → **229 passed** (211 baseline + 18 V2-V)
7. ✅ `bash aether/verify_all.sh` → 8 OK
8. ✅ Smoke E2E manuel GPU 500 ép avec eval activé : best-checkpoint .pt présent sur disque
9. ✅ Section V2-V dans README.md
10. ✅ Section V2-V dans CLAUDE.md (phases, composants, décisions, pièges)
11. ✅ Tag `v0.2.0-v` posé
12. ✅ Tags antérieurs intacts

**DoD non-bloquante (validation scientifique)** :

13. ⏳ Re-benchmark V2-W n=5 à ep=5000 AVEC eval+best-checkpoint
14. ⏳ Documenter dans CLAUDE.md : tableau comparatif "ep=5000 final" vs "ep=5000 best-checkpoint" par seed
15. ⏳ Cible succès : seed 4 best-checkpoint ≥ 60 % winrate (vs final 1 %)

### Pièges anticipés

| # | Piège | Mitigation |
|---|---|---|
| 1 | **Eval env partage le rng generator avec training env** | Construire un générateur fresh à l'init du `PeriodicEvaluator`. Documenter en commentaire. |
| 2 | **`agent.act(obs, greedy=True)` modifie le rng training** | Code V2-Z vérifié : `if (not greedy) and self._rng.random() < ...`. La condition skip le `self._rng.random()` en mode greedy → pas de side-effect. ✓ |
| 3 | **Save/load best clash format checkpoint V1** | Réutilise format V1 (online + target + global_step + cfg). Compatible `agent.load()`. |
| 4 | **`tmp_path` fixture pytest sur Windows : chemins avec espaces** | Utiliser `pathlib.Path` partout. PyTorch gère bien `Path`. |
| 5 | **CLI default `--eval` casse repro V2-Z/W "no eval"** | Documenter "Pour repro baseline V2-W sans eval, passer `--no-eval`". V2-V devrait montrer que best-checkpoint ≥ final → `--no-eval` devient option debug. |
| 6 | **Eval à la diff scheduler.current uniquement** | MVP. Future extension : eval multi-diff caractérisation curve (hors scope V2-V). |
| 7 | **Best-checkpoint écrasé entre runs** si même path | CLI default avec timestamp : `checkpoints/v2v_best_{timestamp}.pt`. Évite collision. |
| 8 | **`eval_seeds` valeurs élevées : risque chevauchement futur si `episodes >> 10000`** | Hors-scope MVP : 5000 ép par défaut, 10000-10009 jamais touchés. Si user passe `--episodes 50000`, edge case à documenter. |

### Pourquoi le sous-projet est plus petit qu'il en a l'air

L'infrastructure RL existe déjà :
- Agent avec `act(..., greedy=True)` et `save/load`
- Runner avec boucle épisode
- Config avec validation
- Env reproductible par seed
- Callbacks pour exposer métriques

V2-V se contente de **brancher** ces briques différemment. Pas de refonte. ~150-200 LOC pour un changement méthodologique structurant. ROI très élevé.

---

## 5. Annexe — récap des décisions clés

| Décision | Choix | Raison |
|---|---|---|
| Scope MVP | Évaluation périodique greedy + best-checkpoint (2 features) | YAGNI : early stopping, MA, rollback reportés |
| Runners ciblés | `ConvProceduralDQNRunner` uniquement | Là où H1 a été confirmée |
| Composant eval | `evaluator.py` séparé (réutilisable) | Architecture propre, extension V2-X/V2-Y facile |
| Méthode publique | `evaluate()` (pas `eval()`) | Évite collision builtin Python + hook security |
| Eval seeds | Seeds fixes 10000-10009 (hors training) | True held-out, reproductible, 10 = bruit réduit |
| Eval frequency | Toutes les 100 épisodes | 50 points eval / 5000 ép, overhead ~10 %, résolution suffisante |
| Eval difficulty | `scheduler.current` (suit le scheduler) | MVP — multi-diff = future extension |
| Eval env | Instance `ProceduralGridWorld` séparée | Zéro pollution training |
| Best-checkpoint trigger | `eval_winrate > best_winrate` strict | Idempotent |
| Format checkpoint | `agent.save(.pt)` format V1 | Compatible `agent.load()` |
| Default CLI | `--eval` activé | V2-V = amélioration recommandée |
| GUI | Pas de bouton dédié | Hérite via `ConvDQNConfig()` |
| Tag | `v0.2.0-v` posé même si re-benchmark non-atteint | Livraison code ≠ validation scientifique |

### Story scientifique post-V2-V (cible)

| Variante | Final winrate seed 4 | Best-checkpoint winrate seed 4 |
|---|---|---|
| V2-W ep=5000 sans eval | 1 % | n/a — final jeté |
| **V2-W ep=5000 + V2-V** | 1 % (final inchangé) | **~71 %** (best capturé vers ep 3000-3500) |

→ V2-V récupère le finding pratique du test ep=3000 **sans avoir besoin de connaître a priori le bon ep_max**.

### Sous-projets V3+ déblocables après V2-V

- Re-benchmark V2-Z et V2-W consolidés avec best-checkpoint (publishable-grade)
- V2-V étendu : early stopping + rollback + MA metrics
- V2-ZY CNN+LSTM+Double DQN (maintenant viable car best-checkpoint protège du collapse)
- Soft target Polyak / lr 1e-4 (tests propres car best-checkpoint isole l'effet vrai)
- Mazes larges (eval permet de tracker généralisation)
- Sous-projet B (mémoire persistante cross-session) — best-checkpoint est la fondation

---

**Fin de la spec V2-V.**
