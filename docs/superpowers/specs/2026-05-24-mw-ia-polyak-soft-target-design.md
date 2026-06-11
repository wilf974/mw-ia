# Spec V2-U — Polyak soft target update

> **Sous-projet** : V2-U (MW_IA — Reinforcement Learning éducatif)
> **Date** : 2026-05-24
> **Statut** : Spec validée — implémentation à dérouler via `superpowers:writing-plans` puis `superpowers:subagent-driven-development`
> **Tag livraison cible** : `v0.2.0-u`
> **Sous-projets prérequis livrés** : V1, V2-A, V2-X, V2-Y, V2-Z, V2-W, V2-V, V2-ZY

---

## 1. Vue d'ensemble & hypothèse

### Motivation empirique (V2-ZY benchmark n=5 consolidé 2026-05-24)

Le benchmark V2-ZY n=5 same-seed avec V2-V eval rigoureux a révélé un pattern inattendu :

| Variante | Mean best @ diff=0.30 | Std inter-seed | Best ≥ 70 % | Max best | Max diff atteinte |
|---|---|---|---|---|---|
| V2-W (CNN + Double DQN) | 58 % | ~12 pp | 2/5 | 70 % | 0.40 |
| **V2-ZY (CNN + LSTM + Double DQN)** | **42 %** | **~38 pp** | **1/5** (seed 4 = 100 %) | **100 %** ✓ | **0.55** ✓ |

**Découverte majeure seed 4 V2-ZY** : 100 % @ diff=0.30 (premier sous-projet à le faire) + diff finale 0.55 (premier à franchir bucket 1 vers bucket 2). **La capacité maximale du modèle existe.** Mais 4 autres seeds convergent mal (mean 42 %, std 38 pp).

**Lecture causale** : le bottleneck n'est plus la représentation (V2-Z débloque), ni la mémoire (V2-Y compense), ni la capacité théorique (seed 4 V2-ZY prouve qu'elle existe). C'est devenu **la dynamique d'entraînement**, et spécifiquement **les oscillations target network** induites par le hard sync tous les 1000 steps.

Pattern observé :
- Seed 4 V2-ZY = trajectoire chanceuse stable (monotone croissante ep 99 → 4899)
- Seeds 1/2 V2-ZY = divergence catastrophique (pas de progression sur 5000 ép)

Cohérent avec un problème de **stabilité target network** : Conv chaining + BPTT 32 + LSTM hidden state + hard target sync = system fragile aux discontinuités.

### Hypothèse V2-U (Polyak 1964, ré-adopté DQN par Lillicrap 2015 DDPG)

> Remplacer le hard sync target `target ← online` (discontinu, tous les N steps) par un soft Polyak update `target ← τ × online + (1−τ) × target` (continu, à chaque train_step) avec `τ ≈ 0.005`. Devrait réduire les oscillations target Q et stabiliser l'apprentissage Conv+LSTM+BPTT.

### Question scientifique

> Polyak soft target réduit-il la variance inter-seed de V2-ZY (std 38 pp → < 20 pp) **sans réduire la capacité maximale** (seed 4 = 100 % doit rester accessible) ?

### Critère succès chiffré

- **Critère primaire** : `std` inter-seed best @ diff=0.30 sur V2-ZY+Polyak n=5 **< 20 pp** (vs 38 pp baseline V2-ZY)
- **Pas de cible sur le mean** : le user a explicitement précisé "c'est variance qui compte, pas max best score"

### Pattern d'intégration

Un seul champ `polyak_tau: float = 0.0` ajouté aux 3 configs DQN existantes (`ConvDQNConfig` V2-Z/W, `DRQNConfig` V2-Y, `ConvRecurrentDQNConfig` V2-ZY) + branche conditionnelle dans les 2 trainers existants (`_ConvDQNTrainer` + `RecurrentDQNTrainer`). Default `polyak_tau = 0.0` = hard sync (préserve baselines V2-W/V2-Y/V2-ZY n=5 reproductibles).

**Activation V2-U** : `--polyak-tau 0.005` opt-in via CLI. Strict A/B comparison V2-ZY-hard vs V2-ZY-polyak sur mêmes seeds.

**Tag livraison cible** : `v0.2.0-u` posé que le critère soit atteint ou non (pattern V2-V/V2-ZY).

---

## 2. Architecture & composants

### Formule Polyak

À chaque `train_step()`, **après** le forward/backward/optimizer :

```
target_param ← τ × online_param + (1 − τ) × target_param  pour chaque paramètre
```

- `τ = 0.0` → no-op (target_param inchangé). Fallback hard sync géré par l'agent qui appelle `sync_target()` tous les `target_sync_steps` steps comme avant V2-U.
- `τ = 1.0` → target_param ← online_param (équivalent hard sync à chaque train_step, trop bruité, autorisé pour tests).
- `τ = 0.005` → 99.5 % de l'ancien target + 0.5 % du online à chaque step (smoothing exponentiel temporel constant ~200 train_steps).

Implémentation PyTorch idiomatique (`torch.no_grad()` + in-place `mul_().add_()`) :

```python
def polyak_update(self, tau: float) -> None:
    """Soft update target ← τ × online + (1−τ) × target, in-place."""
    with torch.no_grad():
        for p_target, p_online in zip(self.target.parameters(), self.online.parameters()):
            p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)
```

### Logique conditionnelle dans `trainer.step()`

Pattern dans les 2 trainers (`_ConvDQNTrainer` et `RecurrentDQNTrainer`), juste après `optimizer.step()` / `scaler.step()` :

```python
# V2-U : soft update à chaque train_step si tau > 0
if self.polyak_tau > 0.0:
    self.polyak_update(self.polyak_tau)
```

### Skip hard sync périodique dans l'agent

Si Polyak activé, l'agent ne doit PAS appeler `sync_target()` périodiquement (sinon double-update) :

```python
# Dans Agent.observe() ou end_episode() :
if self.cfg.polyak_tau == 0.0:
    if self.global_step % self.cfg.target_sync_steps == 0:
        self.trainer.sync_target()  # hard sync V2-Y/Z/W baseline
        self.target_syncs += 1
# Si polyak_tau > 0, le trainer.step() fait déjà l'update Polyak → skip sync périodique
```

### Fichiers modifiés

| Fichier | Modif |
|---|---|
| `mw_ia/config.py` | + champ `polyak_tau: float = 0.0` dans `ConvDQNConfig` (V2-Z/W), `DRQNConfig` (V2-Y), `ConvRecurrentDQNConfig` (V2-ZY). Validation `__post_init__` : `0.0 <= polyak_tau <= 1.0`. |
| `mw_ia/agents/conv_dqn.py` | + paramètre `polyak_tau: float = 0.0` dans `_ConvDQNTrainer.__init__`. + méthode `polyak_update(tau)`. + branche conditionnelle dans `step()` : si `tau > 0`, appel `polyak_update(tau)` post-optimizer. + dans `ConvDQNAgent.observe()` : si `cfg.polyak_tau > 0`, skip `target_sync_steps` periodic hard sync. + `ConvDQNAgent.__init__` passe `cfg.polyak_tau` au trainer. |
| `mw_ia/neural/recurrent_trainer.py` | Idem : + kwarg `polyak_tau` dans `__init__`, + méthode `polyak_update(tau)`, + branche conditionnelle dans `step()`. |
| `mw_ia/agents/recurrent_dqn.py` | `RecurrentDQNAgent` (V2-Y) : si `cfg.polyak_tau > 0`, skip hard sync périodique dans `end_episode()`. Passer `cfg.polyak_tau` au trainer. |
| `mw_ia/agents/conv_recurrent_dqn.py` | `ConvRecurrentDQNAgent` (V2-ZY) : idem V2-Y agent. |
| `scripts/train_cnn_dqn_procedural.py` | + flag CLI `--polyak-tau` (default 0.0) → `ConvDQNConfig.polyak_tau`. |
| `scripts/train_drqn_procedural.py` | + flag CLI `--polyak-tau` (default 0.0) → `DRQNConfig.polyak_tau`. |
| `scripts/train_cnn_lstm_dqn_procedural.py` | + flag CLI `--polyak-tau` (default 0.0) → `ConvRecurrentDQNConfig.polyak_tau`. |

### Nouveaux fichiers tests

| Fichier | Rôle |
|---|---|
| `tests/neural/test_polyak_update.py` | 5 tests unitaires de la formule Polyak (no-op tau=0, exact copy tau=1, smoothing intermédiaire, idempotence, no-modify-online). |

### Tests étendus

| Fichier | Tests ajoutés |
|---|---|
| `tests/agents/test_conv_dqn.py` | +2 tests (Polyak skip hard sync, hard sync conservé si tau=0) |
| `tests/neural/test_recurrent_trainer.py` | +1 test (Polyak update target dans step()) |
| `tests/agents/test_recurrent_dqn.py` | +1 test (V2-Y agent skip hard sync si tau>0) |
| `tests/agents/test_conv_recurrent_dqn.py` | +1 test (V2-ZY agent skip hard sync si tau>0) |
| `tests/test_conv_dqn_config.py` | +1 test (default 0.0 + validation [0, 1]) |
| `tests/test_drqn_config.py` (V2-Y existant) | +1 test similaire |
| `tests/test_conv_recurrent_dqn_config.py` | +1 test similaire |

### Décisions d'API

**`polyak_update(self, tau: float) -> None`** : méthode publique du trainer, modifie `self.target.parameters()` in-place. Pas de retour, pas d'effet sur online net ni optimizer.

**Validation `0.0 <= polyak_tau <= 1.0`** : `ValueError` si hors-bornes.

**Compat backwards V2-W/V2-Y/V2-ZY baselines** : default `polyak_tau = 0.0` dans toutes les configs. Tous les tests existants passent inchangés. Strict opt-in via CLI ou config explicite.

---

## 3. Data flow détaillé + edge cases

### Trace `_ConvDQNTrainer.__init__` (extension V2-W)

```python
def __init__(
    self,
    online: ConvQNetwork,
    target: ConvQNetwork,
    *,
    in_channels: int,
    rows: int,
    cols: int,
    lr: float = 1e-3,
    gamma: float = 0.99,
    device: str = "cuda",
    use_amp: bool = True,
    double_dqn: bool = True,           # V2-W déjà existant
    polyak_tau: float = 0.0,           # V2-U nouveau
) -> None:
    # ... unchanged init ...
    self.double_dqn = double_dqn
    self.polyak_tau = polyak_tau       # V2-U
    self.sync_target()                  # hard sync initial (online == target au step 0)
```

### Trace `_ConvDQNTrainer.step()` post V2-U

```python
def step(self, batch: Batch) -> float:
    # ... unchanged forward + loss + backward ...

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

    # V2-U : soft Polyak update à chaque train_step si tau > 0
    if self.polyak_tau > 0.0:
        self.polyak_update(self.polyak_tau)

    return float(loss.detach().item())
```

### Trace `ConvDQNAgent.observe()` (skip conditionnel)

```python
def observe(self, state, action, reward, next_state, done) -> dict[str, float]:
    self.buffer.push(state.flatten(), action, reward, next_state.flatten(), done)
    self.global_step += 1
    metrics: dict[str, float] = {"epsilon": self.epsilon}
    train_threshold = max(self.cfg.min_replay_to_learn, self.cfg.batch_size)
    if (
        len(self.buffer) >= train_threshold
        and self.global_step % self.cfg.train_every == 0
    ):
        batch = self.buffer.sample(self.cfg.batch_size)
        self.last_loss = self.trainer.step(batch)  # Polyak update appliqué interne si tau>0
        metrics["loss"] = self.last_loss
    # V2-U : skip hard sync périodique si Polyak activé
    if self.cfg.polyak_tau == 0.0:
        if self.global_step % self.cfg.target_sync_steps == 0:
            self.trainer.sync_target()
            self.target_syncs += 1
    return metrics
```

### Trace `RecurrentDQNAgent.end_episode()` (V2-Y) et `ConvRecurrentDQNAgent.end_episode()` (V2-ZY)

Même pattern : skip `target_sync_steps` periodic hard sync si `cfg.polyak_tau > 0`. Polyak update déjà appliqué dans `trainer.step()` per batch BPTT (à chaque train_steps_per_episode batch).

### Edge cases

**1. `polyak_tau = 1.0`** : équivalent hard sync à chaque train_step. Comportement valide mais dégénéré (target devient online sans smoothing). Permis par validation `0.0 <= tau <= 1.0`. Utile pour tests unitaires.

**2. `polyak_tau = 0.0` (default)** : `polyak_update()` skip. `sync_target()` périodique reste actif via l'agent. **Exactement le comportement V2-Y/V2-Z/V2-W/V2-ZY actuel.** Zéro changement comportemental pour les configs existantes.

**3. AMP + Polyak** : `polyak_update()` est dans `torch.no_grad()` mais PAS sous `torch.amp.autocast()`. Les buffers de poids sont en `float32` (Adam optimizer state), donc l'update se fait en `float32`. Pas de cast issue.

**4. Init au step 0** : `sync_target()` est appelé dans `__init__` du trainer (online et target identiques). Polyak update au premier train_step a un effet nul si poids identiques (target = (1-τ)·target + τ·online = target). Le décrochage apparaît dès le premier gradient step sur online.

**5. Save/load checkpoint** : `agent.save()` sauvegarde `online + target + global_step + cfg.__dict__`. `cfg.__dict__` inclut `polyak_tau` automatiquement (frozen dataclass). Pas de modification save/load nécessaire.

**6. Buffers de modules (BN/LN running stats)** : `parameters()` n'itère pas sur les buffers. Réseaux actuels (Conv2d, ReLU, Linear, LSTM) n'ont pas de running stats persistants → safe. Si BatchNorm ajouté plus tard, switcher à `state_dict()` iteration (note dans pièges).

### Overhead estimé

- `polyak_update()` à chaque train_step : `O(num_params)` ops in-place
- ConvDQN 1.66M params : ~0.5 ms / call GPU
- ConvRecurrentDQN 3.3M params : ~1 ms / call GPU
- Train step actuel ~10-20 ms GPU → +5-10 % overhead
- Hard sync supprimé (économise `state_dict() + load_state_dict()` ~10 ms tous les 1000 steps) → bénéfice net léger

### Memory footprint

- Aucun buffer additionnel
- `with torch.no_grad()` empêche accumulation gradients
- Ops in-place (`mul_`, `add_`) → pas de tenseurs intermédiaires

---

## 4. Testing strategy + DoD

### Phases TDD prévisionnelles

| Phase | Composant | Fichier de test | Tests | Cumul |
|---|---|---|---|---|
| 1 | Setup scaffold (test_polyak_update.py vide) | — | 0 | 252 |
| 2 | `polyak_update()` méthode pure | `tests/neural/test_polyak_update.py` | 5 | 257 |
| 3 | `_ConvDQNTrainer` extension + branche conditionnelle | `tests/agents/test_conv_dqn.py` | +1 | 258 |
| 4 | `RecurrentDQNTrainer` extension + branche conditionnelle | `tests/neural/test_recurrent_trainer.py` | +1 | 259 |
| 5 | `ConvDQNAgent` skip hard sync si tau>0 | `tests/agents/test_conv_dqn.py` | +1 | 260 |
| 6 | `RecurrentDQNAgent` + `ConvRecurrentDQNAgent` skip hard sync | `tests/agents/test_recurrent_dqn.py`, `tests/agents/test_conv_recurrent_dqn.py` | +2 | 262 |
| 7 | Config extensions (3 configs) | `tests/test_conv_dqn_config.py`, `tests/test_drqn_config.py`, `tests/test_conv_recurrent_dqn_config.py` | +3 | 265 |
| 8 | CLI flags `--polyak-tau` (3 scripts) | (smoke manuel) | 0 | 265 |
| 9 | CI smoke V2-U avec `--polyak-tau 0.005` | — | 0 | 265 |
| 10 | README + CLAUDE.md V2-U + smoke E2E GPU + tag `v0.2.0-u` | — | 0 | 265 |

**Total ajouté : 13 tests** (252 → 265).

### Détail des tests critiques

**`test_polyak_update.py`** (5 tests) :

1. `test_polyak_tau_zero_is_noop` — `polyak_update(0.0)` ne modifie pas target (`torch.allclose` avant/après)
2. `test_polyak_tau_one_copies_online_to_target` — `polyak_update(1.0)` rend target identique à online
3. `test_polyak_intermediate_tau` — `polyak_update(0.5)` produit target = 0.5 × old_target + 0.5 × online (paramètres connus)
4. `test_polyak_idempotent_when_online_equals_target` — si online == target, polyak_update(τ) ne change rien quel que soit τ
5. `test_polyak_does_not_modify_online` — `polyak_update` ne touche qu'à target, online inchangé

**`test_conv_dqn.py`** (+2 tests) :

6. `test_polyak_tau_skips_hard_sync` — avec `polyak_tau=0.005, target_sync_steps=2`, après 5 `observe()`, `agent.target_syncs == 0`
7. `test_polyak_tau_zero_preserves_hard_sync` — avec `polyak_tau=0.0, target_sync_steps=2`, hard sync s'active comme baseline V2-W (`target_syncs > 0`)

**`test_recurrent_trainer.py`** (+1 test) :

8. `test_polyak_update_changes_target_in_step` — instancier trainer avec `polyak_tau=0.5`, faire 1 `step()` avec batch random, vérifier target params != initial target params

**`test_recurrent_dqn.py`** (+1 test) :

9. Pattern test 6 pour V2-Y : `polyak_tau > 0` → `target_syncs` reste à 0 après plusieurs `end_episode()`.

**`test_conv_recurrent_dqn.py`** (+1 test) :

10. Pattern test 6 pour V2-ZY.

**Config tests** (+3 tests) :

11-13. Pour chaque config : `test_polyak_tau_default_and_validation` — default `0.0`, validation `[0, 1]`, hors-bornes → `ValueError`.

### Definition of Done

**DoD bloquante (livraison code)** :

1. ✅ `mw_ia/config.py` : champ `polyak_tau: float = 0.0` ajouté à `ConvDQNConfig`, `DRQNConfig`, `ConvRecurrentDQNConfig` + validation
2. ✅ `mw_ia/agents/conv_dqn.py` : `_ConvDQNTrainer` extension + `ConvDQNAgent.observe()` skip conditionnel
3. ✅ `mw_ia/neural/recurrent_trainer.py` : `RecurrentDQNTrainer` extension
4. ✅ `mw_ia/agents/recurrent_dqn.py` : `RecurrentDQNAgent.end_episode()` skip conditionnel
5. ✅ `mw_ia/agents/conv_recurrent_dqn.py` : `ConvRecurrentDQNAgent.end_episode()` skip conditionnel + propagation `cfg.polyak_tau`
6. ✅ 3 scripts CLI étendus avec `--polyak-tau` flag
7. ✅ `pytest -q` → **265 passed** (252 baseline + 13 V2-U)
8. ✅ `bash aether/verify_all.sh` → 8 OK
9. ✅ Smoke E2E manuel GPU 500 ép avec `--polyak-tau 0.005` sur V2-ZY : pas de crash, eval logs présents, best-checkpoint .pt créé
10. ✅ Smoke E2E manuel GPU 500 ép avec `--polyak-tau 0.0` (regression check) : reproductibilité V2-ZY baseline
11. ✅ Section V2-U dans README.md
12. ✅ Section V2-U dans CLAUDE.md
13. ✅ Tag `v0.2.0-u` posé
14. ✅ Tags antérieurs intacts
15. ✅ V2-Y/V2-W/V2-ZY baselines restent verts avec `polyak_tau=0.0` default

**DoD non-bloquante (validation scientifique)** :

16. ⏳ Benchmark same-seed n=5 V2-ZY+Polyak vs V2-ZY-hard à ep=5000 GPU
17. ⏳ Section "V2-U — validation empirique" dans CLAUDE.md : tableau comparatif n=5
18. ⏳ **Critère succès** : **std inter-seed best @ diff=0.30 < 20 pp** (vs 38 pp baseline V2-ZY)
19. ⏳ Si critère atteint → re-benchmark V2-W+Polyak (consolidation transverse)
20. ⏳ Si critère non atteint → documenter honnêtement, candidats : R2D2 burn-in, grid search τ

### Pièges anticipés

| # | Piège | Mitigation |
|---|---|---|
| 1 | **Double-update target** si Polyak activé ET hard sync périodique pas skip | Logique skip dans agent (`if cfg.polyak_tau == 0.0: hard_sync`). Tests 6/9/10 vérifient. |
| 2 | **Polyak n'inclut pas les buffers BN/LN** | Réseaux actuels n'ont pas de BN/LN avec running stats. `parameters()` itère sur Conv/Linear/LSTM weights+bias, suffit. À noter pour évolution future (R2D2 LayerNorm). |
| 3 | **AMP + Polyak** : `polyak_update` doit être en `torch.no_grad()` mais PAS sous autocast | Pattern : `with torch.no_grad():` seul. Op in-place sur float32 storage. |
| 4 | **Init step 0** : Polyak no-op si poids identiques | Comportement attendu. Premier vrai update Polyak après le 1er gradient step. |
| 5 | **save/load** : `cfg.__dict__` inclut `polyak_tau` automatiquement | Pas de modif save/load. |
| 6 | **τ = 0.005 trop conservateur ou trop agressif ?** | Default littéraire (Lillicrap 2015 DDPG, OpenAI baselines). Si V2-U n'atteint pas le critère, grid search τ ∈ {0.001, 0.01, 0.05} en suite. |
| 7 | **Repro V2-ZY n=5 baseline** | `polyak_tau=0.005` doit être la SEULE variable changée vs benchmark V2-ZY n=5 (mean 42 %, std 38 pp, seed 4 = 100 %). |
| 8 | **`DRQNConfig` peut ne pas avoir `polyak_tau`** initialement | Ajouter `polyak_tau: float = 0.0` + validation. Préserve les 35 tests V2-Y existants. |

---

## 5. Annexe — récap des décisions clés

| Décision | Choix | Raison |
|---|---|---|
| Scope trainers | `_ConvDQNTrainer` (V2-Z/W) + `RecurrentDQNTrainer` (V2-Y/ZY) | Permet validation transverse. Pas de refactor abstract trainer (yagni). |
| API | Champ `polyak_tau: float = 0.0` dans config | `0.0` = hard sync (backwards compat), `> 0` = soft Polyak. |
| Default | `polyak_tau = 0.0` partout | Préserve repro V2-W/V2-Y/V2-ZY baselines. Strict opt-in via CLI. |
| Formule Polyak | `target ← τ × online + (1−τ) × target` (in-place, no_grad) | Standard Lillicrap 2015 DDPG. |
| Activation per train_step | Dans `trainer.step()` post-optimizer | Mise à jour continue à chaque batch. |
| Hard sync conditionnel | Skip `target_sync_steps` periodic hard sync si `polyak_tau > 0` | Évite double-update. |
| Critère succès | **Std inter-seed best @ diff=0.30 < 20 pp** | Le user a explicitement insisté : "c'est variance qui compte, pas max best score". |
| τ par défaut recommandé CLI | `0.005` | Standard DDPG/SAC. Smoothing constant ~200 train_steps. |
| Tag | `v0.2.0-u` posé même si critère non atteint | Livraison code ≠ validation scientifique. |

### Story scientifique cible (post-V2-U)

**Si critère atteint (std < 20 pp sur V2-ZY+Polyak)** :

| Variante | Mean | Std | Best ≥ 70 % | Max best | Max diff |
|---|---|---|---|---|---|
| V2-W (CNN + DDQN) | 58 % | ~12 pp | 2/5 | 70 % | 0.40 |
| V2-ZY (CNN + LSTM + DDQN) | 42 % | ~38 pp | 1/5 | 100 % ✓ | 0.55 ✓ |
| **V2-ZY + Polyak (cible)** | **≥ 50 %** | **< 20 pp** ✓ | **≥ 3/5** ? | **maintenu ≥ 80 %** ? | **maintenu ≥ 0.40** |

→ Story : "**Le bottleneck final de V2-ZY n'était pas la capacité (déjà existante seed 4) mais la stabilité target network. Polyak soft update résout 50 % de la variance d'apprentissage, transformant V2-ZY de 'occasionnellement excellent' à 'robustement compétitif'.**"

**Si critère non atteint** :
- Documenter honnêtement (pattern V2-ZY n=5)
- Candidats restants : R2D2 burn-in, learning rate scheduler, gradient clipping plus serré
- V2-U reste livré (code complet, tag posé)

### Sous-projets V3+ déblocables après V2-U

**Si V2-U réussit** :
- **Re-benchmark V2-W+Polyak** (consolidation transverse, vérifier que V2-W ne dégrade pas)
- **V2-ZY+Polyak benchmark @ diff=0.40** : caractériser le nouveau plafond
- **R2D2 burn-in** : stabiliser encore plus le LSTM
- **Mazes larges** : test sur grilles 15/20

**Si V2-U échoue** :
- **R2D2 burn-in** prioritaire (cible la stabilité LSTM directement)
- **Grid search τ** avant d'abandonner Polyak

### Pourquoi V2-U est le bon prochain levier

L'infrastructure existe entièrement :
- 2 trainers identifiés, points d'extension propres
- V2-V eval rigoureux permet de mesurer variance précisément
- Pattern flag binaire backwards-compat (V2-W `double_dqn`, V2-V `eval_enabled`) éprouvé 3 fois

V2-U = **changement algorithmique ciblé (~80-120 LOC) avec hypothèse précise et critère testable**. ROI scientifique élevé.

### Pourquoi le critère "std seulement" est intelligent

- **Mean** = score moyen = facile à améliorer par hyperparams (lr, epsilon, etc.) — pas un finding architectural
- **Std** = robustesse d'apprentissage = vrai indicateur de stabilité algorithmique — un finding causal
- Si V2-U réduit la variance sans dégrader le mean, c'est un **pur gain de robustesse** = vraie contribution scientifique

C'est la bonne méthodologie : pas "maximiser le score", mais "stabiliser l'apprentissage". RL mature.

---

**Fin de la spec V2-U.**
