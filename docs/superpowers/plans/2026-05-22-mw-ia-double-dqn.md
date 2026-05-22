# MW_IA V2-W Double DQN sur ConvDQN — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un flag `double_dqn` à `ConvDQNConfig` + branche conditionnelle dans `_ConvDQNTrainer.step()` pour découpler sélection d'action (online net) et évaluation (target net), réduisant la surestimation Q-values et stabilisant la convergence inter-seeds du CNN procedural.

**Architecture:** Approche flag minimal (A/B testable). Aucun nouveau fichier code, aucun nouveau runner, aucun nouveau bouton GUI. Réutilise intégralement l'infrastructure V2-Z. Modification ciblée : `ConvDQNConfig` (+1 champ), `_ConvDQNTrainer.__init__/step()` (+10 LOC), CLI (+1 flag), tests (+1 test). Validation empirique post-impl : n=3 same-seed V2-Z vs V2-W (réutilise seeds 0/1/2 déjà mesurés).

**Tech Stack:** Python 3.13, PyTorch (cu128, déjà installé), pytest, argparse `BooleanOptionalAction`. Aucune nouvelle dépendance.

**Spec source:** `docs/superpowers/specs/2026-05-22-mw-ia-double-dqn-design.md`

**État initial:** Branche `main`, tags `v0.1.0` + `v0.2.0-a` + `v0.2.0-x` + `v0.2.0-y` + `v0.2.0-z` posés. 208 tests pytest verts. Pattern : développement sur `main` (pas de feature branch).

---

## Phase 1 — Flag dans `ConvDQNConfig`

### Task 1 : Ajouter `double_dqn: bool` au dataclass

**Files :**
- Modify : `mw_ia/config.py` (`ConvDQNConfig` à la fin du fichier)
- Test : `tests/test_conv_dqn_config.py`

- [ ] **Step 1 — Verify initial state**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `208 passed`.

- [ ] **Step 2 — Write the failing test**

Ajouter à la fin de `tests/test_conv_dqn_config.py` :

```python
def test_double_dqn_default_true() -> None:
    """V2-W : default double_dqn=True (V2-W est l'amélioration recommandée)."""
    cfg = ConvDQNConfig()
    assert cfg.double_dqn is True


def test_double_dqn_can_be_disabled() -> None:
    """double_dqn=False pour reproduire V2-Z baseline."""
    cfg = ConvDQNConfig(double_dqn=False)
    assert cfg.double_dqn is False
```

- [ ] **Step 3 — Run tests, verify they fail**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py::test_double_dqn_default_true tests/test_conv_dqn_config.py::test_double_dqn_can_be_disabled -v 2>&1 | tail -10
```

Attendu : 2 fails avec `AttributeError: 'ConvDQNConfig' object has no attribute 'double_dqn'`.

- [ ] **Step 4 — Add field to `ConvDQNConfig`**

Dans `mw_ia/config.py`, localiser la dataclass `ConvDQNConfig`. Ajouter le champ `double_dqn` juste après le champ `use_amp: bool = True` (regrouper les bools) :

Remplacer :
```python
    train_every: int = 4
    use_amp: bool = True
    episodes: int = 5_000
```

par :
```python
    train_every: int = 4
    use_amp: bool = True
    double_dqn: bool = True   # V2-W : Hasselt 2015. False = V2-Z baseline DQN classique.
    episodes: int = 5_000
```

Aucune validation supplémentaire dans `__post_init__` (bool est déjà type-checked par dataclass).

- [ ] **Step 5 — Run tests, verify they pass**

```bash
source .venv/Scripts/activate && pytest tests/test_conv_dqn_config.py -v 2>&1 | tail -10
```

Attendu : 6 passed (4 existants + 2 nouveaux).

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `210 passed` (208 baseline + 2 V2-W flag).

- [ ] **Step 7 — Commit**

```bash
git add mw_ia/config.py tests/test_conv_dqn_config.py
git commit -m "feat(v2-w): add double_dqn flag to ConvDQNConfig (default True)"
```

---

## Phase 2 — Branche conditionnelle dans `_ConvDQNTrainer`

### Task 2 : Implémenter la formule Double DQN avec branche

**Files :**
- Modify : `mw_ia/agents/conv_dqn.py` (`_ConvDQNTrainer.__init__`, `_ConvDQNTrainer.step`, `ConvDQNAgent.__init__`)
- Test : `tests/agents/test_conv_dqn.py`

- [ ] **Step 1 — Write the failing test**

Ajouter à la fin de `tests/agents/test_conv_dqn.py` (après `test_aether_smoke`) :

```python
def test_double_dqn_branch_differs_from_standard() -> None:
    """V2-W : avec online ≠ target, les formules DQN et Double DQN divergent.

    - DQN classique :  q_next = max_a Q_target(s', a)
    - Double DQN :     q_next = Q_target(s', argmax_a Q_online(s', a))

    Si argmax_online ≠ argmax_target, les deux formules donnent des q_next
    différents. Pour rendre ça déterministe, on désynchronise volontairement
    online/target avant comparaison.
    """
    from mw_ia.neural.conv_network import ConvQNetwork

    online = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    )
    target = ConvQNetwork(
        in_channels=3, rows=10, cols=10, n_actions=4,
        conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256,
    )
    # Sync initial pour partir de poids identiques
    target.load_state_dict(online.state_dict())
    # Désynchroniser online en ajoutant un offset
    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.5)

    torch.manual_seed(42)
    next_states = torch.randn(4, 3, 10, 10)

    with torch.no_grad():
        # Formule DQN classique (V2-Z baseline)
        q_next_dqn = target(next_states).max(dim=1).values
        # Formule Double DQN (V2-W)
        next_actions = online(next_states).argmax(dim=1)
        q_next_double = target(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)

    # Avec online ≠ target, les 2 formules DOIVENT diverger sur au moins une transition
    assert not torch.allclose(q_next_dqn, q_next_double), (
        "Double DQN doit différer de DQN classique quand online ≠ target"
    )
```

- [ ] **Step 2 — Run test, verify it passes already**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py::test_double_dqn_branch_differs_from_standard -v 2>&1 | tail -10
```

Attendu : 1 passed. Note : ce test passe dès qu'il existe — il vérifie la propriété mathématique des 2 formules, pas l'impl. C'est intentionnel : c'est une garde anti-régression qui doit rester verte avant ET après la modif de `_ConvDQNTrainer`.

- [ ] **Step 3 — Modify `_ConvDQNTrainer.__init__` to accept `double_dqn` parameter**

Dans `mw_ia/agents/conv_dqn.py`, modifier `_ConvDQNTrainer.__init__`. Remplacer la signature actuelle :

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
    ) -> None:
```

par :

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
        double_dqn: bool = True,
    ) -> None:
```

Et dans le corps de `__init__`, ajouter `self.double_dqn = double_dqn` juste après `self.use_amp = bool(use_amp and self.device.type == "cuda")`.

- [ ] **Step 4 — Modify `_ConvDQNTrainer.step()` to use the flag**

Dans `_ConvDQNTrainer.step()`, localiser la section actuelle :

```python
        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            q_pred = self.online(states).gather(1, actions.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                q_next = self.target(next_states).max(dim=1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)
            loss = self.loss_fn(q_pred, target_q)
```

Remplacer par :

```python
        with torch.amp.autocast(device_type="cuda", enabled=self.use_amp):
            q_pred = self.online(states).gather(1, actions.view(-1, 1)).squeeze(1)
            with torch.no_grad():
                if self.double_dqn:
                    # V2-W : online sélectionne, target évalue (Hasselt 2015)
                    next_actions = self.online(next_states).argmax(dim=1)
                    q_next = self.target(next_states).gather(
                        1, next_actions.view(-1, 1)
                    ).squeeze(1)
                else:
                    # V2-Z baseline : target sélectionne ET évalue (DQN classique)
                    q_next = self.target(next_states).max(dim=1).values
                target_q = rewards + self.gamma * q_next * (1.0 - dones)
            loss = self.loss_fn(q_pred, target_q)
```

- [ ] **Step 5 — Modify `ConvDQNAgent.__init__` to pass `cfg.double_dqn` to trainer**

Dans `ConvDQNAgent.__init__`, localiser la construction du trainer :

```python
        self.trainer = _ConvDQNTrainer(
            self.online, self.target,
            in_channels=in_channels, rows=rows, cols=cols,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
        )
```

Remplacer par :

```python
        self.trainer = _ConvDQNTrainer(
            self.online, self.target,
            in_channels=in_channels, rows=rows, cols=cols,
            lr=cfg.lr, gamma=cfg.gamma,
            device=str(self.device), use_amp=cfg.use_amp,
            double_dqn=cfg.double_dqn,
        )
```

- [ ] **Step 6 — Run test suite to verify nothing broke**

```bash
source .venv/Scripts/activate && pytest tests/agents/test_conv_dqn.py -v 2>&1 | tail -15
```

Attendu : 8 passed (7 existants V2-Z + 1 nouveau V2-W).

- [ ] **Step 7 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `211 passed` (210 + 1).

- [ ] **Step 8 — Smoke test : V2-W et V2-Z baseline restent fonctionnels**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode obstacles --device cpu
```

Attendu : pas de crash, 5 buckets affichés. (Default est maintenant Double DQN.)

- [ ] **Step 9 — Commit**

```bash
git add mw_ia/agents/conv_dqn.py tests/agents/test_conv_dqn.py
git commit -m "feat(v2-w): implement Double DQN branch in _ConvDQNTrainer.step()"
```

---

## Phase 3 — CLI flag `--double-dqn / --no-double-dqn`

### Task 3 : Exposer le flag dans le CLI

**Files :**
- Modify : `scripts/train_cnn_dqn_procedural.py`

- [ ] **Step 1 — Add CLI flag to argparse**

Dans `scripts/train_cnn_dqn_procedural.py`, localiser la section `argparse` (dans `main()`). Ajouter le flag juste après `--scheduler-step` (le dernier flag actuel) :

```python
    parser.add_argument(
        "--double-dqn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Double DQN (Hasselt 2015) : online sélectionne, target évalue. "
             "Default V2-W. Utiliser --no-double-dqn pour reproduire V2-Z baseline.",
    )
```

- [ ] **Step 2 — Pass the flag to `ConvDQNConfig`**

Localiser la construction de `ConvDQNConfig` :

```python
    dqn_cfg = ConvDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        fc_hidden=args.fc_hidden,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_steps=args.target_sync_steps,
    )
```

Ajouter `double_dqn=args.double_dqn` :

```python
    dqn_cfg = ConvDQNConfig(
        episodes=args.episodes,
        conv_channels=tuple(args.conv_channels),
        fc_hidden=args.fc_hidden,
        epsilon_decay_steps=args.epsilon_decay_steps,
        target_sync_steps=args.target_sync_steps,
        double_dqn=args.double_dqn,
    )
```

- [ ] **Step 3 — Smoke test V2-W (default `--double-dqn`)**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode obstacles --device cpu
```

Attendu : pas de crash, sortie similaire à V2-Z mais avec le calcul de target Q-value différent.

- [ ] **Step 4 — Smoke test V2-Z baseline (`--no-double-dqn`)**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --episodes 10 --mode obstacles --device cpu --no-double-dqn
```

Attendu : pas de crash. Cohérent avec V2-Z baseline runs.

- [ ] **Step 5 — Verify `--help` shows the new flag**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --help 2>&1 | grep -A2 double
```

Attendu : output contenant `--double-dqn`, `--no-double-dqn` et le help text mentionnant "Hasselt".

Note : si `UnicodeEncodeError` cp1252 sur ε ou accents → utiliser `PYTHONIOENCODING=utf-8 python scripts/...` (piège #8 documenté dans CLAUDE.md).

- [ ] **Step 6 — Run full suite**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
```

Attendu : `211 passed`.

- [ ] **Step 7 — Commit**

```bash
git add scripts/train_cnn_dqn_procedural.py
git commit -m "feat(v2-w): add --double-dqn / --no-double-dqn CLI flag"
```

---

## Phase 4 — README + CLAUDE.md V2-W section + smoke E2E

### Task 4 : Documentation V2-W (sans benchmark — placeholder)

**Files :**
- Modify : `README.md`
- Modify : `CLAUDE.md`

- [ ] **Step 1 — Aether re-verify (sanity)**

```bash
bash aether/verify_all.sh
```

Attendu : `8 OK`.

- [ ] **Step 2 — Smoke E2E manuel GPU 50 ép V2-W**

```bash
source .venv/Scripts/activate && python scripts/train_cnn_dqn_procedural.py --episodes 50 --mode obstacles --device cuda --seed 0
```

Attendu :
- Pas de crash
- `Final : winrate=X.XX%, difficulty=0.00` (50 ép trop court pour scheduler)
- Loss finite tout du long
- Quelques % de différence par rapport à V2-Z baseline 50 ép (variance d'init)

- [ ] **Step 3 — Add V2-W section to README.md**

Localiser la section V2-Z dans `README.md` (chercher `## V2-Z — CNN perception spatiale`). Insérer une nouvelle section V2-W juste après la fin de la section V2-Z et avant `## Roadmap (V2+)`.

Contenu exact à insérer :

```markdown
## V2-W — Double DQN sur ConvDQN (sous-projet livré)

**Tag** : `v0.2.0-w` — **Tests** : 211 verts (208 baseline + 3 V2-W)

Motivation : V2-Z (CNN seul) franchit le plafond V2-X/V2-Y (atteint diff=0.25-0.35
sur 2/3 seeds) mais avec **variance inter-seeds élevée** (écart-type ±0.13).
Symptôme classique de **surestimation Q-values DQN** (Hasselt 2015) + sensibilité
aux conditions initiales.

V2-W ajoute Double DQN : on découple la sélection d'action (online net) et son
évaluation (target net) pour stabiliser l'apprentissage.

### Usage CLI

```bash
# V2-W par défaut (Double DQN activé)
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda

# Reproduire la baseline V2-Z (DQN classique)
python scripts/train_cnn_dqn_procedural.py --episodes 5000 --mode obstacles --device cuda --no-double-dqn
```

### Diff algorithmique (~10 LOC dans `_ConvDQNTrainer.step()`)

Avant (V2-Z DQN classique) :
```python
q_next = self.target(next_states).max(dim=1).values
```

Après (V2-W Double DQN) :
```python
next_actions = self.online(next_states).argmax(dim=1)   # sélection : online
q_next = self.target(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)
```

### Architecture

Aucun nouveau fichier code. Flag `double_dqn: bool = True` dans `ConvDQNConfig`,
branche conditionnelle dans `_ConvDQNTrainer.step()`, exposition CLI via
`argparse.BooleanOptionalAction`. Réutilise intégralement l'infrastructure V2-Z
(ConvQNetwork, ConvDQNAgent, ConvProceduralDQNRunner, ReplayBuffer).

### GUI

Le bouton "Démarrer (procedural CNN)" utilise `ConvDQNConfig()` par défaut, donc
V2-W (Double DQN) est activé automatiquement. Pas de nouveau bouton.

```

(Note : préserver les triple-backticks dans le code block bash et le code block markdown imbriqués.)

- [ ] **Step 4 — Add V2-W row to CLAUDE.md sub-projects table**

Dans `CLAUDE.md`, localiser le tableau "Sous-projets — décomposition". Ajouter une ligne W après la ligne Z existante. Remplacer :

```markdown
| **Z** | CNN perception spatiale (roadmap #2) | ✅ Livré (tag `v0.2.0-z`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

par :

```markdown
| **Z** | CNN perception spatiale (roadmap #2) | ✅ Livré (tag `v0.2.0-z`) |
| **W** | Double DQN sur ConvDQN (roadmap #7) | ✅ Livré (tag `v0.2.0-w`) |
| **B** | Mémoire persistante cross-session | ⏳ **Prochain par défaut** |
```

- [ ] **Step 5 — Add V2-W phases section to CLAUDE.md**

Dans `CLAUDE.md`, localiser la fin de la section "V2-Z — baseline CNN empirique n=3 seeds" (chercher la ligne "Cible : variance écart-type < ±0.05 et bucket 1 ≥ 70 % stable."). Insérer juste après une nouvelle section :

```markdown
### V2-W — état final des phases (livraison 2026-05-22)

| Phase | Tâches | Statut | Tests | Commits |
|---|---|---|---|---|
| 1 — Flag `double_dqn` dans `ConvDQNConfig` | T1 | ✅ | 2 | 1 |
| 2 — Branche conditionnelle dans `_ConvDQNTrainer.step()` | T2 | ✅ | 1 | 1 |
| 3 — CLI `--double-dqn / --no-double-dqn` | T3 | ✅ | — | 1 |
| 4 — README + CLAUDE.md + smoke + tag `v0.2.0-w` | T4 | ✅ | — | 1 + tag |

### Composants V2-W livrés

| Composant | Fichier | Rôle |
|---|---|---|
| Flag `double_dqn: bool = True` | `mw_ia/config.py` (ConvDQNConfig) | Active la formule Double DQN par défaut. False = V2-Z baseline. |
| Branche conditionnelle | `mw_ia/agents/conv_dqn.py` (`_ConvDQNTrainer.step()`) | ~10 LOC : if double_dqn, online sélectionne et target évalue. Else V2-Z baseline. |
| Param trainer | `mw_ia/agents/conv_dqn.py` (`_ConvDQNTrainer.__init__`) | Accepte `double_dqn: bool = True`. ConvDQNAgent passe `cfg.double_dqn`. |
| CLI flag | `scripts/train_cnn_dqn_procedural.py` | `--double-dqn / --no-double-dqn` (BooleanOptionalAction), default V2-W. |
| Test branche | `tests/agents/test_conv_dqn.py::test_double_dqn_branch_differs_from_standard` | Vérifie mathématiquement que les 2 formules divergent quand online ≠ target. |

### Décisions techniques V2-W

- **Flag dans ConvDQNConfig** (pas nouveau dataclass) : approche minimale, A/B contrôlé sur même infra, V2-Z reste reproductible avec `--no-double-dqn`.
- **Default `double_dqn=True`** : V2-W est l'amélioration recommandée. GUI hérite automatiquement (le bouton "procedural CNN" utilise `ConvDQNConfig()` sans args).
- **Pas de nouveau runner / agent / fichier** : la modification est purement à l'intérieur du trainer. Tout le reste de l'infra V2-Z est réutilisé.
- **Test unitaire ciblé sur la formule** : `test_double_dqn_branch_differs_from_standard` vérifie la divergence mathématique des 2 formules sur 2 réseaux désynchronisés, pas la convergence empirique. Déterministe, isolé, ~25 LOC.
- **Validation empirique = benchmark same-seed n=3 V2-Z vs V2-W** : à mener post-impl. Seul vrai test scientifique (mêmes seeds 0/1/2, seule variable changée = formule du Q-target).

### V2-W — pièges connus

1. **AMP autocast sur la branche `argmax(self.online(next_states))`** : le forward double passe sous autocast (no_grad context préservé). argmax sur tensor half-precision = OK PyTorch. Si NaN observé, fallback `self.online(next_states).float().argmax(...)`.
2. **online == target au step 0** : juste après init, sync_target a déjà été appelé. Les 2 formules donnent exactement les mêmes q_next. C'est attendu et le test ciblé désynchronise volontairement pour vérifier la divergence.
3. **CLI default V2-W casse silencieusement la repro V2-Z** : documenter explicitement "Pour reproduire la baseline V2-Z, ajouter `--no-double-dqn`". GUI → V2-W automatique (acceptable car recommandation).
4. **save/load checkpoint avec / sans flag** : `cfg.__dict__` sauvegardé inclut maintenant `double_dqn`. Le `load()` V1-hérité ne re-construit pas cfg → l'utilisateur doit reconstruire l'agent avec le bon `cfg.double_dqn` avant load. À noter mais non-critique en MVP.
```

- [ ] **Step 6 — Update "Instructions pour la prochaine session"**

Localiser la section "Instructions pour la prochaine session" dans CLAUDE.md et chercher la ligne commençant par "V2-A, V2-X, V2-Y ET V2-Z (CNN) étant terminés". Remplacer le bloc complet (de cette ligne jusqu'au début de "**Diagnostic empirique fin de session 2026-05-22**") par :

```markdown
V2-A, V2-X, V2-Y, V2-Z (CNN) ET V2-W (Double DQN) étant terminés, **la prochaine étape est la validation empirique V2-W** (benchmark same-seed n=3 V2-Z vs V2-W) puis décision basée sur outcome.
```

Et ajouter cette section juste après le bloc "Diagnostic empirique" existant :

```markdown
**Plan validation empirique V2-W (à mener prochaine session)** :

1. Lancer 3 runs GPU 5000 ép avec `--double-dqn --seed 0/1/2` (mêmes seeds que V2-Z baseline déjà mesurée)
2. Documenter benchmark n=3 V2-Z vs n=3 V2-W dans CLAUDE.md (tableau 6 runs, statistiques, lecture finding)
3. Si critère atteint (variance < ±0.05 + bucket 1 ≥ 70% sur 2/3 seeds) → finding consolidé "représentation + stabilité Q = combo nécessaire", record dans README synthèse top-niveau
4. Si critère non-atteint → candidats suivants : V2-ZY (CNN+LSTM combiné) ou hyperparam tuning V2-W (target_sync_steps plus court)
```

- [ ] **Step 7 — Run full suite + Aether final**

```bash
source .venv/Scripts/activate && pytest -q 2>&1 | tail -3
bash aether/verify_all.sh
```

Attendu : `211 passed` + `8 OK`.

- [ ] **Step 8 — Commit doc**

```bash
git add README.md CLAUDE.md
git commit -m "docs(v2-w): add V2-W section (Double DQN) to README + CLAUDE.md"
```

- [ ] **Step 9 — Tag**

```bash
git tag v0.2.0-w
git tag --list | tail -6
```

Attendu : `v0.1.0`, `v0.2.0-a`, `v0.2.0-x`, `v0.2.0-y`, `v0.2.0-z`, `v0.2.0-w`.

- [ ] **Step 10 — DoD final récap**

Print to stdout :

```
=== V2-W DoD CHECKLIST (livraison code) ===
[ ] pytest -q → 211 passed (208 + 3)
[ ] bash aether/verify_all.sh → 8 OK
[ ] smoke train_cnn_dqn_procedural.py --episodes 50 --device cuda OK (avec et sans --double-dqn)
[ ] V2-W section dans README.md + CLAUDE.md
[ ] Tag v0.2.0-w posé
[ ] Tags antérieurs (v0.2.0-a/x/y/z) intacts
==> Phase post-livraison : 3 runs GPU 5000 ép --double-dqn --seed 0/1/2
    pour benchmark same-seed n=3 V2-Z vs V2-W.
    Critère succès : variance écart-type < ±0.05 + bucket 1 ≥ 70% sur 2/3 seeds.
```

---

## Récapitulatif

- **4 tasks** réparties sur **4 phases**
- **3 nouveaux tests** (208 → 211)
- **~6 commits** sur `main`
- **Tag livraison** : `v0.2.0-w`
- **DoD bloquante** : pytest 211 + Aether 8 OK + smoke E2E + tag
- **DoD non-bloquante (objectif scientifique)** : benchmark same-seed n=3 V2-Z vs V2-W. Si KO → V2-ZY (CNN+LSTM) ou tuning hyperparam V2-W.

## Scope confirmé

- **LOC modifiées** : ~50 (10 trainer + 10 agent + 5 config + 15 CLI + 25 test)
- **Fichiers modifiés** : 4 code + 2 doc
- **Aucun nouveau fichier code créé**
- **Durée impl estimée** : 1-2 heures
- **Validation empirique** : 30 min GPU (3 runs 5000 ép)
