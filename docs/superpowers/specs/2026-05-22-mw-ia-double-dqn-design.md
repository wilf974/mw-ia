# Spec V2-W — Double DQN sur ConvDQN

> **Sous-projet** : V2-W (MW_IA — Reinforcement Learning éducatif)
> **Date** : 2026-05-22
> **Statut** : Spec validée — implémentation à dérouler via `superpowers:writing-plans` puis `superpowers:subagent-driven-development`
> **Tag livraison cible** : `v0.2.0-w`
> **Sous-projets prérequis livrés** : V1 (`v0.1.0`), V2-A (`v0.2.0-a`), V2-X (`v0.2.0-x`), V2-Y (`v0.2.0-y`), V2-Z (`v0.2.0-z`)

---

## 1. Vue d'ensemble & hypothèse

### Motivation empirique consolidée n=3 (V2-Z baseline)

Validation V2-Z post-livraison sur 3 seeds GPU 5000 ép obstacles :

| Seed | Final winrate | Final diff | Bucket 1 (0.2-0.4) |
|---|---|---|---|
| 0 | 54 % | 0.10 | — vide (worst-case) |
| 1 | 65 % | 0.25 | 65 % ✓ rempli |
| 2 | 51 % | 0.35 | 51 % ✓ rempli |

**Statistiques** :
- Diff max moyenne **0.23**, écart-type **±0.13** (variance haute)
- Bucket 1 rempli **2/3 seeds** mais max **65 %** (sous le seuil 70 %)

### Hypothèse V2-W

Hasselt 2015, "Deep Reinforcement Learning with Double Q-Learning" :

> Découpler la sélection d'action (online net) et son évaluation (target net) réduit la surestimation systématique de Q\* induite par l'opérateur max sur le target net.

**Effets attendus sur V2-W vs V2-Z** :
1. **Variance inter-seeds réduite** : écart-type cible < **±0.05** (V2-Z = ±0.13) → réduction × 2.6
2. **Bucket 1 ≥ 70 %** sur **≥ 2/3 seeds** (V2-Z = 0/3 seeds atteignent 70 %)
3. **Possiblement bucket 2 (0.4-0.6) débloqué** sur les meilleurs seeds (V2-Z = jamais atteint)

### Story scientifique cible

Si V2-W atteint le critère succès, l'ablation V2-X → V2-Y → V2-Z → V2-W raconte :

1. **Mémoire seule (V2-Y LSTM)** → insuffisante (plafond diff=0.05, même que V2-X MLP)
2. **Perception spatiale seule (V2-Z CNN)** → débloque le palier mais convergence **instable**
3. **Perception spatiale + objectif stable (V2-W CNN + Double DQN)** → combo nécessaire pour curriculum **robuste**

C'est précisément l'ablation orthogonale qui rend l'étude lisible : on isole expérimentalement l'impact de chaque facteur.

---

## 2. Composants & fichiers modifiés

### Périmètre minimal (intentionnellement)

**Pas de nouveau fichier code créé**. Pas de nouveau runner. Pas de nouveau bouton GUI. Réutilise intégralement `ConvProceduralDQNRunner`, `ConvDQNAgent`, `ConvQNetwork`, `ReplayBuffer`. C'est précisément le but de l'approche flag : **A/B contrôlé sur exactement la même infrastructure**.

### Fichiers modifiés

| Fichier | Modif |
|---|---|
| `mw_ia/config.py` | + champ `double_dqn: bool = True` dans `ConvDQNConfig`. Default `True` = V2-W est l'amélioration recommandée. Pour reproduire V2-Z baseline n=3, passer explicitement `double_dqn=False`. |
| `mw_ia/agents/conv_dqn.py` | + paramètre `double_dqn: bool` dans `_ConvDQNTrainer.__init__`. + branche conditionnelle dans `_ConvDQNTrainer.step()` (~10 LOC). `ConvDQNAgent` passe `cfg.double_dqn` au trainer. |
| `scripts/train_cnn_dqn_procedural.py` | + flag `--double-dqn / --no-double-dqn` (argparse `BooleanOptionalAction`). Default `--double-dqn`. |
| `tests/agents/test_conv_dqn.py` | + 1 test `test_double_dqn_branch_differs_from_standard` (~30 LOC) |
| `README.md` + `CLAUDE.md` | Section V2-W avec hypothèse, code-diff, benchmark n=3 results. |

### Diff précis dans `_ConvDQNTrainer.step()`

**Avant V2-Z (code actuel)** :
```python
with torch.no_grad():
    q_next = self.target(next_states).max(dim=1).values
    target_q = rewards + self.gamma * q_next * (1.0 - dones)
```

**Après V2-W** :
```python
with torch.no_grad():
    if self.double_dqn:
        # Double DQN : online sélectionne, target évalue
        next_actions = self.online(next_states).argmax(dim=1)
        q_next = self.target(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)
    else:
        # DQN classique : target sélectionne ET évalue (V2-Z baseline)
        q_next = self.target(next_states).max(dim=1).values
    target_q = rewards + self.gamma * q_next * (1.0 - dones)
```

### Compatibilité Aether

V2-W ne touche pas aux hyperparams `VariantSpec` (gamma, lr, epsilon, batch, replay, target_sync). Les invariants I1-I8 restent valides sans modification. Le test `test_aether_smoke` V2-Z continue de passer car `verify_formal(VariantSpec(ConvDQNConfig()))` ne consulte pas `double_dqn`.

### Compatibilité GUI

Le bouton "Démarrer (procedural CNN)" continue d'utiliser `ConvDQNConfig()` instancié sans argument → après V2-W, la GUI utilise Double DQN par défaut. Acceptable et même souhaitable : la GUI doit refléter la recommandation actuelle.

---

## 3. Testing + validation empirique

### Test unitaire (TDD)

**1 nouveau test** dans `tests/agents/test_conv_dqn.py` :

```python
def test_double_dqn_branch_differs_from_standard() -> None:
    """Avec mêmes poids et même transition, q_next différe entre DQN et Double DQN.

    Mathématiquement :
    - DQN classique :  q_next = max_a Q_target(s', a)
    - Double DQN :     q_next = Q_target(s', argmax_a Q_online(s', a))

    Si argmax_online ≠ argmax_target, les deux formules donnent des q_next différents.
    On désynchronise volontairement online/target avant comparaison.
    """
    import torch
    from mw_ia.neural.conv_network import ConvQNetwork

    online = ConvQNetwork(in_channels=3, rows=10, cols=10, n_actions=4,
                          conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256)
    target = ConvQNetwork(in_channels=3, rows=10, cols=10, n_actions=4,
                          conv_channels=(32, 64), kernel_size=3, padding=1, fc_hidden=256)
    # Sync initial pour partir de poids identiques
    target.load_state_dict(online.state_dict())
    # Désynchroniser online en ajoutant un offset
    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.5)

    # Construire un batch reproductible
    torch.manual_seed(42)
    next_states = torch.randn(4, 3, 10, 10)

    with torch.no_grad():
        # Formule DQN classique
        q_next_dqn = target(next_states).max(dim=1).values
        # Formule Double DQN
        next_actions = online(next_states).argmax(dim=1)
        q_next_double = target(next_states).gather(1, next_actions.view(-1, 1)).squeeze(1)

    # Avec online ≠ target, les 2 formules DOIVENT diverger sur au moins une transition
    assert not torch.allclose(q_next_dqn, q_next_double), (
        "Double DQN doit différer de DQN classique quand online ≠ target"
    )
```

**Pourquoi ce design de test** : direct, déterministe, pas de side-effect agent/buffer, ~25 LOC. Teste mathématiquement la divergence des deux formules, pas la convergence empirique (qui est le job du benchmark GPU).

### Tests existants

Les 7 tests V2-Z existants (`tests/agents/test_conv_dqn.py`) restent verts. Avec `double_dqn=True` par défaut, les Q-values changent légèrement mais les invariants structurels tiennent : `global_step`, target sync, buffer push, ε-greedy, Aether smoke.

**Total cumulé attendu** : **209 tests** (208 baseline V2-Z + 1 V2-W).

### Validation empirique post-impl

**Protocole rigoureux n=3 vs n=3** (réutilise les seeds V2-Z déjà mesurés) :

| Run | Seed | Flag | Statut |
|---|---|---|---|
| V2-Z-0 | 0 | `--no-double-dqn` | déjà fait — 54 % @ diff=0.10 |
| V2-Z-1 | 1 | `--no-double-dqn` | déjà fait — 65 % @ diff=0.25 |
| V2-Z-2 | 2 | `--no-double-dqn` | déjà fait — 51 % @ diff=0.35 |
| V2-W-0 | 0 | `--double-dqn` | à lancer (même seed que V2-Z-0) |
| V2-W-1 | 1 | `--double-dqn` | à lancer (même seed que V2-Z-1) |
| V2-W-2 | 2 | `--double-dqn` | à lancer (même seed que V2-Z-2) |

**Pourquoi same-seed V2-Z vs V2-W** : c'est le seul vrai test scientifique. Avec un seed identique, l'environnement (mazes générés à chaque épisode via `seed=ep`), l'init du réseau (`torch.manual_seed(seed)`) et le rng du buffer sont identiques. La **seule variable changée** est la formule de calcul du Q-target. Si V2-W-0 ne reste pas bloqué comme V2-Z-0, c'est **causal** — pas du bruit.

**Note importante** : same-seed ne signifie PAS "trajectoires identiques post-step-1". Dès le 1er train_step, les Q-targets diffèrent entre les deux variantes → buffers divergent → trajectoires divergent. Ce qui est contrôlé, c'est le point de départ (init + 1er sample env).

### Critère succès V2-W

**Atteint si sur 3 seeds (0/1/2) avec `--double-dqn`** :

1. **Variance** : écart-type de la diff_max **< ±0.05** (V2-Z = ±0.13) — réduction × 2.6
2. **Bucket 1** ≥ 70 % sur **≥ 2/3 seeds** (V2-Z = 0/3 seeds atteignent 70 %)

**Si critère atteint** → tag `v0.2.0-w` + récit "représentation + stabilité Q = combo nécessaire".

**Si critère non atteint** → tag `v0.2.0-w` posé quand même (livraison sous-projet complète — finding négatif est aussi un finding). CLAUDE.md documente honnêtement, candidat suivant = CNN+LSTM combiné (V2-ZY) ou continuer à pousser hyperparams V2-W (target_sync_steps plus court ?).

### Documentation comparative

Section CLAUDE.md "V2-W — benchmark Double DQN" structurée comme V2-Z n=3 :
- Tableau 6 runs (V2-Z + V2-W same-seeds)
- Trajectoires scheduler par seed
- Statistiques agrégées (moyenne, écart-type, intervalle de confiance approx)
- Lecture finding (atteint / non-atteint, implications)

---

## 4. DoD + pièges anticipés

### Definition of Done

**DoD bloquante (livraison code)** :

1. ✅ `mw_ia/config.py` : champ `double_dqn: bool = True` ajouté à `ConvDQNConfig`. Validation `__post_init__` triviale (bool, pas de check supplémentaire au-delà du type checking dataclass).
2. ✅ `mw_ia/agents/conv_dqn.py` : `_ConvDQNTrainer.__init__` accepte `double_dqn: bool`, `step()` contient la branche conditionnelle (~10 LOC).
3. ✅ `mw_ia/agents/conv_dqn.py` : `ConvDQNAgent.__init__` passe `cfg.double_dqn` au trainer à la construction.
4. ✅ `scripts/train_cnn_dqn_procedural.py` : flag CLI `--double-dqn / --no-double-dqn` via `argparse.BooleanOptionalAction`, default `--double-dqn`.
5. ✅ `tests/agents/test_conv_dqn.py` : test `test_double_dqn_branch_differs_from_standard` ajouté (~30 LOC).
6. ✅ `pytest -q` → **209 passed** (208 baseline + 1 V2-W).
7. ✅ `bash aether/verify_all.sh` → 8 OK inchangé.
8. ✅ Smoke local CPU 10 ép avec `--double-dqn` ET `--no-double-dqn` — pas de crash, sortie similaire à V2-Z.
9. ✅ Section V2-W ajoutée à README.md (parallèle à V2-Z).
10. ✅ Tag `v0.2.0-w` posé sur le commit doc final.

**DoD non-bloquante (validation scientifique)** :

11. ⏳ 3 runs GPU 5000 ép avec `--double-dqn --seed 0/1/2` (~30 min total).
12. ⏳ Section "V2-W — benchmark Double DQN" dans CLAUDE.md avec tableau comparatif 6 runs + finding.
13. ⏳ Si critère atteint : documenter "story complète V2-X→V2-W" en synthèse top-niveau du README.

### Pièges anticipés

| # | Piège | Mitigation |
|---|---|---|
| 1 | **AMP autocast sur la branche `argmax(self.online(next_states))`** | Le forward double passe à autocast = OK (no_grad context + autocast compat). Si bug NaN observé, fallback : `self.online(next_states).float().argmax(...)` explicite. |
| 2 | **Asymétrie online/target juste après `target.load_state_dict(online.state_dict())`** au step 0 | online == target → DQN et Double DQN donnent exactement les mêmes `q_next`. C'est attendu (pas de divergence avant le 1er gradient step). Le test cible désynchronise volontairement avant comparaison. |
| 3 | **Test paramétrisé risque flaky** si on compare deux trajectoires entières | Mitigation : tester directement la formule sur 2 `ConvQNetwork` désynchronisés, pas via 2 trainers complets. Plus court, déterministe à 100 %. |
| 4 | **Same-seed V2-Z vs V2-W diverge dès le 1er train_step** | C'est précisément le **point** du benchmark same-seed : on isole l'effet du changement de target rule. Documenter clairement que "same seed" signifie "même init + même env reset + même rng buffer initial", PAS "trajectoire identique post-step-1". |
| 5 | **CLI default `--double-dqn` casse silencieusement la repro V2-Z** | Documenter explicitement dans README + CLAUDE.md : "Pour reproduire la baseline V2-Z, ajouter `--no-double-dqn`". L'utilisateur GUI clique le bouton procedural CNN = obtient V2-W par défaut (acceptable car recommandation). |
| 6 | **Si V2-W ne réduit PAS la variance** (hypothèse échoue) | Tag posé quand même (livraison code complète). CLAUDE.md documente honnêtement, candidat suivant = V2-ZY CNN+LSTM combiné. Finding négatif = finding aussi. |
| 7 | **save/load checkpoint avec / sans flag double_dqn** | Le `cfg.__dict__` sauvegardé inclut maintenant `double_dqn`. Load avec un `cfg` qui aurait `double_dqn` différent ne re-construit pas le trainer (pattern V1 hérité — `load()` restaure online/target/global_step, pas cfg). À noter mais non-critique en MVP. |

### Récapitulatif scope V2-W

| Métrique | Valeur |
|---|---|
| LOC ajoutées (estimation) | ~50 (10 trainer + 10 agent + 5 config + 25 CLI/test) |
| Tests ajoutés | 1 (cumul 209) |
| Fichiers modifiés | 4 code + 2 doc |
| Phases TDD prévisionnelles | 3 (config + trainer + CLI) |
| Durée estimée impl | 1-2 heures (vs V2-Z = 4-6 h) |
| Validation empirique GPU | 30 min (3 runs 5000 ép) |
| Coût/heure de découverte scientifique | Probablement le plus élevé du programme V2 |

### Pourquoi le sous-projet est intentionnellement petit

Le scope minimal est délibéré :
- **Hypothèse précisément définie** (n=3 V2-Z baseline mesurée)
- **Infrastructure existante** (ConvDQN, runner, CLI, GUI, tests, CI)
- **Modif algorithmique ciblée** (~10 LOC dans une seule fonction)
- **A/B testable proprement** (flag = ablation contrôlée)

C'est précisément le type d'expérience qui maximise le ROI scientifique : un changement minimal pour tester une hypothèse forte sur une infrastructure mûre.

---

## Annexe — récap des décisions clés

| Décision | Choix | Raison |
|---|---|---|
| Intégration | Flag `double_dqn: bool` dans `ConvDQNConfig` | Pattern minimal, A/B contrôlé sur même infra, V2-Z reste reproductible avec `--no-double-dqn` |
| Default flag | `double_dqn=True` | V2-W = amélioration recommandée. GUI utilise V2-W par défaut sans modification UI. |
| Test | 1 test paramétrisé direct sur `ConvQNetwork` | Déterministe, isolé, ~25 LOC. Pas de fichier test séparé V2-W. |
| Critère succès | Variance < ±0.05 + bucket 1 ≥ 70 % sur 2/3 | Critère mixte mesurable, capture les 2 effets attendus de Double DQN |
| Validation | n=3 same-seed V2-Z vs V2-W | Ablation contrôlée sur même init + même env, seul le target rule change |
| CLI | `--double-dqn / --no-double-dqn` (BooleanOptionalAction) | Compatibilité argparse standard, default V2-W |
| GUI | Pas de bouton dédié | Le bouton "procedural CNN" utilise default `ConvDQNConfig()` → V2-W automatique |
| Tag | `v0.2.0-w` même si critère non-atteint | Livraison sous-projet ≠ publication. Finding négatif documenté = aussi du progrès. |

---

**Fin de la spec V2-W.**
