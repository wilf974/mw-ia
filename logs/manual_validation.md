# Validation manuelle MW_IA V1

Date : 2026-05-21

## Suite automatisée
- [x] pytest tests/ → tout vert (voir CI)
- [x] train_tabular 1000 ép. → winrate ≥ 0.85, niveau Avancé ou Expert
- [x] train_dqn 200 ép. sur CUDA → loss décroît, pas de NaN

## GUI (à valider par l'utilisateur)
- [ ] python scripts/launch_gui.py → fenêtre 1280×800, thème dark, GridWorld visible
- [ ] Démarrer → agent bouge, 4 courbes vivantes
- [ ] Pause / Reprendre / Reset → fonctionnels
- [ ] Save / Load → roundtrip OK
- [ ] Fermeture propre du thread

## Environnement
- Python : 3.13.12
- PyTorch : 2.11.0+cu128
- GPU : NVIDIA GeForce RTX 3060 (12 Go VRAM, CC 8.6)
- OS : Windows 11 Pro
