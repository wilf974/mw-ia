"""SumTree -- arbre binaire de sommes pour PER (Schaul 2015) O(log N).

Convention : capacite paddee a la prochaine puissance de 2 en interne
pour garantir un arbre binaire complet (toutes les feuilles a la meme
profondeur), ce qui assure que la descente find(value) parcourt les
feuilles dans l'ordre des leaf_idx.

- _tree_capacity = prochaine puissance de 2 >= capacity (>=1)
- Array de taille 2 * _tree_capacity - 1
- Feuilles aux indices [_tree_capacity - 1, 2*_tree_capacity - 2]
- Noeuds internes [0, _tree_capacity - 2]
- parent(i) = (i - 1) // 2
- left(i)   = 2*i + 1
- right(i)  = 2*i + 2

Les feuilles d'index >= capacity sont "padding" (priorite zero,
jamais selectionnees par find). L'utilisateur ne voit que des
leaf_idx dans [0, capacity).
"""
from __future__ import annotations

import numpy as np


def _next_pow2(n: int) -> int:
    """Plus petite puissance de 2 superieure ou egale a n (>=1)."""
    p = 1
    while p < n:
        p *= 2
    return p


class SumTree:
    """Sum tree O(log N) pour PER. Capacite fixe."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity doit etre >= 1, recu {capacity}")
        self._capacity = capacity
        self._tree_capacity = _next_pow2(capacity)
        # Array de taille 2*_tree_capacity - 1 (feuilles + internes)
        self._tree = np.zeros(2 * self._tree_capacity - 1, dtype=np.float64)

    @property
    def capacity(self) -> int:
        return self._capacity

    def total(self) -> float:
        """Somme totale (valeur de la racine, index 0)."""
        return float(self._tree[0])

    def update(self, leaf_idx: int, priority: float) -> None:
        """Met a jour la priorite d'une feuille et propage a la racine."""
        if not (0 <= leaf_idx < self._capacity):
            raise ValueError(
                f"leaf_idx {leaf_idx} hors [0, {self._capacity})"
            )
        # Index interne dans le tableau
        tree_idx = leaf_idx + self._tree_capacity - 1
        delta = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        # Propagation vers la racine
        parent = (tree_idx - 1) // 2
        while parent >= 0:
            self._tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def find(self, value: float) -> tuple[int, float]:
        """Trouve la feuille ou le cumul atteint `value`.

        `value` est clampe a [0.0, total()] pour eviter de retourner un
        leaf_idx dans la zone de padding (puissance de 2 superieure a capacity).
        Si l'arbre est vide (total() == 0), retourne (0, 0.0).

        Retourne (leaf_idx in [0, capacity), priority de la feuille).
        """
        # Clamp pour ne jamais descendre dans la zone padding (cas value >= total
        # ou roundoff float aux bornes du stratified sampling caller-side).
        value = max(0.0, min(value, float(self._tree[0])))
        # Descente depuis la racine
        idx = 0
        while idx < self._tree_capacity - 1:  # Tant que noeud interne
            left = 2 * idx + 1
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = 2 * idx + 2
        # idx est l'index dans le tableau, convertir en leaf_idx
        leaf_idx = idx - (self._tree_capacity - 1)
        # Clamp defensif si arbre vide (toutes priorites a 0, leaf_idx peut
        # tomber dans la zone padding via la branche "else")
        leaf_idx = min(leaf_idx, self._capacity - 1)
        return leaf_idx, float(self._tree[idx])
