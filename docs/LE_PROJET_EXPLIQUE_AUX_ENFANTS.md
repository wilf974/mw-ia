# 🤖 L'histoire de Glyph, le petit robot qui apprend tout seul

> Une histoire pour comprendre le projet **MW_IA**, même si tu as 8 ans.
> (Et même les grands vont comprendre des trucs malins ici 😉)

---

## 🎮 1. C'est quoi ce projet ?

Imagine un petit robot. On va l'appeler **Glyph**.

Glyph est lâché dans un **labyrinthe** (un peu comme un jeu vidéo, vu de dessus,
comme Pac-Man). Quelque part dans le labyrinthe, il y a une **étoile** ⭐.

Le but de Glyph : **trouver l'étoile**, le plus vite possible, sans foncer dans les murs.

Le truc magique, c'est que **personne ne lui montre le chemin**.
Au début, Glyph est complètement nul : il tourne en rond, il se cogne, il se perd.

Mais à chaque essai, on lui dit juste :
- « Bravo ! » 🎉 quand il touche l'étoile,
- « Raté... » 😕 quand il se perd.

Et à force d'essayer **des milliers de fois**, Glyph **apprend tout seul** le bon chemin.
Ça s'appelle l'**apprentissage par renforcement** (en anglais : *Reinforcement Learning*).
C'est exactement comme quand tu apprends à faire du vélo : tu tombes, tu te relèves,
et un jour... ça marche, sans que personne te tienne !

---

## 🧠 2. Le cerveau de Glyph

Glyph a un **cerveau électronique** (on appelle ça un *réseau de neurones*).
C'est une grosse calculatrice qui regarde le labyrinthe et décide :
« Je vais à gauche ? à droite ? en haut ? en bas ? »

Au début, son cerveau est tout vide. Petit à petit, il se remplit de bonnes idées.

Dans le projet, on a construit **plusieurs cerveaux différents** pour Glyph,
pour voir lequel est le plus malin. Comme si on testait plusieurs élèves dans une classe !

---

## 🏫 3. On ne jette pas Glyph dans le grand bain tout de suite

Si on mettait Glyph direct dans un labyrinthe **super dur**, il abandonnerait.
Trop difficile = on apprend rien.

Alors on fait comme une vraie école : on commence **FACILE**, puis on rend ça
**de plus en plus dur** seulement quand Glyph est prêt.

- Labyrinthe vide → facile 🟢
- Quelques murs → moyen 🟡
- Plein de murs partout → difficile 🔴

Ça s'appelle le **curriculum** (comme le programme scolaire : CP, puis CE1, puis CE2...).
Un petit prof automatique surveille Glyph : s'il réussit bien, il rend le jeu plus dur.
S'il galère, il rend le jeu plus facile. Trop fort, ce prof ! 👨‍🏫

---

## 🛠️ 4. Tout ce qu'on a déjà construit (le « déjà fait »)

On a amélioré Glyph **petit bout par petit bout**. Voici ses grandes étapes,
racontées simplement :

| Étape | Ce qu'on a donné à Glyph | En mots d'enfant |
|---|---|---|
| **V1** | Son tout premier cerveau | Glyph naît et apprend un labyrinthe simple. Ça marche ! |
| **V2-A** | Des **règles de sécurité** | On met des garde-fous pour qu'il n'apprenne pas n'importe quoi. |
| **V2-X** | Des **labyrinthes au hasard** | On arrête de lui montrer toujours le même ! Sinon il triche en mémorisant. |
| **V2-Y** | Une **mémoire** (LSTM) | Glyph se souvient des cases qu'il vient de voir. Il devient moins tête en l'air. |
| **V2-Z** | Des **yeux** (CNN) | Glyph "voit" enfin le labyrinthe comme une image, pas juste des chiffres. 👀 **Grosse avancée !** |
| **V2-W** | Moins de **vantardise** (Double DQN) | Avant, Glyph se croyait trop fort. On le rend plus réaliste. |
| **V2-V** | Un **vrai examen** | On teste Glyph sur des labyrinthes **jamais vus**. On garde son meilleur essai. |
| **V2-ZY** | Les **yeux + la mémoire** ensemble | Le combo ! Très puissant... mais parfois trop nerveux. |
| **V2-U** | Du **calme** (Polyak) | On apprend à Glyph à changer d'avis **doucement**, pas d'un coup. 🧘 **Le truc qui change tout !** |
| **V2-B0 / B1a** | Des essais de **mémoire spéciale** | On a essayé de lui faire rejouer ses meilleurs souvenirs... mais bon, ça n'a pas vraiment aidé. |

### 💡 La plus grande découverte du projet

Quand on a donné à Glyph **des yeux 👀 + de la mémoire 🧠 + du calme 🧘** en même temps,
il est devenu **vraiment bon** :

- Dans un petit labyrinthe (10×10), il réussit **9 fois sur 10** ! 🥳
- Dans un labyrinthe plus grand (15×15), il réussit **6 fois sur 10** — pas mal, c'est plus dur !

On a appris une **leçon importante** :
> Un seul super-pouvoir ne suffit pas. C'est en **combinant** les yeux, la mémoire et le calme
> que Glyph devient malin et **régulier** (il ne fait plus n'importe quoi un jour sur deux).

### 🔬 Une autre leçon rigolote

On a aussi découvert un truc bizarre : une astuce qui **aide** Glyph dans les **petits**
labyrinthes le **gêne** dans les **grands** ! Comme un médicament qui marche pour un petit
bobo mais pas pour un gros. En science, c'est important de noter aussi ce qui **ne marche pas**.

---

## 🌟 5. La grande vision (le rêve du projet)

On ne veut pas juste un robot qui sort d'un labyrinthe.

Le **grand rêve**, c'est une **IA qui s'améliore toute seule** :
- elle **teste** ses propres idées pour devenir meilleure,
- elle **se souvient** de ce qu'elle a appris, même après qu'on éteint l'ordinateur,
- elle **respecte des règles** qu'on vérifie pour qu'elle reste sage et sûre.

Et un jour, très loin, ce cerveau pourrait ne plus servir à se déplacer dans un labyrinthe...
mais à **discuter** et **réfléchir** comme un assistant intelligent. 🗣️✨

C'est un projet pour **apprendre comment fonctionne l'intelligence artificielle**,
en construisant tout étape par étape, comme des LEGO. 🧱

---

## 🚧 6. Ce qu'il reste à faire (la suite de l'aventure)

Glyph sait bien se débrouiller dans les petits labyrinthes.
Mais dans les **grands**, il bloque encore un peu. Alors on a joué au **détective** 🔍 :
on a fait **3 expériences-tests** pour trouver ce qui le bloque vraiment, en lui donnant
des super-pouvoirs « pour tricher » et en regardant lesquels l'aident :

1. **🔭 De meilleurs yeux** — on lui a montré exactement où était l'étoile. Résultat : **ça ne l'aide pas !** Donc ses yeux vont déjà très bien.
2. **🗺️ Mieux compter ses pas** — on a changé sa façon de calculer les récompenses. Résultat : **ça ne l'aide pas non plus**, ça le perturbe même.
3. **🧭 Le rendre curieux** — on lui a donné envie d'aller voir les endroits **nouveaux** au lieu de tourner en rond. Résultat : **ÇA, ça l'aide !** 🎉

**La découverte :** le vrai problème de Glyph dans les grands labyrinthes, c'est qu'il
**n'explore pas assez** — il ne tombe pas assez souvent sur le bon chemin pour l'apprendre.
Ce n'est ni ses yeux, ni sa mémoire, ni son calcul. C'est sa **curiosité**.

La prochaine grande étape, du coup : lui construire une **vraie curiosité** 🧭✨ —
lui apprendre à avoir *envie* de découvrir, comme un explorateur qui veut voir
ce qu'il y a derrière chaque mur.

Et plus tard, les grands objectifs du projet :
- **Mémoire qui dure** (même éteint, Glyph se souvient).
- **Glyph qui se note lui-même** (il juge si son chemin était bon).
- **Glyph qui apprend sans oublier** ce qu'il savait avant.
- **Glyph qui invente ses propres améliorations** tout seul. 🤯

---

## 📏 7. Le petit lexique des grands mots

| Mot compliqué | Ce que ça veut dire vraiment |
|---|---|
| **Reinforcement Learning** | Apprendre en essayant, avec des « bravo » et des « raté ». |
| **Réseau de neurones** | Le cerveau électronique de Glyph. |
| **CNN** | Les **yeux** de Glyph (il voit le labyrinthe comme une photo). |
| **LSTM** | La **mémoire** de Glyph (il se souvient d'où il vient). |
| **Curriculum** | Le programme scolaire : commencer facile, finir difficile. |
| **Polyak** | Apprendre à changer d'avis **tout doucement**, pour rester calme. |
| **Double DQN** | L'astuce pour que Glyph **arrête de se vanter**. |
| **Eval / examen** | Tester Glyph sur des labyrinthes **jamais vus**, pour savoir s'il a vraiment compris. |

---

*Fait avec ❤️ pour comprendre l'IA en s'amusant.
Le vrai projet technique est dans le fichier `CLAUDE.md` et le `README.md` — mais c'est pour les grands ! 😄*
