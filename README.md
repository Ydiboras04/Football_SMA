# Football SMA

TP guidés Systèmes Multi-Agents – football multi-agent (Python / Mesa).

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell : .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

## Lancer

```bash
python main.py          # un match complet
python experiments.py   # scenarios controles (tests des TP)
```

## Architecture du joueur (TP 1)

```
                 +-------------------+
 Terrain ------> |    PERCEPTION     |  perceive() : ballon, porteur,
 et joueurs      |      locale       |  joueurs dans un rayon de 25 m
                 +---------+---------+
                           |
                 +---------v---------+
                 |  MODELE DU MONDE  |  update_world_model() : has_ball,
                 |     (beliefs)     |  pressure, best_mate, goal_dist...
                 +---------+---------+
                           |
                 +---------v---------+   urgence
                 |  COUCHE REACTIVE  |-------------------------+
                 |   (subsomption)   |                         |
                 | N4 urgence defens.|                         |
                 | N3 proteger ballon|                         |
                 | N2 collision/touche                         |
                 +---------+---------+                         |
                           | aucune urgence                    |
                 +---------v---------+                         |
                 |COUCHE DELIBERATIVE|                         |
                 | porteur : regles  |                         |
                 |   de production   |                         |
                 | autres : automate |                         |
                 |      du TP 0      |                         |
                 +---------+---------+                         |
                           |                                   |
                 +---------v---------+                         |
                 |     EXECUTION     |<------------------------+
                 |     execute()     |  MOVE / PASS / SHOOT / ...
                 +---------+---------+
                           |
                           v
                  effet sur le terrain
```
