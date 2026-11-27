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
python experiments.py tp3
python experiments.py tp4
```

Le modele accepte plusieurs architectures de joueur :

```python
FootballModel(architecture="REACTIVE")                  # TP 1 : automate + regles
FootballModel(architecture="BDI", commitment="single")  # TP 2 : "blind" | "single" | "open"
FootballModel(architecture="BDI_PLAN", plan_mode="closed")  # TP 3 : "closed" | "open"
FootballModel(architecture="HYBRID")                    # TP 4 : + communication (defaut)
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
