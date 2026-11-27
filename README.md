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
python experiments.py tp5   # entrainement Q-learning (~1 min), CSV dans results/
python experiments.py tp6   # comparaison des configurations (~4 min), CSV + traces dans results/
```

Le modele accepte plusieurs architectures de joueur :

```python
FootballModel(architecture="REACTIVE")                  # TP 1 : automate + regles
FootballModel(architecture="BDI", commitment="single")  # TP 2 : "blind" | "single" | "open"
FootballModel(architecture="BDI_PLAN", plan_mode="closed")  # TP 3 : "closed" | "open"
FootballModel(architecture="HYBRID")                    # TP 4 : + communication (defaut)

# TP 5 : une equipe apprend (Q-learning) contre un adversaire a regles fixes
learner = QLearner(epsilon=0.2)
FootballModel(team_architectures={"REAL_MADRID": "RL", "BARCELONA": "REACTIVE"},
              learners={"REAL_MADRID": learner})

# TP 6 : joueur hybride complet (BDI + plan + communication + Q-learning)
FootballModel(team_architectures={"REAL_MADRID": "HYBRID", "BARCELONA": "REACTIVE"},
              learners={"REAL_MADRID": learner}, collect_data=True)
```

## Architecture du joueur final (TP 6)

```
  messages recus (INFORM, REQUEST, PROPOSE, CONFIRM, REFUSE)
        |
        v
 +-------------+    +-------------------------+
 | PERCEPTION  |--->|  REVISION DES CROYANCES |  source, tick, confiance ;
 |   locale    |    |  (beliefs.py)           |  messages des coequipiers
 +-------------+    +------------+------------+
                                 |
                    +------------v------------+   urgence (1)
                    |     COUCHE REACTIVE     |----------------------------+
                    | N4 > N3 > N2 (subsomp.) |                            |
                    +------------+------------+                            |
                                 | pas d'urgence                           |
                    +------------v------------+                            |
                    |  BDI : desirs -> (2)    |  engagement aveugle /      |
                    |  intention              |  simple / ouvert           |
                    +------------+------------+                            |
                                 |                                         |
                    +------------v------------+   REQUEST / CONFIRM (4)    |
                    |  PLANIFICATION (3)      |<-------------------------> |
                    |  BFS, plan d'equipe,    |   coequipiers              |
                    |  boucle fermee, replan  |                            |
                    +------------+------------+                            |
                                 | action proposee + alternatives          |
                    +------------v------------+                            |
                    |  Q-LEARNING (5)         |  ne contredit le plan que  |
                    |  choix parmi candidats  |  si Q nettement meilleur   |
                    +------------+------------+                            |
                                 |                                         |
                    +------------v------------+                            |
                    |       EXECUTION         |<---------------------------+
                    +------------+------------+
                                 |  r, s'  (tick suivant)
                                 +------> mise a jour de Q
```

(1) a (5) : regle d'arbitrage 6.1 du sujet.

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
