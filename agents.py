"""Agents joueurs.

TP 0 : agent reactif base sur un automate a 4 etats
(CHASE, ATTACK, SUPPORT, DEFEND) -- a completer a l'etape 2.
"""

import math

import mesa

SPEED = 1.0          # metres parcourus par tick sans ballon
SPEED_WITH_BALL = 0.8


class PlayerAgent(mesa.Agent):

    def __init__(self, model, team, role, name, home):
        super().__init__(model)
        self.team = team      # "A" ou "B"
        self.role = role      # "DEF", "MID", "ATT"
        self.name = name      # ex. "A2"
        self.home = home      # position de depart / zone naturelle
        self.state = "CHASE"
        self.last_action = None

    # ------------------------------------------------------------ proprietes
    @property
    def has_ball(self):
        return self.model.get_ball_owner() is self

    def distance_to(self, point):
        return math.hypot(self.pos[0] - point[0], self.pos[1] - point[1])

    # ------------------------------------------------------------ primitive
    def move_toward(self, target, speed=None):
        """Avance d'au plus `speed` metres vers `target`, sans sortir du terrain."""
        if speed is None:
            speed = SPEED_WITH_BALL if self.has_ball else SPEED
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return
        step = min(speed, dist)
        x = self.pos[0] + dx / dist * step
        y = self.pos[1] + dy / dist * step
        space = self.model.space
        x = min(max(x, 0.0), space.x_max - 1e-6)
        y = min(max(y, 0.0), space.y_max - 1e-6)
        space.move_agent(self, (x, y))

    # ------------------------------------------------- interface commune (2.3)
    def perceive(self):
        """Observe localement l'environnement."""
        pass

    def revise_beliefs(self):
        """Met a jour l'etat interne a partir des perceptions."""
        pass

    def deliberate(self):
        """Produit ou met a jour buts, intentions ou plan."""
        pass

    def communicate(self):
        """Echange des informations avec les autres agents."""
        pass

    def act(self):
        """Execute l'action retenue."""
        pass

    def learn(self, transition):
        """Met a jour la politique apres observation d'un retour."""
        pass

    # ----------------------------------------------------------------- cycle
    def step(self):
        # Etape 2 : ici on appellera transition_state() puis reactive_action().
        pass
