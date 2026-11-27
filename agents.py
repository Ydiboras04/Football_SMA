"""Agents joueurs.

TP 0 : agent reactif base sur un automate a 4 etats
(CHASE, ATTACK, SUPPORT, DEFEND).
"""

import math

import mesa

SPEED = 1.0          # metres parcourus par tick sans ballon
SPEED_WITH_BALL = 0.8


class PlayerAgent(mesa.Agent):

    def __init__(self, model, team, role, name, home):
        super().__init__(model)
        self.team = team      # nom de l'equipe (voir LEFT_TEAM / RIGHT_TEAM)
        self.role = role      # "DEF", "MID", "ATT"
        self.name = name      # ex. "A2"
        self.home = home      # position de depart / zone naturelle
        self.state = "CHASE"
        self.last_action = None

    @property
    def has_ball(self):
        return self.model.get_ball_owner() is self

    def distance_to(self, point):
        return math.hypot(self.pos[0] - point[0], self.pos[1] - point[1])

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

    # ============================================================ PERCEPTION
    def perceive(self):
        """Observe localement l'environnement.

        TP 0 : la perception se limite a ce dont l'automate a besoin,
        le porteur du ballon et la distance au ballon.
        """
        owner = self.model.get_ball_owner()
        ball_pos = self.model.ball.pos
        return {
            "owner": owner,
            "dist_ball": self.distance_to(ball_pos),
            "closest_mate_to_ball": self.is_closest_of_team_to(ball_pos),
        }

    def is_closest_of_team_to(self, point):
        mates = self.model.players(self.team)
        return min(mates, key=lambda p: p.distance_to(point)) is self

    def transition_state(self, percept):
        """Fonction de transition de l'automate : choisit l'etat courant
        a partir du porteur du ballon."""
        owner = percept["owner"]

        if self.has_ball:
            self.state = "ATTACK"
        elif owner is None:
            self.state = "CHASE"
        elif owner.team == self.team:
            self.state = "SUPPORT"
        else:
            self.state = "DEFEND"

    # =============================================================== ACTION
    def reactive_action(self, percept):
        """Associe une action simple a chaque etat."""
        if self.state == "CHASE":
            self.move_toward_ball()
        elif self.state == "ATTACK":
            self.move_toward_opponent_goal()
        elif self.state == "SUPPORT":
            self.move_to_support_position(percept["owner"])
        elif self.state == "DEFEND":
            # La distance au ballon intervient ici : le defenseur le plus
            # proche presse le porteur, les autres se replacent.
            if percept["closest_mate_to_ball"]:
                self.press_carrier()
            else:
                self.move_to_defensive_zone()

    def move_toward_ball(self):
        self.last_action = "move_toward_ball"
        self.move_toward(self.model.ball.pos)

    def move_toward_opponent_goal(self):
        self.last_action = "move_toward_opponent_goal"
        self.move_toward(self.model.opponent_goal(self.team))

    def move_to_support_position(self, owner):
        """Se placer devant le porteur, decale sur le cote de sa zone."""
        self.last_action = "move_to_support_position"
        forward = 10.0 * self.model.attack_direction(self.team)
        side = 8.0 if self.home[1] >= owner.pos[1] else -8.0
        self.move_toward((owner.pos[0] + forward, owner.pos[1] + side))

    def move_to_defensive_zone(self):
        """Revenir entre sa position de depart et son propre but."""
        self.last_action = "move_to_defensive_zone"
        goal = self.model.own_goal(self.team)
        self.move_toward(((self.home[0] + goal[0]) / 2,
                          (self.home[1] + goal[1]) / 2))

    def press_carrier(self):
        self.last_action = "press_carrier"
        self.move_toward(self.model.ball.pos)

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
        """Boucle perception -> decision -> action du TP 0."""
        percept = self.perceive()
        self.transition_state(percept)
        self.reactive_action(percept)
        self.log_tick(percept)

    def log_tick(self, percept):
        """Journal : etat, perception principale, action."""
        if not self.model.trace:
            return
        owner = percept["owner"]
        self.model.log(
            f"{self.name} | state={self.state:<7} | "
            f"owner={owner.name if owner else None}, "
            f"dist_ball={percept['dist_ball']:5.1f} | {self.last_action}"
        )
