import math

import mesa
from mesa.space import ContinuousSpace

from agents import PlayerAgent

# --- Dimensions du terrain (metres) ----------------------------------------
FIELD_W = 105.0
FIELD_H = 68.0
GOAL_Y_MIN = 28.0   # le but occupe 28 <= y <= 40 sur la ligne de but
GOAL_Y_MAX = 40.0

# --- Regles simplifiees ------------------------------------------------------
CONTROL_RADIUS = 1.0  # distance pour prendre un ballon libre
TACKLE_RADIUS = 1.0   # distance pour tenter de voler le ballon au porteur
TACKLE_PROB = 0.3     # probabilite de reussite d'un tacle par tick

# Passe et tir (TP 1) : actions instantanees, resultat tire au hasard.
INTERCEPT_RADIUS = 2.0  # un adversaire a moins de 2 m de la trajectoire peut couper
INTERCEPT_PROB = 0.5    # probabilite qu'il intercepte effectivement
PASS_MAX_RANGE = 40.0   # au-dela, la passe est tres imprecise
SHOT_RANGE_SCALE = 25.0 # probabilite de marquer = 0.75 - distance / 25

# --- Equipes ------------------------------------------------------------------
# Le reste du code ne depend jamais du nom des equipes, seulement de leur cote :
# changer un nom ici suffit.
LEFT_TEAM = "REAL_MADRID"   # defend le but de gauche (x = 0), attaque a droite
RIGHT_TEAM = "BARCELONA"    # defend le but de droite (x = 105), attaque a gauche

# Positions de depart (x, y) de chaque equipe.
# L'equipe A (LEFT_TEAM) defend le but de gauche (x = 0) et attaque vers la droite.
FORMATION_A = [
    ("DEF", (20.0, 34.0)),
    ("MID", (35.0, 20.0)),
    ("ATT", (45.0, 48.0)),
]

# L'equipe B (RIGHT_TEAM) defend le but de droite (x = 105) et attaque vers la gauche.
FORMATION_B = [
    ("DEF", (85.0, 34.0)),
    ("MID", (70.0, 48.0)),
    ("ATT", (60.0, 20.0)),
]


def distance(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def distance_to_segment(p, a, b):
    """Distance du point p au segment [a, b] (trajectoire d'une passe)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return distance(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2
    t = max(0.0, min(1.0, t))
    return distance(p, (ax + t * dx, ay + t * dy))


class Ball:
    """Le ballon : soit libre (owner=None), soit colle a son porteur."""

    def __init__(self, pos):
        self.pos = pos
        self.owner = None

    def follow_owner(self):
        if self.owner is not None:
            self.pos = self.owner.pos


class FootballModel(mesa.Model):

    def __init__(self, seed=None, verbose=True):
        super().__init__(rng=seed)
        self.verbose = verbose   # affiche les evenements (prises, buts)
        self.trace = verbose     # affiche le journal de chaque joueur
        self.space = ContinuousSpace(FIELD_W, FIELD_H, torus=False)
        self.ball = Ball((FIELD_W / 2, FIELD_H / 2))
        self.score = {LEFT_TEAM: 0, RIGHT_TEAM: 0}

        # Indicateurs (utiles pour les comparaisons des TP suivants)
        self.total_passes = 0
        self.successful_passes = 0
        self.total_shots = 0

        formations = {LEFT_TEAM: FORMATION_A, RIGHT_TEAM: FORMATION_B}
        for team, formation in formations.items():
            for i, (role, home) in enumerate(formation, start=1):
                player = PlayerAgent(self, team, role, name=f"{team}{i}", home=home)
                self.space.place_agent(player, home)

    def get_ball_owner(self):
        return self.ball.owner

    def players(self, team=None):
        return [a for a in self.agents if team is None or a.team == team]

    def get_player(self, name):
        return next(p for p in self.agents if p.name == name)

    def attack_direction(self, team):
        """+1 si `team` attaque vers la droite (x croissant), -1 sinon."""
        return 1 if team == LEFT_TEAM else -1

    def opponent_goal(self, team):
        """Centre du but que `team` doit attaquer."""
        x = FIELD_W if self.attack_direction(team) > 0 else 0.0
        return (x, (GOAL_Y_MIN + GOAL_Y_MAX) / 2)

    def own_goal(self, team):
        """Centre du but que `team` doit defendre."""
        x = 0.0 if self.attack_direction(team) > 0 else FIELD_W
        return (x, (GOAL_Y_MIN + GOAL_Y_MAX) / 2)

    def score_text(self):
        return (f"{LEFT_TEAM} {self.score[LEFT_TEAM]} - "
                f"{self.score[RIGHT_TEAM]} {RIGHT_TEAM}")

    # ---------------------------------------------------------------- regles
    def give_ball(self, player):
        self.ball.owner = player
        self.ball.follow_owner()

    def resolve_ball(self):
        """Prise d'un ballon libre ou tacle sur le porteur."""
        owner = self.ball.owner

        if owner is None:
            close = [p for p in self.players()
                     if distance(p.pos, self.ball.pos) <= CONTROL_RADIUS]
            if close:
                winner = min(close, key=lambda p: distance(p.pos, self.ball.pos))
                self.give_ball(winner)
                self.log(f"{winner.name} recupere le ballon libre")
            return

        tacklers = [p for p in self.players()
                    if p.team != owner.team
                    and distance(p.pos, owner.pos) <= TACKLE_RADIUS]
        for p in tacklers:
            if self.random.random() < TACKLE_PROB:
                self.give_ball(p)
                self.log(f"{p.name} prend le ballon a {owner.name}")
                return

    def drop_ball(self, pos):
        """Le ballon devient libre a la position `pos` (bornee au terrain)."""
        x = min(max(pos[0], 0.0), FIELD_W - 1e-6)
        y = min(max(pos[1], 0.0), FIELD_H - 1e-6)
        self.ball.owner = None
        self.ball.pos = (x, y)

    def attempt_pass(self, passer, receiver):
        """Passe instantanee de `passer` vers `receiver`.

        1. un adversaire proche de la trajectoire peut intercepter ;
        2. sinon la passe arrive avec une probabilite qui baisse avec la
           distance ; si elle est ratee, le ballon tombe libre pres du receveur.
        Renvoie "PASS_OK", "PASS_INTERCEPTED" ou "PASS_MISSED".
        """
        self.total_passes += 1
        start, end = passer.pos, receiver.pos

        for opp in self.players():
            if opp.team == passer.team:
                continue
            if (distance_to_segment(opp.pos, start, end) <= INTERCEPT_RADIUS
                    and self.random.random() < INTERCEPT_PROB):
                self.give_ball(opp)
                self.log(f"passe {passer.name} -> {receiver.name} "
                         f"INTERCEPTEE par {opp.name}")
                return "PASS_INTERCEPTED"

        accuracy = max(0.4, 1.0 - distance(start, end) / PASS_MAX_RANGE)
        if self.random.random() < accuracy:
            self.give_ball(receiver)
            self.successful_passes += 1
            self.log(f"passe {passer.name} -> {receiver.name} reussie")
            return "PASS_OK"

        offset = (self.random.uniform(-4, 4), self.random.uniform(-4, 4))
        self.drop_ball((end[0] + offset[0], end[1] + offset[1]))
        self.log(f"passe {passer.name} -> {receiver.name} ratee, ballon libre")
        return "PASS_MISSED"

    def attempt_shot(self, shooter):
        """Tir instantane : probabilite de but decroissante avec la distance.
        En cas d'echec, le ballon revient au defenseur adverse le plus proche
        de son but (sortie de but simplifiee).
        Renvoie "GOAL" ou "SHOT_MISSED".
        """
        self.total_shots += 1
        goal = self.opponent_goal(shooter.team)
        dist = distance(shooter.pos, goal)
        p_goal = max(0.05, 0.75 - dist / SHOT_RANGE_SCALE)

        if self.random.random() < p_goal:
            self.log(f"{shooter.name} tire de {dist:.1f} m ... BUT !")
            self.goal_scored(shooter.team)
            return "GOAL"

        defenders = [p for p in self.players() if p.team != shooter.team]
        keeper = min(defenders, key=lambda p: distance(p.pos, goal))
        self.give_ball(keeper)
        self.log(f"{shooter.name} tire de {dist:.1f} m ... rate, "
                 f"ballon a {keeper.name}")
        return "SHOT_MISSED"

    def check_goal(self):
        owner = self.ball.owner
        if owner is None:
            return
        x, y = owner.pos
        in_goal_mouth = GOAL_Y_MIN <= y <= GOAL_Y_MAX
        goal_line_x = self.opponent_goal(owner.team)[0]
        if in_goal_mouth and abs(x - goal_line_x) <= 0.5:
            self.goal_scored(owner.team)

    def goal_scored(self, team):
        self.score[team] += 1
        self.log(f"BUT de {team} ! {self.score_text()}")
        self.kickoff()

    def kickoff(self):
        """Replace tous les joueurs et remet le ballon libre au centre."""
        for p in self.players():
            self.space.move_agent(p, p.home)
        self.ball.owner = None
        self.ball.pos = (FIELD_W / 2, FIELD_H / 2)

    # ----------------------------------------------------------------- cycle
    def step(self):
        # 1. chaque joueur percoit, decide et se deplace (ordre aleatoire)
        self.agents.shuffle_do("step")
        # 2. le ballon suit son porteur
        self.ball.follow_owner()
        # 3. regles du jeu
        self.resolve_ball()
        self.check_goal()

    def log(self, text):
        if self.verbose:
            print(f"[tick {self.steps:3d}] {text}")
