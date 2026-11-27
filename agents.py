"""Agents joueurs.

TP 0 : automate reactif a 4 etats (CHASE, ATTACK, SUPPORT, DEFEND).
TP 1 : architecture horizontale explicite

    perception -> modele du monde (beliefs)
               -> couche reactive (subsomption)  --urgence--> execution
               -> couche deliberative                       -> execution
                    * porteur      : regles de production (SHOOT / PASS / DRIBBLE)
                    * non-porteur  : automate du TP 0
"""

import math

import mesa

# --- Deplacement -------------------------------------------------------------
SPEED = 1.0              # metres parcourus par tick sans ballon
SPEED_WITH_BALL = 0.8

# --- Perception --------------------------------------------------------------
PERCEPTION_RADIUS = 25.0  # le joueur ne voit que les joueurs dans ce rayon
PRESSURE_HIGH = 3.0       # adversaire a moins de 3 m  -> pression HIGH
PRESSURE_MEDIUM = 6.0     # adversaire a moins de 6 m  -> pression MEDIUM
FREE_MARGIN = 4.0         # coequipier libre : aucun adversaire a moins de 4 m

# --- Couche reactive ---------------------------------------------------------
DANGER_ZONE = 25.0        # porteur adverse a moins de 25 m de notre but
STRONG_PRESSURE = 1.5     # adversaire au contact du porteur
COLLISION_DIST = 1.5      # coequipier trop proche
SIDELINE_MARGIN = 0.5     # trop pres de la ligne de touche

# --- Regles de production ----------------------------------------------------
SHOOT_RANGE = 16.5        # distance au but pour tirer (surface de reparation)


# =============================================================================
# 1.3  Regles de production : (nom, condition(beliefs), action(agent), priorite)
# =============================================================================
RULES = [
    (
        "shoot_if_close",
        lambda b: b["has_ball"] and b["goal_dist"] < SHOOT_RANGE,
        lambda a: ("SHOOT", None),
        100,
    ),
    (
        "pass_if_pressed",
        lambda b: (b["has_ball"]
                   and b["pressure"] == "HIGH"
                   and b["best_mate"] is not None),
        lambda a: ("PASS", a.beliefs["best_mate"]),
        80,
    ),
    (
        "dribble_forward",
        lambda b: b["has_ball"],
        lambda a: ("DRIBBLE", None),
        20,
    ),
]


class PlayerAgent(mesa.Agent):

    def __init__(self, model, team, role, name, home):
        super().__init__(model)
        self.team = team      # nom de l'equipe (voir LEFT_TEAM / RIGHT_TEAM)
        self.role = role      # "DEF", "MID", "ATT"
        self.name = name
        self.home = home      # position de depart / zone naturelle
        self.state = "CHASE"  # etat de l'automate (TP 0)

        # Modele du monde : memoire de travail des regles de production
        self.beliefs = {}

        # Explicabilite : qui a decide quoi, et pourquoi
        self.last_module = None        # "REACTIVE:..." / "RULES:..." / "AUTOMATON:..."
        self.last_rule = None
        self.applicable_rules = []
        self.inhibited_decision = None  # decision deliberative ecrasee par la reaction
        self.last_action = None
        self.last_result = None

    # ------------------------------------------------------------ outils
    @property
    def has_ball(self):
        return self.model.get_ball_owner() is self

    def distance_to(self, point):
        return math.hypot(self.pos[0] - point[0], self.pos[1] - point[1])

    # =================================================================
    # CYCLE (1.1) : step() ne decide rien, il orchestre les modules
    # =================================================================
    def step(self):
        percept = self.perceive()

        self.update_world_model(percept)

        # La couche reactive peut prendre la main si une urgence est detectee.
        urgent = self.reactive_layer()

        if urgent is not None:
            if self.model.trace:
                # Pour le journal seulement : ce que la deliberation aurait fait.
                reactive_module = self.last_module
                self.inhibited_decision = self.deliberative_layer()
                self.last_module = reactive_module
                self.applicable_rules = []
            self.execute(urgent)
            self.log_tick()
            return

        self.inhibited_decision = None
        decision = self.deliberative_layer()

        self.execute(decision)
        self.log_tick()

    # =================================================================
    # PERCEPTION : ce que le joueur observe localement
    # =================================================================
    def perceive(self):
        """Le ballon et son porteur sont toujours visibles (simplification) ;
        les autres joueurs ne sont vus que dans un rayon PERCEPTION_RADIUS."""
        visible = self.model.space.get_neighbors(
            self.pos, PERCEPTION_RADIUS, include_center=False)
        return {
            "ball_pos": self.model.ball.pos,
            "owner": self.model.get_ball_owner(),
            "mates": [p for p in visible if p.team == self.team],
            "opponents": [p for p in visible if p.team != self.team],
        }

    # =================================================================
    # MODELE DU MONDE : transformer la perception en faits exploitables
    # =================================================================
    def update_world_model(self, percept):
        b = self.beliefs
        owner = percept["owner"]
        ball_pos = percept["ball_pos"]
        opponents = percept["opponents"]

        nearest_opp = min((self.distance_to(o.pos) for o in opponents),
                          default=math.inf)

        b["has_ball"] = self.has_ball
        b["owner"] = owner
        b["ball_pos"] = ball_pos
        b["dist_ball"] = self.distance_to(ball_pos)
        b["goal_dist"] = self.distance_to(self.model.opponent_goal(self.team))
        b["nearest_opp_dist"] = nearest_opp
        b["pressure"] = ("HIGH" if nearest_opp < PRESSURE_HIGH
                         else "MEDIUM" if nearest_opp < PRESSURE_MEDIUM
                         else "LOW")
        b["best_mate"] = self.find_best_teammate(percept)
        b["closest_mate_to_ball"] = all(
            m.distance_to(ball_pos) >= b["dist_ball"] for m in percept["mates"])
        b["mate_too_close"] = min(
            percept["mates"], key=lambda m: self.distance_to(m.pos), default=None)
        if (b["mate_too_close"] is not None
                and self.distance_to(b["mate_too_close"].pos) >= COLLISION_DIST):
            b["mate_too_close"] = None
        b["opp_carrier_near_our_goal"] = (
            owner is not None and owner.team != self.team
            and owner.distance_to(self.model.own_goal(self.team)) < DANGER_ZONE)

    def find_best_teammate(self, percept):
        """Coequipier visible, libre (aucun adversaire a moins de FREE_MARGIN)
        et le plus avance vers le but adverse."""
        direction = self.model.attack_direction(self.team)
        free = [m for m in percept["mates"]
                if all(o.distance_to(m.pos) >= FREE_MARGIN
                       for o in percept["opponents"])]
        if not free:
            return None
        return max(free, key=lambda m: m.pos[0] * direction)

    # =================================================================
    # 1.2 COUCHE REACTIVE (subsomption)
    #   Niveau 4 : urgence defensive devant le but
    #   Niveau 3 : proteger le ballon sous forte pression
    #   Niveau 2 : eviter collision / sortie de terrain
    #   Niveau 1 : deplacement de base -> laisse a la couche deliberative
    # Le premier niveau declenche inhibe tous les niveaux inferieurs.
    # =================================================================
    def reactive_layer(self):
        if self.danger_near_own_goal():
            self.last_module = "REACTIVE:N4_urgence_defensive"
            return ("INTERCEPT", None)

        if self.has_ball and self.is_under_strong_pressure():
            self.last_module = "REACTIVE:N3_protection_ballon"
            return ("PROTECT_BALL", None)

        if not self.has_ball and self.beliefs["mate_too_close"] is not None:
            self.last_module = "REACTIVE:N2_collision"
            return ("AVOID_COLLISION", self.beliefs["mate_too_close"])

        if self.near_sideline():
            self.last_module = "REACTIVE:N2_sortie_terrain"
            return ("STAY_IN_BOUNDS", None)

        return None

    def danger_near_own_goal(self):
        b = self.beliefs
        return b["opp_carrier_near_our_goal"] and b["closest_mate_to_ball"]

    def is_under_strong_pressure(self):
        """Adversaire au contact ET aucune solution de passe.

        Sans la seconde condition, le reflexe se declenche a chaque tick
        tant que le defenseur colle le porteur : il inhibe la passe en
        permanence et le porteur finit toujours par perdre le ballon.
        """
        return (self.beliefs["nearest_opp_dist"] < STRONG_PRESSURE
                and self.beliefs["best_mate"] is None)

    def near_sideline(self):
        y = self.pos[1]
        return y < SIDELINE_MARGIN or y > self.model.space.y_max - SIDELINE_MARGIN

    # =================================================================
    # COUCHE DELIBERATIVE
    # =================================================================
    def deliberative_layer(self):
        self.transition_state()
        if self.beliefs["has_ball"]:
            return self.production_decision()
        return self.automaton_decision()

    # --- 1.3 Regles de production (porteur du ballon) -----------------
    def production_decision(self):
        applicable = []
        for name, condition, action, priority in RULES:
            if condition(self.beliefs):
                applicable.append((priority, name, action))

        self.applicable_rules = [name for _, name, _ in
                                 sorted(applicable, key=lambda r: -r[0])]

        if not applicable:
            self.last_rule = None
            self.last_module = "RULES:aucune"
            return ("HOLD", None)

        # Resolution de conflit : la regle de plus forte priorite l'emporte.
        _, rule_name, action = max(applicable, key=lambda r: r[0])
        self.last_rule = rule_name
        self.last_module = f"RULES:{rule_name}"
        return action(self)

    # --- Automate du TP 0 (joueurs sans ballon) -----------------------
    def transition_state(self):
        owner = self.beliefs["owner"]
        if self.beliefs["has_ball"]:
            self.state = "ATTACK"
        elif owner is None:
            self.state = "CHASE"
        elif owner.team == self.team:
            self.state = "SUPPORT"
        else:
            self.state = "DEFEND"

    def automaton_decision(self):
        self.applicable_rules = []
        self.last_module = f"AUTOMATON:{self.state}"
        if self.state == "CHASE":
            return ("MOVE_TO_BALL", None)
        if self.state == "SUPPORT":
            return ("SUPPORT", self.beliefs["owner"])
        # DEFEND : le plus proche presse le porteur, les autres se replacent.
        if self.beliefs["closest_mate_to_ball"]:
            return ("PRESS", None)
        return ("DEFEND_ZONE", None)

    # =================================================================
    # EXECUTION : traduit une decision (nom, argument) en effet sur le monde
    # =================================================================
    def execute(self, action):
        name, arg = action
        self.last_action = action
        result = None

        if name in ("MOVE_TO_BALL", "PRESS", "INTERCEPT"):
            self.move_toward(self.model.ball.pos)
        elif name == "DRIBBLE":
            self.move_toward(self.model.opponent_goal(self.team))
        elif name == "SUPPORT":
            self.move_to_support_position(arg)
        elif name == "DEFEND_ZONE":
            self.move_to_defensive_zone()
        elif name == "PROTECT_BALL":
            self.protect_ball()
        elif name == "AVOID_COLLISION":
            self.move_away_from(arg.pos)
        elif name == "STAY_IN_BOUNDS":
            self.move_toward((self.pos[0], self.model.space.y_max / 2))
        elif name == "PASS":
            result = self.model.attempt_pass(self, arg)
        elif name == "SHOOT":
            result = self.model.attempt_shot(self)
        elif name == "HOLD":
            pass
        else:
            raise ValueError(f"action inconnue : {name}")

        self.last_result = result
        return result

    # --- Actions motrices ---------------------------------------------
    def move_toward(self, target, speed=None):
        """Avance d'au plus `speed` metres vers `target`, sans sortir du terrain."""
        if speed is None:
            speed = SPEED_WITH_BALL if self.has_ball else SPEED
        dx, dy = target[0] - self.pos[0], target[1] - self.pos[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return
        step = min(speed, dist)
        space = self.model.space
        x = min(max(self.pos[0] + dx / dist * step, 0.0), space.x_max - 1e-6)
        y = min(max(self.pos[1] + dy / dist * step, 0.0), space.y_max - 1e-6)
        space.move_agent(self, (x, y))

    def move_away_from(self, point):
        dx, dy = self.pos[0] - point[0], self.pos[1] - point[1]
        dist = math.hypot(dx, dy) or 1.0
        self.move_toward((self.pos[0] + dx / dist * 5, self.pos[1] + dy / dist * 5))

    def move_to_support_position(self, owner):
        """Se placer devant le porteur, decale sur le cote de sa zone."""
        forward = 10.0 * self.model.attack_direction(self.team)
        side = 8.0 if self.home[1] >= owner.pos[1] else -8.0
        margin = SIDELINE_MARGIN + 2.0
        y = min(max(owner.pos[1] + side, margin), self.model.space.y_max - margin)
        self.move_toward((owner.pos[0] + forward, y))

    def move_to_defensive_zone(self):
        """Revenir entre sa position de depart et son propre but."""
        goal = self.model.own_goal(self.team)
        self.move_toward(((self.home[0] + goal[0]) / 2,
                          (self.home[1] + goal[1]) / 2))

    def protect_ball(self):
        """S'eloigner de l'adversaire le plus proche tout en gardant
        un peu de progression vers le but."""
        opponents = [p for p in self.model.players() if p.team != self.team]
        nearest = min(opponents, key=lambda o: self.distance_to(o.pos))
        away_x, away_y = self.pos[0] - nearest.pos[0], self.pos[1] - nearest.pos[1]
        norm = math.hypot(away_x, away_y) or 1.0
        goal = self.model.opponent_goal(self.team)
        to_goal_x, to_goal_y = goal[0] - self.pos[0], goal[1] - self.pos[1]
        norm_goal = math.hypot(to_goal_x, to_goal_y) or 1.0
        dx = away_x / norm + 0.5 * to_goal_x / norm_goal
        dy = away_y / norm + 0.5 * to_goal_y / norm_goal
        self.move_toward((self.pos[0] + dx * 5, self.pos[1] + dy * 5))

    # =================================================================
    # Interface commune (section 2.3) -- completee dans les TP suivants
    # =================================================================
    def revise_beliefs(self):
        """TP 2 : revision des croyances (sources, confiance)."""
        pass

    def deliberate(self):
        """TP 2 : desirs et intentions."""
        pass

    def communicate(self):
        """TP 4 : messages entre agents."""
        pass

    def act(self):
        """Alias de execute() pour l'interface commune."""
        pass

    def learn(self, transition):
        """TP 5 : Q-learning."""
        pass

    # =================================================================
    # JOURNAL : perception -> module actif -> decision -> action
    # =================================================================
    def log_tick(self):
        if not self.model.trace:
            return
        b = self.beliefs
        owner = b["owner"]
        mate = b["best_mate"]
        name, arg = self.last_action
        target = f"->{arg.name}" if isinstance(arg, PlayerAgent) else ""

        line = (f"{self.name:<13}| percept: owner={owner.name if owner else None}"
                f" d_ball={b['dist_ball']:5.1f} press={b['pressure']:<6}"
                f" mate={mate.name if mate else None}"
                f" | module={self.last_module}")
        if self.applicable_rules:
            line += f" (applicables={self.applicable_rules})"
        line += f" | action={name}{target}"
        if self.last_result:
            line += f" => {self.last_result}"
        if self.inhibited_decision is not None:
            inh_name, inh_arg = self.inhibited_decision
            inh_target = f"->{inh_arg.name}" if isinstance(inh_arg, PlayerAgent) else ""
            line += f" [inhibe la deliberation : {inh_name}{inh_target}]"
        self.model.log(line)
