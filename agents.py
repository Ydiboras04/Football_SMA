"""Agents joueurs.

TP 0 : automate reactif a 4 etats (CHASE, ATTACK, SUPPORT, DEFEND).
TP 1 : architecture horizontale + couche reactive (subsomption)
       + regles de production pour le porteur.
TP 2 : agent BDI. Les croyances ont une source et une confiance et sont
       revisees ; la couche deliberative genere des desirs, choisit une
       intention, s'y engage et choisit l'action qui la sert.

    perception -> revision des croyances (B)
               -> couche reactive (subsomption) ---- urgence ----+
               -> couche deliberative                            |
                    architecture "REACTIVE" : regles (porteur)   |
                                              + automate TP 0    |
                    architecture "BDI"      : desirs (D)         |
                                              -> intention (I)   |
                                              -> action          |
               -> execution  <---------------------------------- +
"""

import math

import mesa

from beliefs import ENVIRONMENT, PERSONAL, RELATIONAL, SOCIAL, BeliefBase

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

# --- Decision ----------------------------------------------------------------
SHOOT_RANGE = 16.5        # distance au but pour tirer (surface de reparation)
PROGRESS_GOAL = 10.0      # PROGRESS est satisfaite apres 10 m gagnes
OPEN_MARGIN = 15.0        # engagement ouvert : changer si un desir vaut +15


# =============================================================================
# TP 1.3  Regles de production : (nom, condition(beliefs), action(agent), prio)
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
        lambda a: ("PASS", a.belief("best_mate")),
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

        # --- B : croyances (TP 2) ---
        self.beliefs = BeliefBase()
        self.perception_confidence = 1.0   # baisse si la vision est genee
        self.tick_revisions = []           # revisions / rejets du tick courant

        # --- D, I : desirs et intention (TP 2) ---
        self.desires = []                  # [(but, utilite)], tries
        self.intention = None
        self.intention_context = {}        # situation au moment de l'adoption
        self.bdi_events = []               # evenements du tick (journal)
        self.intention_changes = 0         # indicateurs micro
        self.intentions_dropped = 0
        self.intentions_achieved = 0

        # --- Explicabilite ---
        self.last_module = None
        self.last_rule = None
        self.applicable_rules = []
        self.inhibited_decision = None
        self.last_action = None
        self.last_result = None

    # ------------------------------------------------------------ outils
    @property
    def has_ball(self):
        return self.model.get_ball_owner() is self

    def distance_to(self, point):
        return math.hypot(self.pos[0] - point[0], self.pos[1] - point[1])

    def belief(self, key, default=None):
        return self.beliefs.value(key, default)

    def forward(self, x):
        """Abscisse mesuree dans le sens d'attaque de l'equipe."""
        return x * self.model.attack_direction(self.team)

    # =================================================================
    # CYCLE
    # =================================================================
    def step(self):
        self.bdi_events = []

        percept = self.perceive()
        self.revise_beliefs_from(percept)

        # La couche reactive est consultee en premier...
        urgent = self.reactive_layer()
        reactive_module = self.last_module

        # ... la deliberation tourne a chaque tick (le BDI continue de
        # suivre ses intentions), mais sa sortie est inhibee en cas d'urgence.
        decision = self.deliberative_layer()

        if urgent is not None:
            self.inhibited_decision = decision
            self.last_module = reactive_module
            self.applicable_rules = []
            self.execute(urgent)
        else:
            self.inhibited_decision = None
            self.execute(decision)

        self.log_tick()

    # =================================================================
    # PERCEPTION
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
    # 2.2 / 2.3  REVISION DES CROYANCES
    # =================================================================
    def revise_beliefs_from(self, percept):
        """Transforme la perception en croyances typees et les revise."""
        owner = percept["owner"]
        ball_pos = percept["ball_pos"]
        mates, opponents = percept["mates"], percept["opponents"]
        dist_ball = self.distance_to(ball_pos)
        nearest_opp = min(opponents, key=lambda o: self.distance_to(o.pos),
                          default=None)
        nearest_opp_dist = (self.distance_to(nearest_opp.pos)
                            if nearest_opp else math.inf)
        closest_mate = min(mates, key=lambda m: self.distance_to(m.pos),
                           default=None)
        if closest_mate and self.distance_to(closest_mate.pos) >= COLLISION_DIST:
            closest_mate = None

        observed = [
            # --- environnementales ---
            ("ball_position", ball_pos, ENVIRONMENT),
            ("ball_owner", owner, ENVIRONMENT),
            ("dist_ball", dist_ball, ENVIRONMENT),
            ("goal_dist", self.distance_to(self.model.opponent_goal(self.team)),
             ENVIRONMENT),
            ("near_sideline", self.near_sideline(), ENVIRONMENT),
            # --- relationnelles ---
            ("nearest_opponent", nearest_opp, RELATIONAL),
            ("nearest_opp_dist", nearest_opp_dist, RELATIONAL),
            ("pressure", "HIGH" if nearest_opp_dist < PRESSURE_HIGH
             else "MEDIUM" if nearest_opp_dist < PRESSURE_MEDIUM else "LOW",
             RELATIONAL),
            ("best_mate", self.find_best_teammate(mates, opponents), RELATIONAL),
            ("mate_too_close", closest_mate, RELATIONAL),
            ("closest_mate_to_ball",
             all(m.distance_to(ball_pos) >= dist_ball for m in mates), RELATIONAL),
            ("opp_carrier_near_our_goal",
             owner is not None and owner.team != self.team
             and owner.distance_to(self.model.own_goal(self.team)) < DANGER_ZONE,
             RELATIONAL),
            # --- personnelles ---
            ("has_ball", self.has_ball, PERSONAL),
            ("last_action", self.last_action[0] if self.last_action else None,
             PERSONAL),
        ]

        self.tick_revisions = []
        now = self.model.steps
        for key, value, category in observed:
            revision = self.beliefs.revise(
                key, value, "perception", tick=now, now=now,
                confidence=self.perception_confidence, category=category)
            if revision.status in ("revised", "rejected"):
                self.tick_revisions.append(revision)

        # --- sociales / organisationnelles : connues par construction ---
        for key, value in (("role", self.role), ("team", self.team),
                           ("home_zone", self.home)):
            self.beliefs.revise(key, value, "defaut", tick=0, now=now,
                                category=SOCIAL)

    def revise_one_belief(self, key, value, source, confidence, tick=None,
                          category=ENVIRONMENT):
        """Revision d'une seule croyance a partir d'une source quelconque
        (message d'un coequipier au TP 4, scenario de test...)."""
        now = self.model.steps
        revision = self.beliefs.revise(
            key, value, source, tick=now if tick is None else tick, now=now,
            confidence=confidence, category=category)
        if revision.status in ("revised", "rejected"):
            self.tick_revisions.append(revision)
        return revision

    def find_best_teammate(self, mates, opponents):
        """Coequipier visible, libre (aucun adversaire a moins de FREE_MARGIN)
        et le plus avance vers le but adverse."""
        free = [m for m in mates
                if all(o.distance_to(m.pos) >= FREE_MARGIN for o in opponents)]
        if not free:
            return None
        return max(free, key=lambda m: self.forward(m.pos[0]))

    # =================================================================
    # TP 1.2  COUCHE REACTIVE (subsomption)
    #   N4 urgence defensive > N3 proteger le ballon > N2 collision / touche
    # =================================================================
    def reactive_layer(self):
        if self.danger_near_own_goal():
            self.last_module = "REACTIVE:N4_urgence_defensive"
            return ("INTERCEPT", None)

        if self.has_ball and self.is_under_strong_pressure():
            self.last_module = "REACTIVE:N3_protection_ballon"
            return ("PROTECT_BALL", None)

        if not self.has_ball and self.belief("mate_too_close") is not None:
            self.last_module = "REACTIVE:N2_collision"
            return ("AVOID_COLLISION", self.belief("mate_too_close"))

        if self.belief("near_sideline"):
            self.last_module = "REACTIVE:N2_sortie_terrain"
            return ("STAY_IN_BOUNDS", None)

        return None

    def danger_near_own_goal(self):
        return (self.belief("opp_carrier_near_our_goal")
                and self.belief("closest_mate_to_ball"))

    def is_under_strong_pressure(self):
        """Adversaire au contact ET aucune solution de passe (sinon le
        reflexe inhiberait la passe en permanence, voir TP 1)."""
        return (self.belief("nearest_opp_dist") < STRONG_PRESSURE
                and self.belief("best_mate") is None)

    def near_sideline(self):
        y = self.pos[1]
        return y < SIDELINE_MARGIN or y > self.model.space.y_max - SIDELINE_MARGIN

    # =================================================================
    # COUCHE DELIBERATIVE
    # =================================================================
    def deliberative_layer(self):
        self.transition_state()
        if self.model.architecture == "BDI":
            return self.bdi_deliberation()
        if self.belief("has_ball"):
            return self.production_decision()
        return self.automaton_decision()

    # =================================================================
    # TP 2  CYCLE BDI
    # =================================================================
    def bdi_deliberation(self):
        # Nettoyage des intentions (fait ici plutot qu'apres l'execution :
        # les croyances viennent d'etre mises a jour, on juge sur du frais).
        self.drop_achieved_intentions()
        self.drop_impossible_intentions()

        # 2.4 desirs candidats, avec leur utilite
        self.desires = sorted(
            ((goal, self.utility(goal)) for goal in self.generate_desires()),
            key=lambda d: -d[1])

        # 2.5 deliberation : desir -> intention (avec engagement)
        self.deliberate_bdi()

        # Selection de l'action qui sert l'intention
        self.applicable_rules = []
        self.last_module = f"BDI:{self.intention}"
        return self.next_action_for_intention()

    # --- 2.4 Desirs -------------------------------------------------------
    def generate_desires(self):
        if self.belief("has_ball"):
            return ["SCORE", "KEEP_POSSESSION", "PROGRESS"]
        owner = self.belief("ball_owner")
        if owner is None:
            return ["RECOVER_BALL"]
        if owner.team == self.team:
            return ["OFFER_SUPPORT", "CREATE_SPACE"]
        return ["RECOVER_BALL", "PROTECT_GOAL"]

    def utility(self, goal):
        """Interet d'un desir dans la situation courante (croyances)."""
        b = self.belief
        if goal == "SCORE":
            return 100.0 if b("goal_dist") < SHOOT_RANGE else max(0.0, 70.0 - b("goal_dist"))
        if goal == "KEEP_POSSESSION":
            return {"HIGH": 80.0, "MEDIUM": 40.0, "LOW": 10.0}[b("pressure")]
        if goal == "PROGRESS":
            return 50.0
        if goal == "RECOVER_BALL":
            return 75.0 if b("closest_mate_to_ball") else 25.0
        if goal == "PROTECT_GOAL":
            return 65.0 if b("opp_carrier_near_our_goal") else 45.0
        if goal == "OFFER_SUPPORT":
            return 50.0
        if goal == "CREATE_SPACE":
            return 60.0 if b("pressure") == "HIGH" else 30.0
        return 0.0

    # --- 2.5 Deliberation et engagement -----------------------------------
    def deliberate_bdi(self):
        """Engagements (model.commitment) :
        - "blind"  : on garde l'intention jusqu'a la croire accomplie ;
        - "single" : on la garde tant qu'elle reste realisable ;
        - "open"   : on la garde tant qu'elle reste pertinente, c'est-a-dire
                     encore desiree et pas nettement depassee par un autre desir.
        """
        if self.intention is not None and self.keep_intention():
            self.note_rejected_desires("intention maintenue")
            return self.intention

        if not self.desires:
            self.adopt("HOLD_POSITION", "aucun desir")
            return self.intention

        best_goal, best_util = self.desires[0]
        self.adopt(best_goal, f"meilleure utilite ({best_util:.0f})")
        self.note_rejected_desires("moins utiles")
        return self.intention

    def note_rejected_desires(self, why):
        rejected = [f"{g}({u:.0f})" for g, u in self.desires if g != self.intention]
        if rejected:
            self.bdi_events.append(
                f"desirs non retenus : {', '.join(rejected)} ({why})")

    def keep_intention(self):
        mode = self.model.commitment
        if mode == "blind":
            return True
        if mode == "single":
            return True   # deja verifiee realisable par drop_impossible
        # open : encore pertinente ?
        utilities = dict(self.desires)
        if self.intention not in utilities:
            self.bdi_events.append(
                f"intention {self.intention} reconsideree : plus desiree")
            return False
        best_goal, best_util = self.desires[0]
        if best_util > utilities[self.intention] + OPEN_MARGIN:
            self.bdi_events.append(
                f"intention {self.intention} reconsideree : {best_goal} "
                f"bien meilleur ({best_util:.0f} vs "
                f"{utilities[self.intention]:.0f})")
            return False
        return True

    def adopt(self, goal, reason):
        previous = self.intention
        if previous == goal:
            return
        if previous is not None:
            self.intention_changes += 1
        self.intention = goal
        self.intention_context = {
            "tick": self.model.steps,
            "score": self.model.score[self.team],
            "ball_x": self.forward(self.belief("ball_position")[0]),
        }
        self.bdi_events.append(
            f"nouvelle intention : {previous} -> {goal} ({reason})")

    # --- Satisfaction / possibilite des intentions ------------------------
    def intention_achieved(self):
        i, b = self.intention, self.belief
        owner = b("ball_owner")
        my_team_has_ball = owner is not None and owner.team == self.team
        if i == "SCORE":
            return self.model.score[self.team] > self.intention_context["score"]
        if i == "KEEP_POSSESSION":
            return ((my_team_has_ball and owner is not self)
                    or (b("has_ball") and b("pressure") == "LOW"))
        if i == "PROGRESS":
            gained = (self.forward(b("ball_position")[0])
                      - self.intention_context["ball_x"])
            return my_team_has_ball and gained >= PROGRESS_GOAL
        if i == "RECOVER_BALL":
            return my_team_has_ball
        if i in ("OFFER_SUPPORT", "CREATE_SPACE"):
            return b("has_ball")
        if i == "PROTECT_GOAL":
            return owner is None or my_team_has_ball
        if i == "HOLD_POSITION":
            return True
        return False

    def intention_possible(self):
        i, b = self.intention, self.belief
        owner = b("ball_owner")
        if i in ("SCORE", "KEEP_POSSESSION", "PROGRESS"):
            return b("has_ball")
        if i == "RECOVER_BALL":
            return owner is None or owner.team != self.team
        if i in ("OFFER_SUPPORT", "CREATE_SPACE"):
            return owner is not None and owner.team == self.team
        return True

    def drop_achieved_intentions(self):
        if self.intention is not None and self.intention_achieved():
            self.bdi_events.append(f"intention satisfaite : {self.intention}")
            self.intentions_achieved += 1
            self.intention = None

    def drop_impossible_intentions(self):
        if self.intention is None or self.intention_possible():
            return
        if self.model.commitment == "blind":
            self.bdi_events.append(
                f"intention {self.intention} impossible mais conservee "
                f"(engagement aveugle)")
            return
        self.bdi_events.append(f"intention abandonnee (impossible) : "
                               f"{self.intention}")
        self.intentions_dropped += 1
        self.intention = None

    # --- Moyens : quelle action sert l'intention ? ------------------------
    def next_action_for_intention(self):
        i, b = self.intention, self.belief
        has_ball = b("has_ball")
        mate = b("best_mate")
        owner = b("ball_owner")

        if i in ("SCORE", "KEEP_POSSESSION", "PROGRESS") and not has_ball:
            return ("MOVE_TO_BALL", None)   # seulement en engagement aveugle

        if i == "SCORE":
            return ("SHOOT", None) if b("goal_dist") < SHOOT_RANGE else ("DRIBBLE", None)
        if i == "KEEP_POSSESSION":
            return ("PASS", mate) if mate else ("PROTECT_BALL", None)
        if i == "PROGRESS":
            if (mate is not None and b("pressure") != "LOW"
                    and self.forward(mate.pos[0]) > self.forward(self.pos[0]) + 5):
                return ("PASS", mate)
            return ("DRIBBLE", None)
        if i == "RECOVER_BALL":
            return ("MOVE_TO_BALL", None)
        if i == "OFFER_SUPPORT":
            if owner is not None and owner.team == self.team:
                return ("SUPPORT", owner)
            return ("MOVE_TO_BALL", None)
        if i == "CREATE_SPACE":
            return ("CREATE_SPACE", None)
        if i == "PROTECT_GOAL":
            return ("DEFEND_ZONE", None)
        return ("GO_HOME", None)

    # =================================================================
    # TP 1.3  Regles de production (architecture "REACTIVE", porteur)
    # =================================================================
    def production_decision(self):
        facts = self.beliefs.snapshot()
        applicable = [(priority, name, action)
                      for name, condition, action, priority in RULES
                      if condition(facts)]
        self.applicable_rules = [name for _, name, _ in
                                 sorted(applicable, key=lambda r: -r[0])]
        if not applicable:
            self.last_rule = None
            self.last_module = "RULES:aucune"
            return ("HOLD", None)
        _, rule_name, action = max(applicable, key=lambda r: r[0])
        self.last_rule = rule_name
        self.last_module = f"RULES:{rule_name}"
        return action(self)

    # =================================================================
    # TP 0  Automate (architecture "REACTIVE", joueurs sans ballon)
    # =================================================================
    def transition_state(self):
        owner = self.belief("ball_owner")
        if self.belief("has_ball"):
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
            return ("SUPPORT", self.belief("ball_owner"))
        if self.belief("closest_mate_to_ball"):
            return ("PRESS", None)
        return ("DEFEND_ZONE", None)

    # =================================================================
    # EXECUTION
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
        elif name == "CREATE_SPACE":
            self.create_space()
        elif name == "GO_HOME":
            self.move_toward(self.home)
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

    def create_space(self):
        """Se demarquer : avancer en s'eloignant de l'adversaire le plus proche."""
        target_x = self.pos[0] + 8.0 * self.model.attack_direction(self.team)
        target_y = self.pos[1]
        opp = self.belief("nearest_opponent")
        if opp is not None:
            target_y += 6.0 if self.pos[1] >= opp.pos[1] else -6.0
        margin = SIDELINE_MARGIN + 2.0
        target_y = min(max(target_y, margin), self.model.space.y_max - margin)
        self.move_toward((target_x, target_y))

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
    # Interface commune (section 2.3)
    # =================================================================
    def revise_beliefs(self):
        """Voir revise_beliefs_from()."""
        pass

    def deliberate(self):
        """Voir bdi_deliberation()."""
        pass

    def communicate(self):
        """TP 4 : messages entre agents."""
        pass

    def act(self):
        """Voir execute()."""
        pass

    def learn(self, transition):
        """TP 5 : Q-learning."""
        pass

    # =================================================================
    # JOURNAL
    # =================================================================
    def traced(self):
        trace = self.model.trace
        if isinstance(trace, (set, list, tuple)):
            return self.name in trace
        return bool(trace)

    def log_tick(self):
        if not self.traced():
            return
        if self.model.architecture == "BDI":
            self.log_bdi()
        else:
            self.log_reactive()

    @staticmethod
    def _name(value):
        return value.name if isinstance(value, PlayerAgent) else value

    def _action_text(self):
        name, arg = self.last_action
        text = name + (f"->{arg.name}" if isinstance(arg, PlayerAgent) else "")
        if self.last_result:
            text += f" => {self.last_result}"
        if self.inhibited_decision is not None:
            inh, inh_arg = self.inhibited_decision
            inh += f"->{inh_arg.name}" if isinstance(inh_arg, PlayerAgent) else ""
            text += f" [inhibe la deliberation : {inh}]"
        return text

    def log_reactive(self):
        b = self.belief
        line = (f"{self.name:<13}| percept: owner={self._name(b('ball_owner'))}"
                f" d_ball={b('dist_ball'):5.1f} press={b('pressure'):<6}"
                f" mate={self._name(b('best_mate'))}"
                f" | module={self.last_module}")
        if self.applicable_rules:
            line += f" (applicables={self.applicable_rules})"
        self.model.log(line + f" | action={self._action_text()}")

    def log_bdi(self):
        b = self.belief
        desires = " ".join(f"{g}({u:.0f})" for g, u in self.desires)
        self.model.log(
            f"{self.name:<13}| B: owner={self._name(b('ball_owner'))} "
            f"press={b('pressure')} mate={self._name(b('best_mate'))} "
            f"but={b('goal_dist'):.0f}m | D: {desires} | I: {self.intention}"
            f" | module={self.last_module} | action={self._action_text()}")
        for rev in self.tick_revisions:
            old = self._name(rev.old.value) if rev.old else None
            verb = "revisee" if rev.status == "revised" else "REJETEE"
            self.model.log(
                f"{'':13}  croyance {verb} : {rev.key} {old} -> "
                f"{self._name(rev.new.value)} (source {rev.new.source}, "
                f"{rev.reason})")
        for event in self.bdi_events:
            self.model.log(f"{'':13}  {event}")
