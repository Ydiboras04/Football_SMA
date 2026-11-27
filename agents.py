"""Agents joueurs.

TP 0 : automate reactif a 4 etats (CHASE, ATTACK, SUPPORT, DEFEND).
TP 1 : architecture horizontale + couche reactive (subsomption)
       + regles de production pour le porteur.
TP 2 : agent BDI. Les croyances ont une source et une confiance et sont
       revisees ; la couche deliberative genere des desirs, choisit une
       intention, s'y engage et choisit l'action qui la sert.
TP 3 : planification (plan d'equipe vers SHOT_CREATED, replanification).
TP 4 : communication par actes de langage (INFORM, REQUEST, PROPOSE,
       CONFIRM, REFUSE) : les taches du plan sont demandees et acceptees.

    perception -> revision des croyances (B)
               -> messages recus (revision, requetes, confirmations)
               -> couche reactive (subsomption) ---- urgence ----+
               -> couche deliberative                            |
                    architecture "REACTIVE" : regles (porteur)   |
                                              + automate TP 0    |
                    architecture "BDI"      : desirs (D)         |
                                              -> intention (I)   |
                                              -> action          |
               -> execution  <---------------------------------- +
               -> communication (messages envoyes)
"""

import math

import mesa

from beliefs import ENVIRONMENT, PERSONAL, RELATIONAL, SOCIAL, BeliefBase
from messages import (CONFIRM, INFORM, PROPOSE, RECEIVE, REFUSE, REQUEST,
                      RUN_INTO_SPACE, TASK_CANCELLED, Message)
from planning import (GOAL, HAS_BALL, MATE_FREE, NO_PRESSURE, ZONE_ORDER,
                      PlannedAction, TeamPlan, bfs_plan, resolve_conflicts,
                      short, states_along, zone_of)

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

# --- Planification (TP 3) ----------------------------------------------------
AHEAD_MARGIN = 8.0        # MATE_FREE : coequipier libre au moins 8 m plus haut
RUN_DEPTH = 12.0          # une course vise 12 m devant le porteur...
RUN_WIDTH = 10.0          # ... decalee de 10 m sur le cote
RUN_DONE_DIST = 3.0       # course terminee a moins de 3 m de la cible
PLAN_SLACK = 15           # retard tolere sur une etape avant de replanifier
ZONE_CELL = (15.0, 17.0)  # taille d'une "zone de reception" (ressource)

# --- Communication (TP 4) ----------------------------------------------------
SELF_REPORT_CONFIDENCE = 0.95  # un joueur sait mieux que quiconque s'il est marque
CONFIRM_CONFIDENCE = 0.9       # "je suis libre" implicite dans un CONFIRM
PROPOSE_CONFIDENCE = 0.8
REPLY_TIMEOUT = 4              # ticks d'attente d'une reponse a un REQUEST
REFUSAL_MEMORY = 6             # on ne redemande pas a qui a refuse depuis 6 ticks
TASK_TIMEOUT = 40              # une tache acceptee expire apres 40 ticks
PROPOSE_INTERVAL = 10          # au plus une proposition tous les 10 ticks
MAX_RUN_DIST = 30.0            # course refusee si la cible est trop loin


def perceived_free_confidence(distance):
    """Confiance dans "ce coequipier est libre" vue de loin : 1.0 jusqu'a 10 m,
    puis decroissante jusqu'a 0.5 au bord du champ de perception."""
    if distance <= 10.0:
        return 1.0
    return max(0.5, 1.0 - (distance - 10.0) / 30.0)


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
        self.visible_mates = []

        # --- D, I : desirs et intention (TP 2) ---
        self.desires = []                  # [(but, utilite)], tries
        self.intention = None
        self.intention_context = {}        # situation au moment de l'adoption
        self.bdi_events = []               # evenements du tick (journal)
        self.intention_changes = 0         # indicateurs micro
        self.intentions_dropped = 0
        self.intentions_achieved = 0

        # --- Planification (TP 3) ---
        self.pending_plan_step = None      # etape de plan en cours d'execution
        self.replans = 0

        # --- Communication (TP 4) ---
        self.inbox = []                    # messages recus, traites au tick suivant
        self.task = None                   # tache acceptee (engagement envers un coequipier)
        self.refusals = {}                 # {joueur: tick du refus}
        self.proposals = {}                # {joueur: tick de la proposition}
        self.last_reported_free = None     # dernier statut "libre" annonce
        self.last_proposal_tick = -PROPOSE_INTERVAL
        self.messages_sent = 0
        self.messages_received = 0

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

        # TP 4 : informations sociales (messages recus au tick precedent)
        self.process_messages()
        self.derive_teammate_beliefs()

        # La couche reactive est consultee en premier...
        urgent = self.reactive_layer()
        reactive_module = self.last_module

        # ... la deliberation tourne a chaque tick (le BDI continue de
        # suivre ses intentions), mais sa sortie est inhibee en cas d'urgence.
        self.pending_plan_step = None
        decision = self.deliberative_layer()

        if urgent is not None:
            self.inhibited_decision = decision
            self.last_module = reactive_module
            self.applicable_rules = []
            self.execute(urgent)
        else:
            self.inhibited_decision = None
            result = self.execute(decision)
            self.finish_plan_step(self.pending_plan_step, result)

        self.communicate()
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

        # Une croyance par coequipier visible : "est-il libre ?" La confiance
        # baisse avec la distance (on voit mal le marquage d'un joueur lointain),
        # ce qui laisse une chance a ce qu'il nous dit lui-meme (TP 4).
        self.visible_mates = mates
        free_names = {m.name for m in self.free_teammates(mates, opponents)}
        for m in mates:
            confidence = (perceived_free_confidence(self.distance_to(m.pos))
                          * self.perception_confidence)
            revision = self.beliefs.revise(
                f"mate_free:{m.name}", m.name in free_names, "perception",
                tick=now, now=now, confidence=confidence, category=RELATIONAL)
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

    def derive_teammate_beliefs(self):
        """Croyances deduites des croyances "mate_free:X" (perception ou
        messages) : coequipiers libres visibles et meilleur receveur."""
        free = [m for m in self.visible_mates
                if self.belief(f"mate_free:{m.name}", False)]
        best = max(free, key=lambda m: self.forward(m.pos[0]), default=None)
        now = self.model.steps
        for key, value in (("free_mates", frozenset(m.name for m in free)),
                           ("best_mate", best)):
            revision = self.beliefs.revise(key, value, "deduction", tick=now,
                                           now=now, category=RELATIONAL)
            if revision.status in ("revised", "rejected"):
                self.tick_revisions.append(revision)

    @staticmethod
    def free_teammates(mates, opponents):
        """Coequipiers visibles sans adversaire visible a moins de FREE_MARGIN."""
        return [m for m in mates
                if all(o.distance_to(m.pos) >= FREE_MARGIN for o in opponents)]

    def find_best_teammate(self, mates, opponents):
        """Coequipier visible, libre et le plus avance vers le but adverse."""
        free = self.free_teammates(mates, opponents)
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
        if self.model.architecture != "REACTIVE":
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
        # TP 3 : une tache attribuee par le plan d'equipe devient un desir
        # (sauf pour qui poursuit deja CREATE_SHOT : il execute son plan).
        plan_task = (["FOLLOW_PLAN"] if self.assigned_plan_step()
                     and self.intention != "CREATE_SHOT" else [])
        if self.belief("has_ball"):
            desires = ["SCORE", "KEEP_POSSESSION", "PROGRESS"]
            if self.model.planning:
                desires.append("CREATE_SHOT")
            return desires + plan_task
        owner = self.belief("ball_owner")
        if owner is None:
            return ["RECOVER_BALL"]
        if owner.team == self.team:
            return ["OFFER_SUPPORT", "CREATE_SPACE"] + plan_task
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
        if goal == "CREATE_SHOT":
            # hors de la surface : il faut plusieurs actions -> un plan
            return 0.0 if b("goal_dist") < SHOOT_RANGE else 65.0
        if goal == "FOLLOW_PLAN":
            return 90.0
        return 0.0

    # --- 2.5 Deliberation et engagement -----------------------------------
    def deliberate_bdi(self):
        """Engagements (model.commitment) :
        - "blind"  : on garde l'intention jusqu'a la croire accomplie ;
        - "single" : on la garde tant qu'elle reste realisable ;
        - "open"   : on la garde tant qu'elle reste pertinente, c'est-a-dire
                     encore desiree et pas nettement depassee par un autre desir.
        """
        # Une tache recue du plan d'equipe est un evenement nouveau : elle
        # declenche une reconsideration, quel que soit l'engagement.
        utilities = dict(self.desires)
        if "FOLLOW_PLAN" in utilities and self.intention not in (
                "FOLLOW_PLAN", None):
            self.bdi_events.append(
                f"tache recue du plan d'equipe : {self.intention} reconsideree")
            self.adopt("FOLLOW_PLAN", "tache attribuee par le plan")
            self.note_rejected_desires("tache prioritaire")
            return self.intention

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
            "shots": self.model.team_shots[self.team],
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
        if i == "CREATE_SHOT":
            return self.model.team_shots[self.team] > self.intention_context["shots"]
        if i == "FOLLOW_PLAN":
            if self.model.communication and not b("has_ball"):
                return False   # la tache acceptee dure jusqu'a la passe (ou annulation)
            plan = self.model.team_plans.get(self.team)
            return plan is not None and not plan.pending_for(self.name)
        return False

    def intention_possible(self):
        i, b = self.intention, self.belief
        owner = b("ball_owner")
        if i in ("SCORE", "KEEP_POSSESSION", "PROGRESS"):
            return b("has_ball")
        if i == "RECOVER_BALL":
            return owner is None or owner.team != self.team
        if i in ("OFFER_SUPPORT", "CREATE_SPACE", "CREATE_SHOT"):
            return owner is not None and owner.team == self.team
        if i == "FOLLOW_PLAN":
            return (owner is not None and owner.team == self.team
                    and self.assigned_plan_step() is not None)
        return True

    # (avec communication, assigned_plan_step() renvoie la tache acceptee)

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
        if i in ("CREATE_SHOT", "FOLLOW_PLAN"):
            return self.plan_action()
        return ("GO_HOME", None)

    # =================================================================
    # TP 3  PLANIFICATION
    # =================================================================
    def assigned_plan_step(self):
        """Prochaine etape du plan d'equipe confiee a ce joueur, ou None.
        Avec communication (TP 4), un joueur sans ballon ne lit pas le plan
        du porteur : sa seule tache est celle qu'il a acceptee par message."""
        if self.model.communication and not self.belief("has_ball"):
            return self.task
        plan = self.model.team_plans.get(self.team)
        if plan is None:
            return None
        pending = plan.pending_for(self.name)
        return pending[0] if pending else None

    # --- 3.1 Etat symbolique, vu par le porteur ---------------------------
    def symbolic_state(self):
        state = set()
        owner = self.belief("ball_owner")
        if owner is not None and owner.team == self.team:
            state.add(HAS_BALL)
        ball_zone = zone_of(self.ball_goal_dist())
        state.add(ball_zone)
        if self.belief("pressure") != "HIGH":
            state.add(NO_PRESSURE)
        if self.free_mate_ahead(self.belief("best_mate")):
            state.add(MATE_FREE)
        return frozenset(state)

    def free_mate_ahead(self, mate):
        """Coequipier visible, libre et au moins AHEAD_MARGIN plus haut."""
        return (mate is not None
                and mate.name in self.belief("free_mates")
                and self.forward(mate.pos[0]) >= self.forward(self.pos[0]) + AHEAD_MARGIN)

    def goal_dist_of(self, pos):
        goal = self.model.opponent_goal(self.team)
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

    def ball_goal_dist(self):
        return self.goal_dist_of(self.belief("ball_position"))

    # --- Execution du plan (boucle fermee ou ouverte) ---------------------
    def plan_action(self):
        if self.model.communication and not self.belief("has_ball"):
            return self.task_action()
        plan = self.model.team_plans.get(self.team)
        if self.belief("has_ball"):
            plan = self.maintain_plan(plan)
            if plan is None:
                return ("DRIBBLE", None)
        elif plan is not None:
            self.advance_plan(plan, None)
            plan = self.model.team_plans.get(self.team)

        step = plan.current() if plan else None
        if step is not None and step.actor_id == self.name:
            return self.concrete_step(step)
        if self.belief("has_ball"):
            # attendre qu'un coequipier termine sa tache (synchronisation)
            return ("PROTECT_BALL", None) if self.belief("pressure") == "HIGH" \
                else ("HOLD", None)
        owner = self.belief("ball_owner")
        if owner is not None and owner.team == self.team:
            return ("SUPPORT", owner)
        return ("MOVE_TO_BALL", None)

    def maintain_plan(self, plan):
        """Le porteur fait avancer le plan, verifie qu'il est encore valide
        et replanifie sinon (boucle fermee)."""
        state = self.symbolic_state()
        if plan is not None:
            self.advance_plan(plan, state)
            plan = self.model.team_plans.get(self.team)

        problem = self.plan_problem(plan, state)
        if problem is None:
            self.request_current_task(plan)
            return plan
        if plan is not None:
            self.replans += 1
            self.model.total_replans += 1
            self.bdi_events.append(f"plan invalide : {problem} -> replanification")
            self.cancel_requested_tasks(plan)
        plan = self.make_plan(state)
        if plan is not None:
            self.request_current_task(plan)
        return plan

    def plan_problem(self, plan, state):
        if plan is None:
            return "aucun plan"
        step = plan.current()
        if step is None:
            return "plan termine"
        if step.kind in ("DRIBBLE", "PASS", "SHOOT") and step.actor_id != self.name:
            return f"le porteur devait etre {step.actor_id}"
        if self.model.communication and step.kind == "RUN":
            status, since = plan.status.get(plan.index, (None, None))
            if status == REFUSE:
                return f"{step.actor_id} a refuse la course"
            if status == REQUEST and self.model.steps - since > REPLY_TIMEOUT:
                return f"{step.actor_id} n'a pas repondu"
            if status == CONFIRM and self.reported_marked(step.actor_id):
                return f"{step.actor_id} signale qu'il est marque"
        if self.model.plan_mode == "open":
            return None   # boucle ouverte : on execute la sequence prevue
        missing = step.schema.preconditions - state
        if missing:
            return f"{step.name} : precondition(s) {sorted(missing)} fausse(s)"
        if step.kind == "PASS" and step.target_id not in self.belief("free_mates"):
            return f"le receveur {step.target_id} n'est plus libre"
        if self.model.steps - plan.step_started > step.duration + PLAN_SLACK:
            return f"{step.name} en retard"
        return None

    def advance_plan(self, plan, state):
        """Passe les etapes deja realisees (conduite ou course terminee)."""
        while not plan.finished():
            step = plan.current()
            if step.kind == "DRIBBLE" and state is not None:
                zones = set(ZONE_ORDER)
                target_zone = next(iter(step.schema.add_effects & zones))
                ball_zone = next(iter(state & zones))
                done = ZONE_ORDER[ball_zone] >= ZONE_ORDER[target_zone]
            elif step.kind == "RUN":
                # terminee si le coureur est arrive, ou si le porteur le voit
                # deja libre et place plus haut que lui
                runner = self.model.get_player(step.actor_id)
                done = (runner.distance_to(step.target_pos) < RUN_DONE_DIST
                        or (state is not None and self.free_mate_ahead(runner)))
                if self.model.communication:
                    # 4.3 : pas de passe tant que la course n'est pas confirmee
                    done = done and plan.status.get(plan.index, (None,))[0] == CONFIRM
            else:
                done = False
            if not done:
                break
            self.bdi_events.append(f"etape terminee : {step.describe()}")
            plan.advance(self.model.steps)
        if plan.finished():
            self.model.team_plans.pop(self.team, None)

    def concrete_step(self, step):
        if step.kind == "DRIBBLE":
            return ("DRIBBLE", None)
        if step.kind == "RUN":
            return ("RUN_TO", step.target_pos)
        self.pending_plan_step = step
        if step.kind == "PASS":
            return ("PASS", self.model.get_player(step.target_id))
        return ("SHOOT", None)

    def finish_plan_step(self, step, result):
        """Apres une passe ou un tir prevus par le plan."""
        if step is None:
            return
        plan = self.model.team_plans.get(self.team)
        if plan is not None and plan.current() is step and result in (
                "PASS_OK", "GOAL", "SHOT_MISSED"):
            self.bdi_events.append(f"etape terminee : {step.describe()} => {result}")
            plan.advance(self.model.steps)
            if plan.finished():
                self.model.team_plans.pop(self.team, None)

    # --- 3.3 Planifier, puis 3.5 allouer les actions aux joueurs ----------
    def make_plan(self, state):
        schemas = bfs_plan(state, lambda s: GOAL <= s)
        if not schemas:
            self.bdi_events.append(f"aucun plan trouve depuis {short(state)}")
            return None
        steps = self.allocate(schemas)
        if steps is None:
            self.bdi_events.append("plan trouve mais aucun coequipier disponible")
            return None
        steps, resolutions = resolve_conflicts(steps, "reallocate",
                                               self.free_reception_zone)
        plan = TeamPlan("CREATE_SHOT", steps, states_along(state, schemas),
                        author=self.name, created_tick=self.model.steps,
                        step_started=self.model.steps)
        self.model.team_plans[self.team] = plan
        self.model.plans_created += 1

        chain = f"s0 {short(plan.states[0])}"
        for k, sc in enumerate(schemas, start=1):
            chain += f" --{sc.name}--> s{k} {short(plan.states[k])}"
        self.bdi_events.append(f"plan cree (BFS, {len(schemas)} actions) : {chain}")
        self.bdi_events.append(
            "allocation : " + " | ".join(s.describe() for s in steps))
        for r in resolutions:
            self.bdi_events.append(f"conflit resolu : {r}")
        return plan

    def allocate(self, schemas):
        """Associe chaque action symbolique a un executant, une date au plus
        tot et une zone de reception (pour les courses)."""
        now = self.model.steps
        t = now
        carrier = self
        runner = None
        busy = self.unavailable_mates()
        steps = []
        for sc in schemas:
            if sc.kind == "RUN":
                choice = self.choose_runner(carrier, busy)
                if choice is None:
                    return None
                runner, target = choice
                busy.add(runner.name)
                duration = math.ceil(runner.distance_to(target) / SPEED)
                steps.append(PlannedAction(
                    runner.name, sc.name, earliest_tick=t, duration=duration,
                    resource=self.zone_id(target), target_pos=target, schema=sc))
                t += duration
            elif sc.kind == "PASS":
                receiver = runner if runner is not None else self.belief("best_mate")
                if receiver is None:
                    return None
                steps.append(PlannedAction(carrier.name, sc.name,
                                           target_id=receiver.name,
                                           earliest_tick=t, schema=sc))
                carrier, runner = receiver, None
                t += 1
            elif sc.kind == "DRIBBLE":
                duration = math.ceil(18.5 / SPEED_WITH_BALL)
                steps.append(PlannedAction(carrier.name, sc.name, earliest_tick=t,
                                           duration=duration, schema=sc))
                t += duration
            else:
                steps.append(PlannedAction(carrier.name, sc.name, earliest_tick=t,
                                           schema=sc))
                t += 1
        return steps

    def choose_runner(self, carrier, busy):
        """Allocation selon les savoir-faire : on prefere un attaquant, puis le
        coequipier le plus proche de la zone d'appel. Le planificateur connait
        la position de ses coequipiers (connaissance d'equipe, simplification)."""
        mates = [m for m in self.model.players(self.team)
                 if m is not carrier and m.name not in busy]
        if not mates:
            return None
        # La course reste dans le champ de perception du porteur (12 m devant,
        # 10 m sur le cote) pour que MATE_FREE puisse devenir vrai.
        direction = self.model.attack_direction(self.team)
        x = carrier.pos[0] + RUN_DEPTH * direction
        x = min(max(x, 3.0), self.model.space.x_max - 3.0)

        def target_for(mate):
            side = RUN_WIDTH if mate.pos[1] >= carrier.pos[1] else -RUN_WIDTH
            y = min(max(carrier.pos[1] + side, 3.0), self.model.space.y_max - 3.0)
            return (x, y)

        # TP 4 : un coequipier qui s'est propose recemment passe en premier.
        now = self.model.steps
        runner = min(mates, key=lambda m: (
            now - self.proposals.get(m.name, -10**6) > PROPOSE_INTERVAL,
            m.role != "ATT",
            m.distance_to(target_for(m))))
        return runner, target_for(runner)

    def zone_id(self, pos):
        return (int(pos[0] // ZONE_CELL[0]), int(pos[1] // ZONE_CELL[1]))

    def free_reception_zone(self, step, taken):
        """Reallocation : une autre zone de reception, libre, pres de la cible."""
        x, y = step.target_pos
        for dx, dy in ((0, ZONE_CELL[1]), (0, -ZONE_CELL[1]),
                       (ZONE_CELL[0], 0), (-ZONE_CELL[0], 0)):
            pos = (min(max(x + dx, 3.0), self.model.space.x_max - 3.0),
                   min(max(y + dy, 3.0), self.model.space.y_max - 3.0))
            zone = self.zone_id(pos)
            if zone not in taken:
                return zone, pos
        return step.resource, step.target_pos

    # =================================================================
    # TP 4  COMMUNICATION ET COORDINATION
    # =================================================================
    def send(self, receiver, performative, content, **payload):
        msg = Message(self.name, receiver, performative, content, payload,
                      tick=self.model.steps)
        self.model.bus.send(msg)
        self.messages_sent += 1
        self.bdi_events.append(f"-> envoye : {msg.describe()}")
        return msg

    # --- Cote porteur : demander, suivre, annuler -------------------------
    def request_current_task(self, plan):
        """4.3 etape 2 : la course n'est pas imposee, elle est DEMANDEE."""
        if not self.model.communication or plan is None:
            return
        step = plan.current()
        if step is None or step.kind != "RUN" or plan.index in plan.status:
            return
        plan.status[plan.index] = (REQUEST, self.model.steps)
        self.send(step.actor_id, REQUEST, RUN_INTO_SPACE,
                  target=step.target_pos, plan_id=plan.plan_id)

    def cancel_requested_tasks(self, plan):
        if not self.model.communication:
            return
        for index, (status, _) in plan.status.items():
            if status in (REQUEST, CONFIRM):
                self.send(plan.steps[index].actor_id, INFORM, TASK_CANCELLED,
                          plan_id=plan.plan_id)

    def unavailable_mates(self):
        """Coequipiers a ne pas solliciter : refus recent ou marquage signale."""
        if not self.model.communication:
            return set()
        now = self.model.steps
        names = {n for n, t in self.refusals.items() if now - t <= REFUSAL_MEMORY}
        names |= {m.name for m in self.model.players(self.team)
                  if self.reported_marked(m.name)}
        return names

    def reported_marked(self, name):
        """Le coequipier nous a-t-il dit lui-meme qu'il etait marque ?"""
        belief = self.beliefs.get(f"mate_free:{name}")
        return (belief is not None and belief.value is False
                and belief.source.startswith("agent:"))

    # --- Reception des messages (4.2) -------------------------------------
    def process_messages(self):
        inbox, self.inbox = self.inbox, []
        for msg in inbox:
            self.messages_received += 1
            self.bdi_events.append(f"<- recu : {msg.describe()}")
            handler = {INFORM: self.on_inform, REQUEST: self.on_request,
                       PROPOSE: self.on_propose, CONFIRM: self.on_confirm,
                       REFUSE: self.on_refuse}[msg.performative]
            handler(msg)

    def on_inform(self, msg):
        if msg.content == TASK_CANCELLED:
            if self.task and self.task["plan_id"] == msg.payload["plan_id"]:
                self.bdi_events.append("tache annulee par le porteur")
                self.task = None
            return
        # Partage d'une croyance : elle passe par la revision (source, confiance)
        self.revise_one_belief(
            msg.content, msg.payload.get("value"), source=f"agent:{msg.sender_id}",
            confidence=msg.payload.get("confidence", 0.7), tick=msg.tick,
            category=RELATIONAL)

    def on_request(self, msg):
        ok, reason = self.can_do(msg.content, msg.payload)
        if not ok:
            self.send(msg.sender_id, REFUSE, msg.content, reason=reason,
                      plan_id=msg.payload["plan_id"])
            return
        self.task = {"plan_id": msg.payload["plan_id"], "kind": RUN_INTO_SPACE,
                     "target": msg.payload["target"], "carrier": msg.sender_id,
                     "since": self.model.steps}
        self.last_reported_free = True
        self.send(msg.sender_id, CONFIRM, msg.content,
                  plan_id=msg.payload["plan_id"], value=True,
                  confidence=CONFIRM_CONFIDENCE)
        self.bdi_events.append(
            f"engagement : course vers ({msg.payload['target'][0]:.0f},"
            f"{msg.payload['target'][1]:.0f}) pour {msg.sender_id}")

    def can_do(self, content, payload):
        """4.3 etape 3 : le coequipier verifie SES croyances avant d'accepter."""
        if content != RUN_INTO_SPACE:
            return False, "requete inconnue"
        owner = self.belief("ball_owner")
        if owner is None or owner.team != self.team or self.belief("has_ball"):
            return False, "situation de jeu changee"
        if self.task is not None:
            return False, "deja engage dans une autre tache"
        if self.belief("nearest_opp_dist") < FREE_MARGIN:
            return False, "je suis marque"
        if self.distance_to(payload["target"]) > MAX_RUN_DIST:
            return False, "zone trop loin"
        return True, "ok"

    def on_propose(self, msg):
        self.proposals[msg.sender_id] = msg.tick
        self.revise_one_belief(f"mate_free:{msg.sender_id}", True,
                               source=f"agent:{msg.sender_id}",
                               confidence=PROPOSE_CONFIDENCE, tick=msg.tick,
                               category=RELATIONAL)

    def on_confirm(self, msg):
        plan = self.model.team_plans.get(self.team)
        if plan is None or plan.plan_id != msg.payload["plan_id"]:
            return
        for index, step in enumerate(plan.steps):
            if step.actor_id == msg.sender_id and plan.status.get(index, (None,))[0] == REQUEST:
                plan.status[index] = (CONFIRM, self.model.steps)
                self.bdi_events.append(
                    f"engagement confirme par {msg.sender_id} : {step.describe()}")
        # "je suis libre" est implicite dans l'acceptation
        self.revise_one_belief(f"mate_free:{msg.sender_id}", True,
                               source=f"agent:{msg.sender_id}",
                               confidence=msg.payload.get("confidence", CONFIRM_CONFIDENCE),
                               tick=msg.tick, category=RELATIONAL)

    def on_refuse(self, msg):
        self.refusals[msg.sender_id] = msg.tick
        plan = self.model.team_plans.get(self.team)
        if plan is None or plan.plan_id != msg.payload.get("plan_id"):
            return
        for index, step in enumerate(plan.steps):
            if step.actor_id == msg.sender_id and plan.status.get(index, (None,))[0] == REQUEST:
                plan.status[index] = (REFUSE, self.model.steps)

    # --- Cote coureur : executer la tache acceptee ------------------------
    def task_action(self):
        if self.task is None:
            owner = self.belief("ball_owner")
            if owner is not None and owner.team == self.team:
                return ("SUPPORT", owner)
            return ("MOVE_TO_BALL", None)
        target = self.task["target"]
        if self.distance_to(target) > RUN_DONE_DIST:
            return ("RUN_TO", target)
        return ("HOLD", None)   # arrive : attendre la passe

    # --- Envoi spontane (fin de cycle) -------------------------------------
    def communicate(self):
        """Echange des informations avec les autres agents."""
        if not self.model.communication:
            return
        self.update_task()
        owner = self.belief("ball_owner")

        # 4.3 etape 5 : le coureur engage signale tout changement de marquage
        if self.task is not None:
            free_now = self.belief("nearest_opp_dist") >= FREE_MARGIN
            if free_now != self.last_reported_free:
                self.last_reported_free = free_now
                self.send(self.task["carrier"], INFORM, f"mate_free:{self.name}",
                          value=free_now, confidence=SELF_REPORT_CONFIDENCE)
            return

        # PROPOSE : un joueur libre et devant le porteur se propose
        if (owner is not None and owner.team == self.team and owner is not self
                and self.belief("pressure") == "LOW"
                and self.forward(self.pos[0]) > self.forward(owner.pos[0]) + AHEAD_MARGIN
                and self.model.steps - self.last_proposal_tick >= PROPOSE_INTERVAL):
            self.last_proposal_tick = self.model.steps
            self.send(owner.name, PROPOSE, RECEIVE, zone=self.zone_id(self.pos))

    def update_task(self):
        """Une tache acceptee s'eteint si l'equipe perd le ballon, si je le
        recois (tache accomplie) ou si elle dure trop longtemps."""
        if self.task is None:
            return
        owner = self.model.get_ball_owner()
        if owner is self:
            self.bdi_events.append("tache accomplie : ballon recu")
            self.task = None
        elif owner is None or owner.team != self.team:
            self.task = None
        elif self.model.steps - self.task["since"] > TASK_TIMEOUT:
            self.bdi_events.append("tache expiree")
            self.task = None

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
        elif name == "RUN_TO":
            self.move_toward(arg)
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
        if self.model.architecture != "REACTIVE":
            self.log_bdi()
        else:
            self.log_reactive()

    @staticmethod
    def _name(value):
        return value.name if isinstance(value, PlayerAgent) else value

    def _action_text(self):
        name, arg = self.last_action
        text = name + (f"->{arg.name}" if isinstance(arg, PlayerAgent) else "")
        if name == "RUN_TO":
            text += f"({arg[0]:.0f},{arg[1]:.0f})"
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
        plan = self.model.team_plans.get(self.team)
        if plan is not None and self.intention in ("CREATE_SHOT", "FOLLOW_PLAN"):
            self.model.log(f"{'':13}  {plan.describe()}")
