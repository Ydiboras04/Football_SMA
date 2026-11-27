"""Planification (TP 3).

3.1  Etat symbolique abstrait : un ensemble de propositions vraies,
     calcule a partir des croyances du porteur.
3.2  Actions decrites par preconditions / effets ajoutes / effets supprimes.
3.3  Recherche en largeur (BFS) dans l'espace des etats vers SHOT_CREATED.
3.5  Plan multi-agent : chaque action a un executant, une date au plus tot
     et eventuellement une ressource (zone de reception) ; detection et
     resolution des conflits de ressources.
"""

from collections import deque
from dataclasses import dataclass, replace

# =============================================================================
# 3.1  Propositions symboliques
# =============================================================================
HAS_BALL = "HAS_BALL"                    # l'equipe a le ballon
BALL_MIDDLE = "BALL_IN_MIDDLE_ZONE"      # ballon a plus de 35 m du but adverse
BALL_ADVANCED = "BALL_IN_ADVANCED_ZONE"  # entre 16.5 et 35 m
BALL_SHOOTING = "BALL_IN_SHOOTING_ZONE"  # a moins de 16.5 m (surface)
NO_PRESSURE = "NO_PRESSURE"              # le porteur n'est pas presse (pression != HIGH)
MATE_FREE = "MATE_FREE"                  # un coequipier visible et libre, 8 m plus haut
SHOT_CREATED = "SHOT_CREATED"            # but symbolique : une occasion de tir

GOAL = frozenset({SHOT_CREATED})

SHOOTING_DIST = 16.5
ADVANCED_DIST = 35.0
ZONE_ORDER = {BALL_MIDDLE: 0, BALL_ADVANCED: 1, BALL_SHOOTING: 2}


def zone_of(goal_dist):
    if goal_dist < SHOOTING_DIST:
        return BALL_SHOOTING
    if goal_dist < ADVANCED_DIST:
        return BALL_ADVANCED
    return BALL_MIDDLE


# =============================================================================
# 3.2  Actions symboliques
# =============================================================================
@dataclass(frozen=True)
class ActionSchema:
    name: str
    preconditions: frozenset
    add_effects: frozenset
    del_effects: frozenset
    kind: str          # "DRIBBLE", "PASS", "RUN", "SHOOT" : action concrete associee


def _action(name, kind, pre, add, delete=()):
    return ActionSchema(name, frozenset(pre), frozenset(add), frozenset(delete), kind)


ACTIONS = [
    # Passe vers un coequipier libre place plus haut : le ballon change de
    # zone et le receveur, libre, n'est pas presse.
    _action("PASS_FORWARD(MIDDLE->ADVANCED)", "PASS",
            pre={HAS_BALL, BALL_MIDDLE, MATE_FREE},
            add={BALL_ADVANCED, NO_PRESSURE},
            delete={BALL_MIDDLE, MATE_FREE}),
    _action("PASS_FORWARD(ADVANCED->SHOOTING)", "PASS",
            pre={HAS_BALL, BALL_ADVANCED, MATE_FREE},
            add={BALL_SHOOTING, NO_PRESSURE},
            delete={BALL_ADVANCED, MATE_FREE}),
    # Conduite de balle : seulement si le porteur n'est pas presse.
    _action("DRIBBLE_FORWARD(MIDDLE->ADVANCED)", "DRIBBLE",
            pre={HAS_BALL, BALL_MIDDLE, NO_PRESSURE},
            add={BALL_ADVANCED},
            delete={BALL_MIDDLE}),
    _action("DRIBBLE_FORWARD(ADVANCED->SHOOTING)", "DRIBBLE",
            pre={HAS_BALL, BALL_ADVANCED, NO_PRESSURE},
            add={BALL_SHOOTING},
            delete={BALL_ADVANCED}),
    # Appel de balle : un coequipier se demarque plus haut.
    _action("RUN_INTO_SPACE", "RUN",
            pre={HAS_BALL},
            add={MATE_FREE}),
    # Tir depuis la surface : l'occasion est creee.
    _action("SHOOT", "SHOOT",
            pre={HAS_BALL, BALL_SHOOTING},
            add={SHOT_CREATED}),
]


def applicable(action, state):
    return action.preconditions <= state


def apply(action, state):
    return frozenset((state - action.del_effects) | action.add_effects)


def bfs_plan(initial_state, goal_test, actions=ACTIONS):
    """Recherche en largeur : renvoie le plan le plus court (en nombre
    d'actions) sous forme de liste d'ActionSchema, ou None."""
    initial_state = frozenset(initial_state)
    frontier = deque([(initial_state, [])])
    visited = {initial_state}
    while frontier:
        state, plan = frontier.popleft()
        if goal_test(state):
            return plan
        for action in actions:
            if applicable(action, state):
                new_state = apply(action, state)
                if new_state not in visited:
                    visited.add(new_state)
                    frontier.append((new_state, plan + [action]))
    return None


def states_along(initial_state, schemas):
    """Etats intermediaires s0, s1, ..., sn produits par le plan."""
    states = [frozenset(initial_state)]
    for action in schemas:
        states.append(apply(action, states[-1]))
    return states


def check_plan(initial_state, schemas, goal=GOAL):
    """Verifie que chaque precondition est vraie au moment de l'action et que
    le but est vrai dans l'etat final. Renvoie (ok, message)."""
    state = frozenset(initial_state)
    for i, action in enumerate(schemas):
        missing = action.preconditions - state
        if missing:
            return False, f"etape {i + 1} {action.name} : precondition(s) {sorted(missing)} fausse(s)"
        state = apply(action, state)
    if not goal <= state:
        return False, f"but {sorted(goal)} faux dans l'etat final"
    return True, "toutes les preconditions sont respectees et le but est atteint"


def short(state):
    """Affichage compact d'un etat symbolique."""
    names = {HAS_BALL: "ballon", BALL_MIDDLE: "zone=milieu",
             BALL_ADVANCED: "zone=avancee", BALL_SHOOTING: "zone=tir",
             NO_PRESSURE: "libre", MATE_FREE: "coequipier_libre",
             SHOT_CREATED: "TIR_CREE"}
    return "{" + ", ".join(names.get(p, p) for p in sorted(state)) + "}"


# =============================================================================
# 3.5  Plan multi-agent
# =============================================================================
@dataclass(frozen=True)
class PlannedAction:
    actor_id: str                 # executant (nom du joueur)
    name: str                     # nom de l'action symbolique
    target_id: str | None = None  # receveur d'une passe
    earliest_tick: int = 0        # date au plus tot
    duration: int = 1             # duree estimee (ticks)
    resource: tuple | None = None # zone de reception reservee
    target_pos: tuple | None = None
    schema: ActionSchema | None = None

    @property
    def kind(self):
        return self.schema.kind if self.schema else self.name

    @property
    def window(self):
        return (self.earliest_tick, self.earliest_tick + self.duration)

    def describe(self):
        text = f"{self.actor_id}:{self.name}"
        if self.target_id:
            text += f"->{self.target_id}"
        if self.resource:
            text += f" zone{self.resource}"
        return text + f" @t{self.earliest_tick}"


@dataclass
class TeamPlan:
    """Plan partage par l'equipe (tableau commun ; au TP 4 il sera
    negocie par messages)."""
    goal: str
    steps: list
    states: list                  # etats symboliques attendus s0..sn
    author: str
    created_tick: int
    step_started: int = 0         # tick de debut de l'etape courante
    index: int = 0

    def current(self):
        return self.steps[self.index] if self.index < len(self.steps) else None

    def finished(self):
        return self.index >= len(self.steps)

    def advance(self, now):
        self.index += 1
        self.step_started = now

    def pending_for(self, name):
        return [s for s in self.steps[self.index:] if s.actor_id == name]

    def describe(self):
        steps = " | ".join(
            ("*" if i == self.index else "") + s.describe()
            for i, s in enumerate(self.steps))
        return f"plan[{self.goal}] {steps}"


# --- Conflits de ressources / d'executants ------------------------------------
def _overlap(w1, w2):
    return w1[0] < w2[1] and w2[0] < w1[1]


def detect_conflicts(steps):
    """Renvoie les paires (i, j, raison) d'actions incompatibles :
    - meme zone de reception occupee par deux joueurs au meme moment ;
    - un meme joueur charge de deux taches simultanees."""
    conflicts = []
    for i in range(len(steps)):
        for j in range(i + 1, len(steps)):
            a, b = steps[i], steps[j]
            if not _overlap(a.window, b.window):
                continue
            if a.resource and a.resource == b.resource and a.actor_id != b.actor_id:
                conflicts.append((i, j, f"zone {a.resource} demandee par "
                                        f"{a.actor_id} et {b.actor_id}"))
            elif (a.actor_id == b.actor_id and a.kind == "RUN" and b.kind == "RUN"):
                conflicts.append((i, j, f"{a.actor_id} a deux courses simultanees"))
    return conflicts


def resolve_conflicts(steps, strategy, free_zone=None):
    """strategy = "schedule"   : decale la seconde action apres la premiere ;
                  "reallocate" : donne une autre zone (free_zone(step)) a la seconde.
    Renvoie (nouvelles_etapes, liste des resolutions)."""
    steps = list(steps)
    resolutions = []
    for _ in range(len(steps) ** 2 + 1):
        conflicts = detect_conflicts(steps)
        if not conflicts:
            break
        i, j, reason = conflicts[0]
        a, b = steps[i], steps[j]
        if strategy == "reallocate" and free_zone is not None and a.resource:
            zone, pos = free_zone(b, {s.resource for s in steps if s.resource})
            steps[j] = replace(b, resource=zone, target_pos=pos)
            resolutions.append(f"{reason} -> reallocation : {b.actor_id} "
                               f"recoit la zone {zone}")
        else:
            shift = a.window[1] - b.earliest_tick
            for k in range(j, len(steps)):
                steps[k] = replace(steps[k], earliest_tick=steps[k].earliest_tick + shift)
            resolutions.append(f"{reason} -> ordonnancement : {b.actor_id} "
                               f"decale de {shift} ticks (t{b.earliest_tick + shift})")
    return steps, resolutions
