"""Scenarios controles pour valider les TP.

Chaque scenario place les joueurs a la main, donne le ballon a un joueur,
puis execute UN seul cycle du joueur observe avec le journal active.

    python experiments.py         # tous les scenarios
    python experiments.py tp1     # seulement ceux du TP 1
    python experiments.py tp2     # seulement ceux du TP 2
"""

import sys

from model import LEFT_TEAM, RIGHT_TEAM, FootballModel

RM = LEFT_TEAM    # attaque vers la droite (x croissant)
BA = RIGHT_TEAM   # attaque vers la gauche


def set_scene(positions, carrier=None, ball_pos=None, seed=0,
              architecture="BDI", commitment="single"):
    """positions : {nom_joueur: (x, y)} ; carrier : nom du porteur,
    ou None avec ball_pos pour un ballon libre."""
    model = FootballModel(seed=seed, verbose=True, architecture=architecture,
                          commitment=commitment)
    for name, pos in positions.items():
        model.space.move_agent(model.get_player(name), pos)
    if carrier is not None:
        model.give_ball(model.get_player(carrier))
    else:
        model.drop_ball(ball_pos)
    return model


def run_one_cycle(model, name):
    model.trace = True
    model.get_player(name).step()


def title(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


# -----------------------------------------------------------------------------
# TP 1 - Test 1 : la couche reactive inhibe la couche deliberative
# -----------------------------------------------------------------------------
def tp1_reactive_inhibits():
    title("TP1 / Test 1a - Niveau 3 : porteur au contact, aucun coequipier visible\n"
          "-> PROTECT_BALL inhibe la decision deliberative (DRIBBLE)")
    model = set_scene({
        f"{RM}3": (60.0, 34.0),   # porteur
        f"{BA}3": (61.0, 34.0),   # adversaire a 1 m
        f"{RM}1": (20.0, 34.0),   # coequipiers hors du champ de perception
        f"{RM}2": (30.0, 10.0),
        f"{BA}1": (85.0, 34.0),
        f"{BA}2": (90.0, 60.0),
    }, carrier=f"{RM}3", architecture="REACTIVE")
    run_one_cycle(model, f"{RM}3")

    title("TP1 / Test 1b - Niveau 4 : porteur adverse a 15 m de notre but\n"
          "-> INTERCEPT inhibe la decision de l'automate (PRESS)")
    model = set_scene({
        f"{BA}3": (15.0, 34.0),   # porteur adverse devant le but de RM
        f"{RM}1": (19.0, 36.0),   # defenseur le plus proche
        f"{RM}2": (35.0, 20.0),
        f"{RM}3": (45.0, 48.0),
        f"{BA}1": (85.0, 34.0),
        f"{BA}2": (40.0, 50.0),
    }, carrier=f"{BA}3", architecture="REACTIVE")
    run_one_cycle(model, f"{RM}1")


# -----------------------------------------------------------------------------
# TP 1 - Test 2 : conflit entre regles de production
# -----------------------------------------------------------------------------
def tp1_rule_conflict():
    title("TP1 / Test 2a - Conflit a 2 regles : pression HIGH + coequipier libre\n"
          "-> pass_if_pressed (80) l'emporte sur dribble_forward (20)")
    model = set_scene({
        f"{RM}3": (60.0, 34.0),   # porteur
        f"{BA}3": (62.5, 34.0),   # adversaire a 2.5 m : HIGH mais pas au contact
        f"{RM}2": (70.0, 20.0),   # coequipier libre et devant
        f"{RM}1": (20.0, 34.0),
        f"{BA}1": (85.0, 34.0),
        f"{BA}2": (90.0, 60.0),
    }, carrier=f"{RM}3", architecture="REACTIVE")
    run_one_cycle(model, f"{RM}3")

    title("TP1 / Test 2b - Conflit a 3 regles : a 13 m du but, presse, coequipier libre\n"
          "-> shoot_if_close (100) l'emporte sur pass_if_pressed (80) et dribble (20)")
    model = set_scene({
        f"{RM}3": (92.0, 34.0),   # porteur a 13 m du but adverse
        f"{BA}1": (94.5, 34.0),   # adversaire a 2.5 m
        f"{RM}2": (85.0, 20.0),   # coequipier libre
        f"{RM}1": (40.0, 34.0),
        f"{BA}2": (60.0, 60.0),
        f"{BA}3": (60.0, 10.0),
    }, carrier=f"{RM}3", architecture="REACTIVE")
    run_one_cycle(model, f"{RM}3")


# =============================================================================
# TP 2 - BDI
# =============================================================================

# Une scene "calme" : le porteur RM3 au milieu, personne autour.
CALM_SCENE = {
    f"{RM}3": (50.0, 34.0),
    f"{RM}1": (20.0, 34.0),
    f"{RM}2": (30.0, 10.0),
    f"{BA}1": (85.0, 34.0),
    f"{BA}2": (90.0, 60.0),
    f"{BA}3": (80.0, 10.0),
}


def tp2_bdi_five_ticks():
    title("TP2 / Test 1 - Afficher B, D et I d'un meme joueur pendant 5 ticks\n"
          "(match normal, graine 42, on observe REAL_MADRID3 des le tick 15)")
    model = FootballModel(seed=42, verbose=True)
    model.trace = False
    model.verbose = False
    for _ in range(14):
        model.step()
    model.verbose = True
    model.trace = {f"{RM}3"}
    for _ in range(5):
        model.step()


def show_revision(revision, agent):
    belief = agent.beliefs.get(revision.key)
    print(f"   -> {revision.status.upper()} : {revision.reason}")
    print(f"   -> croyance finale : {revision.key} = {belief.value} "
          f"(source {belief.source}, tick {belief.tick}, "
          f"confiance {belief.confidence})")


def tp2_belief_revision():
    title("TP2 / Test 2 - Revision des croyances : information contradictoire\n"
          "(travail guide 2.3 et scenario B)")
    agent_name = f"{RM}2"

    print("\n2a) RM2 VOIT le ballon en (40, 20) au tick 10 (confiance 1.0).\n"
          "    Il recoit ensuite un message ANCIEN (tick 5, confiance 0.7) qui\n"
          "    affirme que le ballon est en (35, 20).")
    model = set_scene(CALM_SCENE, ball_pos=(40.0, 20.0))
    model.steps = 10
    agent = model.get_player(agent_name)
    agent.revise_beliefs_from(agent.perceive())
    rev = agent.revise_one_belief("ball_position", (35.0, 20.0),
                                  source=f"message:{RM}3", confidence=0.7, tick=5)
    show_revision(rev, agent)
    print("   Explication : le message est moins fiable au depart (0.7 < 1.0)\n"
          "   ET il a 5 ticks d'age (0.7 x 0.95^5 = 0.54). La perception\n"
          "   directe est conservee.")

    print("\n2b) Meme situation, confiances INVERSEES : la vision de RM2 est genee\n"
          "    (perception a 0.6) et le message est recent et fiable (tick 9, 0.95).")
    model = set_scene(CALM_SCENE, ball_pos=(40.0, 20.0))
    model.steps = 10
    agent = model.get_player(agent_name)
    agent.perception_confidence = 0.6
    agent.revise_beliefs_from(agent.perceive())
    rev = agent.revise_one_belief("ball_position", (35.0, 20.0),
                                  source=f"message:{RM}3", confidence=0.95, tick=9)
    show_revision(rev, agent)
    print("   Explication : 0.95 x 0.95^1 = 0.90 > 0.60 : le message l'emporte.")

    print("\n2c) Confiance egale (0.9 et 0.9, meme tick) : la regle de recence\n"
          "    s'applique, l'information arrivee en dernier est retenue.")
    model = set_scene(CALM_SCENE, ball_pos=(40.0, 20.0))
    model.steps = 10
    agent = model.get_player(agent_name)
    agent.perception_confidence = 0.9
    agent.revise_beliefs_from(agent.perceive())
    rev = agent.revise_one_belief("ball_position", (35.0, 20.0),
                                  source=f"message:{RM}3", confidence=0.9, tick=10)
    show_revision(rev, agent)

    print(f"\nScenario B) RM2 percoit {RM}3 LIBRE ; un message ancien dit "
          f"{RM}3 MARQUE.")
    for perception_conf, msg_conf, msg_tick in ((1.0, 0.7, 6), (0.5, 0.9, 9)):
        model = set_scene(CALM_SCENE, ball_pos=(40.0, 20.0))
        model.steps = 10
        agent = model.get_player(agent_name)
        key = f"{RM}3_free"
        agent.revise_one_belief(key, True, "perception",
                                confidence=perception_conf, category="relationnelle")
        rev = agent.revise_one_belief(key, False, f"message:{BA}1",
                                      confidence=msg_conf, tick=msg_tick,
                                      category="relationnelle")
        print(f"   perception {perception_conf} vs message {msg_conf} "
              f"(tick {msg_tick}) :")
        show_revision(rev, agent)


def tp2_desire_not_retained():
    title("TP2 / Test 3 - Un desir n'est pas retenu comme intention\n"
          "(porteur presse, coequipier libre : 3 desirs, un seul devient intention)")
    model = set_scene({
        f"{RM}3": (60.0, 34.0),
        f"{BA}3": (62.5, 34.0),   # adversaire a 2.5 m : pression HIGH
        f"{RM}2": (70.0, 20.0),   # coequipier libre
        f"{RM}1": (20.0, 34.0),
        f"{BA}1": (85.0, 34.0),
        f"{BA}2": (90.0, 60.0),
    }, carrier=f"{RM}3")
    run_one_cycle(model, f"{RM}3")


def tp2_intention_dropped():
    for commitment in ("single", "blind"):
        title(f"TP2 / Test 4 - Intention abandonnee car impossible "
              f"(engagement '{commitment}')\n"
              f"Tick 1 : RM3 seul avec le ballon -> intention PROGRESS.\n"
              f"Tick 2 : BA3 lui a pris le ballon -> PROGRESS devient impossible.")
        model = set_scene(CALM_SCENE, carrier=f"{RM}3", commitment=commitment)
        model.trace = {f"{RM}3"}
        rm3 = model.get_player(f"{RM}3")
        rm3.step()
        ba3 = model.get_player(f"{BA}3")
        model.space.move_agent(ba3, (rm3.pos[0] + 1.0, rm3.pos[1]))
        model.give_ball(ba3)
        print("   ... BA3 prend le ballon a RM3 ...")
        model.steps += 1
        rm3.step()


def tp2_commitment_comparison():
    for commitment in ("single", "open"):
        title(f"TP2 / Test 5 - Engagement '{commitment}' quand la situation change\n"
              f"Tick 1 : RM3 seul -> PROGRESS (50).\n"
              f"Tick 2 : un adversaire arrive a 2.5 m -> KEEP_POSSESSION vaut 80.")
        model = set_scene(CALM_SCENE, carrier=f"{RM}3", commitment=commitment)
        model.trace = {f"{RM}3"}
        rm3 = model.get_player(f"{RM}3")
        rm3.step()
        model.space.move_agent(model.get_player(f"{BA}3"),
                               (rm3.pos[0] + 2.5, rm3.pos[1]))
        print("   ... BA3 arrive au pressing ...")
        model.steps += 1
        rm3.step()


# =============================================================================
# TP 3 - Planification
# =============================================================================
from planning import (ACTIONS, BALL_MIDDLE, GOAL, HAS_BALL, MATE_FREE,  # noqa: E402
                      NO_PRESSURE, PlannedAction, bfs_plan, check_plan,
                      detect_conflicts, resolve_conflicts, short, states_along)


def print_plan(initial_state, schemas):
    states = states_along(initial_state, schemas)
    print(f"   s0 = {short(states[0])}")
    for k, action in enumerate(schemas, start=1):
        print(f"      --{action.name}-->")
        print(f"   s{k} = {short(states[k])}")
    ok, message = check_plan(initial_state, schemas)
    print(f"   verification : {'OK' if ok else 'ECHEC'} - {message}")


def tp3_plan_generation():
    title("TP3 / Test 1 - Plans symboliques vers SHOT_CREATED (BFS)\n"
          "Le plan respecte toutes les preconditions et le but est vrai a la fin")
    cases = [
        ("Ballon au milieu, porteur presse, coequipier libre devant",
         {HAS_BALL, BALL_MIDDLE, MATE_FREE}),
        ("Ballon au milieu, porteur libre, personne de demarque",
         {HAS_BALL, BALL_MIDDLE, NO_PRESSURE}),
        ("Ballon au milieu, porteur presse, personne de demarque",
         {HAS_BALL, BALL_MIDDLE}),
    ]
    for label, state in cases:
        print(f"\n{label} :")
        print_plan(state, bfs_plan(state, lambda s: GOAL <= s))

    print("\nContre-exemple : plan ecrit a la main [SHOOT] depuis le milieu :")
    shoot = next(a for a in ACTIONS if a.name == "SHOOT")
    print(f"   verification : {check_plan({HAS_BALL, BALL_MIDDLE}, [shoot])[1]}")


# Scene du scenario C : RM1 porte le ballon au milieu, RM2 et RM3 sont libres
# plus haut (zone avancee) ; RM2 est le plus avance, donc le receveur choisi.
SCENE_C = {
    f"{RM}1": (52.0, 34.0),
    f"{RM}2": (73.0, 26.0),
    f"{RM}3": (71.0, 42.0),
    f"{BA}3": (50.0, 30.0),   # a 4.5 m, hors des lignes de passe : MEDIUM
    f"{BA}1": (90.0, 34.0),
    f"{BA}2": (90.0, 60.0),
}


def think(agent):
    """Un cycle de decision SANS execution : on montre le plan choisi."""
    agent.bdi_events = []
    agent.revise_beliefs_from(agent.perceive())
    agent.process_messages()
    agent.derive_teammate_beliefs()
    agent.last_action = agent.deliberative_layer()
    agent.last_result = None
    agent.inhibited_decision = None
    agent.log_tick()


def tp3_replan_scenario_c():
    for mode in ("closed", "open"):
        title(f"TP3 / Test 2 - Scenario C, boucle {'FERMEE' if mode == 'closed' else 'OUVERTE'}\n"
              f"Tick 1 : RM1 planifie passe -> RM2 -> tir.\n"
              f"Tick 2 : avant la passe, BA2 vient marquer RM2.")
        model = set_scene(SCENE_C, carrier=f"{RM}1", architecture="BDI_PLAN",
                          seed=1)
        model.plan_mode = mode
        model.trace = {f"{RM}1"}
        rm1 = model.get_player(f"{RM}1")
        model.steps = 1
        think(rm1)
        rm2 = model.get_player(f"{RM}2")
        model.space.move_agent(model.get_player(f"{BA}2"),
                               (rm2.pos[0] + 1.5, rm2.pos[1]))
        print("   ... BA2 vient marquer RM2 ...")
        model.steps = 2
        rm1.step()


def tp3_multiagent_execution():
    title("TP3 / Test 3 - Execution d'un plan multi-agent dans le match\n"
          "RM2 part seul (plan de conduite), BA3 arrive au pressing -> replan :\n"
          "RM3 fait un appel de balle, RM2 attend la fin de la course\n"
          "(synchronisation), passe, puis RM3 continue le plan jusqu'au tir.\n"
          "(graine 10 : sur d'autres graines BA3 recupere le ballon avant)")
    model = set_scene({
        f"{RM}2": (45.0, 34.0),
        f"{BA}3": (52.0, 34.0),   # a 7 m : pas encore de pression
        f"{RM}3": (60.0, 50.0),
        f"{BA}2": (62.0, 50.0),   # RM3 marque au depart
        f"{RM}1": (25.0, 34.0),
        f"{BA}1": (88.0, 34.0),
    }, carrier=f"{RM}2", architecture="BDI_PLAN", seed=10)
    model.trace = False
    plan_words = ("plan", "etape", "tache", "conflit", "aucun")
    last = None
    for _ in range(80):
        model.step()
        for p in model.players(RM):
            for event in p.bdi_events:
                if any(w in event for w in plan_words):
                    print(f"[tick {model.steps:3d}] {p.name}: {event}")
        plan = model.team_plans.get(RM)
        state = plan.describe() if plan else None
        if state != last:
            print(f"[tick {model.steps:3d}] >> {state or 'pas de plan RM'}")
            last = state
        if model.team_shots[RM] > 0:
            break


def tp3_resource_conflict():
    title("TP3 / Test 4 - Conflit de ressources : deux joueurs visent la meme\n"
          "zone de reception au meme moment")
    steps = [
        PlannedAction(f"{RM}3", "RUN_INTO_SPACE", earliest_tick=10, duration=5,
                      resource=(4, 2), target_pos=(65.0, 44.0)),
        PlannedAction(f"{RM}2", "RUN_INTO_SPACE", earliest_tick=11, duration=5,
                      resource=(4, 2), target_pos=(66.0, 40.0)),
        PlannedAction(f"{RM}1", "PASS", target_id=f"{RM}3", earliest_tick=15),
        PlannedAction(f"{RM}3", "SHOOT", earliest_tick=16),
    ]
    print("Plan fautif :")
    for s in steps:
        print(f"   {s.describe()}  fenetre {s.window}")
    print("Conflits detectes :")
    for i, j, reason in detect_conflicts(steps):
        print(f"   etapes {i + 1} et {j + 1} : {reason}")

    print("\nResolution 1 - ordonnancement (on decale la seconde course) :")
    fixed, notes = resolve_conflicts(steps, "schedule")
    for n in notes:
        print(f"   {n}")
    for s in fixed:
        print(f"   {s.describe()}  fenetre {s.window}")
    print(f"   conflits restants : {len(detect_conflicts(fixed))}")

    print("\nResolution 2 - reallocation (on donne une autre zone a RM2) :")
    helper = FootballModel(seed=0, verbose=False).get_player(f"{RM}1")
    fixed, notes = resolve_conflicts(steps, "reallocate", helper.free_reception_zone)
    for n in notes:
        print(f"   {n}")
    for s in fixed:
        print(f"   {s.describe()}  fenetre {s.window}")
    print(f"   conflits restants : {len(detect_conflicts(fixed))}")


def tp3_open_vs_closed():
    title("TP3 / Test 5 - Boucle ouverte vs boucle fermee (10 matchs chacun)")
    print(f"   {'mode':<8}{'buts':>6}{'tirs':>6}{'passes ok/tentees':>20}"
          f"{'plans':>7}{'replans':>9}")
    for mode in ("closed", "open"):
        goals = shots = ok = tried = plans = replans = 0
        for seed in range(10):
            m = FootballModel(seed=seed, verbose=False, architecture="BDI_PLAN",
                              plan_mode=mode)
            for _ in range(600):
                m.step()
            goals += sum(m.score.values())
            shots += m.total_shots
            ok += m.successful_passes
            tried += m.total_passes
            plans += m.plans_created
            replans += m.total_replans
        print(f"   {mode:<8}{goals:>6}{shots:>6}{f'{ok}/{tried}':>20}"
              f"{plans:>7}{replans:>9}")


# =============================================================================
# TP 4 - Communication et coordination
# =============================================================================
def manual_tick(model, names, label=None, before=None):
    """Un tick ou seuls les joueurs `names` agissent (les autres sont figes),
    puis les messages sont remis : scenario entierement controle."""
    model.steps += 1
    if label:
        print(f"\n--- tick {model.steps} : {label}")
    if before:
        before()
    for name in names:
        model.get_player(name).step()
    model.bus.deliver()


def tp4_protocol():
    title("TP4 - Mini-protocole 4.3 (joueurs adverses figes, deplaces a la main)\n"
          "1. RM1 a l'intention CREATE_SHOT     2. REQUEST de course\n"
          "3. verification + CONFIRM (ou REFUSE) 4. passe planifiee apres confirmation\n"
          "5. INFORM si le coureur est marque    6. revision + replanification")
    model = set_scene({
        f"{RM}1": (45.0, 34.0),   # porteur
        f"{BA}3": (47.5, 34.0),   # le presse (2.5 m) : pas de conduite possible
        f"{RM}3": (55.0, 50.0),   # attaquant... marque par BA2
        f"{BA}2": (56.0, 51.0),
        f"{RM}2": (35.0, 8.0),    # milieu libre, hors du champ de vision de RM1
        f"{BA}1": (88.0, 34.0),
    }, carrier=f"{RM}1", architecture="HYBRID", seed=0)
    model.trace = {f"{RM}1", f"{RM}2", f"{RM}3"}
    rm1 = model.get_player(f"{RM}1")
    team = [f"{RM}1", f"{RM}2", f"{RM}3"]

    # 1. l'intention est la premisse du protocole
    model.steps = 0
    rm1.revise_beliefs_from(rm1.perceive())
    rm1.derive_teammate_beliefs()
    rm1.adopt("CREATE_SHOT", "imposee par le scenario")

    ba3 = model.get_player(f"{BA}3")

    def keep_pressing():
        """BA3 reste colle a 2.5 m devant RM1 (pression HIGH, sans tacle)."""
        model.space.move_agent(ba3, (rm1.pos[0] + 2.5, rm1.pos[1]))

    def tick(label):
        manual_tick(model, team, label, before=keep_pressing)

    tick("RM1 planifie et DEMANDE une course a l'attaquant RM3")
    tick("RM3 verifie ses croyances : il est marque -> REFUSE")
    tick("RM1 recoit le refus, replanifie sans RM3, demande a RM2")
    tick("RM2 verifie ses croyances, accepte -> CONFIRM (engagement)")
    tick("RM1 recoit la confirmation : la passe attend la fin de la course")

    rm2 = model.get_player(f"{RM}2")
    model.space.move_agent(model.get_player(f"{BA}2"),
                           (rm2.pos[0] - 1.0, rm2.pos[1] - 2.0))
    print(f"\n   ... BA2 quitte RM3 et vient marquer RM2, a la limite du champ de "
          f"vision de RM1 (RM1-BA2 = "
          f"{rm1.distance_to(model.get_player(f'{BA}2').pos):.1f} m) :\n"
          f"   RM1 le voit mal, sa perception du marquage est peu fiable ...")
    tick("RM2 se voit marque -> INFORM a RM1")
    tick("RM1 revise sa croyance (message) et replanifie")
    tick("suite")

    b = rm1.beliefs.get(f"mate_free:{RM}2")
    print(f"\nCroyance finale de RM1 sur RM2 : mate_free = {b.value} "
          f"(source {b.source}, confiance {b.confidence})")
    print(f"Messages echanges : {model.bus.total} {model.bus.by_performative}")


def tp4_comparison():
    title("TP4 - Effet de la communication (10 matchs chacun)")
    print(f"   {'architecture':<12}{'buts':>6}{'tirs':>6}{'passes ok/tentees':>19}"
          f"{'replans':>9}{'messages':>10}")
    for arch in ("BDI_PLAN", "HYBRID"):
        goals = shots = ok = tried = replans = msgs = 0
        for seed in range(10):
            m = FootballModel(seed=seed, verbose=False, architecture=arch)
            for _ in range(600):
                m.step()
            goals += sum(m.score.values())
            shots += m.total_shots
            ok += m.successful_passes
            tried += m.total_passes
            replans += m.total_replans
            msgs += m.total_messages
        print(f"   {arch:<12}{goals:>6}{shots:>6}{f'{ok}/{tried}':>19}"
              f"{replans:>9}{msgs:>10}")


# =============================================================================
# TP 5 - Apprentissage par renforcement
# =============================================================================
import csv  # noqa: E402
import os  # noqa: E402

from learning import QLearner  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TRAIN_SEEDS = range(0, 40)       # episodes d'entrainement
EVAL_SEEDS = range(1000, 1020)   # episodes d'evaluation, jamais vus a l'entrainement
EPISODE_TICKS = 600
LEARNER_TEAM, OPPONENT = RM, BA  # RM apprend, BA garde les regles du TP 1


def run_episode(seed, learner=None, architecture="RL"):
    """Un match : RM (RL ou autre architecture) contre BA a regles fixes."""
    teams = {LEARNER_TEAM: architecture, OPPONENT: "REACTIVE"}
    learners = {LEARNER_TEAM: learner} if learner is not None else {}
    if learner is not None:
        learner.reset_stats()
    m = FootballModel(seed=seed, verbose=False, team_architectures=teams,
                      learners=learners)
    for _ in range(EPISODE_TICKS):
        m.step()
    acts = m.carrier_actions[LEARNER_TEAM]
    on_ball = sum(acts[a] for a in ("PASS", "DRIBBLE", "SHOOT")) or 1
    decisions = learner.decisions if learner is not None else on_ball
    return {
        "seed": seed,
        "epsilon": round(learner.epsilon, 3) if learner is not None else "",
        "decisions": decisions,
        "mean_reward": (learner.total_reward / max(learner.decisions, 1)
                        if learner is not None else ""),
        "loss_rate": m.ball_losses[LEARNER_TEAM] / on_ball,
        "pass": acts["PASS"] / on_ball,
        "dribble": acts["DRIBBLE"] / on_ball,
        "shoot": acts["SHOOT"] / on_ball,
        "lateral_pass": (m.lateral_passes[LEARNER_TEAM]
                         / max(m.team_passes[LEARNER_TEAM], 1)),
        "goals_for": m.score[LEARNER_TEAM],
        "goals_against": m.score[OPPONENT],
    }


def epsilon_schedule(kind, episode, n_episodes):
    if kind == "decay":   # 0.5 -> 0.05, lineaire
        return 0.5 + (0.05 - 0.5) * episode / max(n_episodes - 1, 1)
    return float(kind)


def train_and_evaluate(name, schedule, reward_table="default", pass_target="best"):
    learner = QLearner(epsilon=epsilon_schedule(schedule, 0, len(TRAIN_SEEDS)),
                       reward_table=reward_table, pass_target=pass_target, seed=0)
    train = []
    for k, seed in enumerate(TRAIN_SEEDS):
        learner.epsilon = epsilon_schedule(schedule, k, len(TRAIN_SEEDS))
        train.append(run_episode(seed, learner) | {"phase": "train"})

    # Evaluation : episodes separes, sans apprentissage ni exploration
    learner.learning = False
    learner.epsilon = 0.0
    frozen = dict(learner.q)
    evaluation = [run_episode(seed, learner) | {"phase": "eval"}
                  for seed in EVAL_SEEDS]
    assert dict(learner.q) == frozen, "Q a change pendant l'evaluation"

    export_csv(f"tp5_{name}.csv", train + evaluation)
    return learner, train, evaluation


def export_csv(filename, rows):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    fields = ["phase", "seed", "epsilon", "decisions", "mean_reward", "loss_rate",
              "pass", "dribble", "shoot", "lateral_pass", "goals_for",
              "goals_against"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def mean(rows, key):
    values = [r[key] for r in rows if r[key] != ""]
    return sum(values) / len(values) if values else float("nan")


def summary_line(label, rows):
    reward = mean(rows, "mean_reward")
    reward_text = f"{reward:+7.2f}" if reward == reward else "      -"
    return (f"   {label:<20}{reward_text}{mean(rows, 'loss_rate'):>8.1%}"
            f"{mean(rows, 'pass'):>7.0%}{mean(rows, 'dribble'):>8.0%}"
            f"{mean(rows, 'shoot'):>7.0%}{mean(rows, 'lateral_pass'):>9.0%}"
            f"{sum(r['goals_for'] for r in rows):>6}"
            f"{sum(r['goals_against'] for r in rows):>5}")


SUMMARY_HEADER = (f"   {'':<20}{'r/dec':>7}{'pertes':>8}{'PASS':>7}{'DRIBBLE':>8}"
                  f"{'SHOOT':>7}{'lateral':>9}{'BP':>6}{'BC':>5}")


def tp5_exploration():
    title("TP5 / 5.5 - Exploration / exploitation : 3 reglages de epsilon\n"
          f"REAL_MADRID apprend ({len(TRAIN_SEEDS)} matchs), puis est evalue sur "
          f"{len(EVAL_SEEDS)} autres matchs\nsans apprentissage ni exploration, "
          f"contre BARCELONA a regles fixes (TP 1).\n"
          "r/dec = recompense moyenne par decision, pertes = pertes de balle par "
          "action du porteur,\nlateral = part des passes laterales ou en retrait, "
          "BP/BC = buts pour / contre")
    configs = [("epsilon_0", "0"), ("epsilon_0.2", "0.2"), ("epsilon_decay", "decay")]
    results = {}
    print(SUMMARY_HEADER)
    for name, schedule in configs:
        learner, train, evaluation = train_and_evaluate(name, schedule)
        results[name] = (learner, train, evaluation)
        print(summary_line(f"{name} debut", train[:5]))
        print(summary_line(f"{name} fin", train[-5:]))
        print(summary_line(f"{name} EVAL", evaluation))
    rules = [run_episode(seed, architecture="REACTIVE") | {"phase": "eval"}
             for seed in EVAL_SEEDS]
    export_csv("tp5_rules.csv", rules)
    print(summary_line("regles TP1 EVAL", rules))
    print(f"\n   CSV exportes dans {RESULTS_DIR}")
    return results


def tp5_tests(results):
    learner, train, evaluation = results["epsilon_0.2"]
    title("TP5 / Tests a reussir (configuration epsilon = 0.2)")

    print("1) Les valeurs Q changent apres une recompense non nulle :")
    for s, a, r, s2, old, new in learner.history[:4]:
        print(f"   Q({s}, {a}) : {old:+.2f} -> {new:+.2f}   (r = {r:+.0f})")

    print("\n2) Une meme situation peut conduire a plusieurs actions (exploration) :")
    state, counts = max(learner.state_actions.items(),
                        key=lambda kv: sum(kv[1].values()))
    print(f"   etat le plus frequent {state} : {dict(counts)}")
    greedy = results["epsilon_0"][0]
    if state in greedy.state_actions:
        print(f"   meme etat avec epsilon = 0 : {dict(greedy.state_actions[state])}")

    print("\n3) Apres entrainement, la distribution d'actions est differente :")
    for name, (_, tr, ev) in results.items():
        first = tr[0]
        print(f"   {name:<14} 1er match : PASS {first['pass']:>4.0%} DRIBBLE "
              f"{first['dribble']:>4.0%} SHOOT {first['shoot']:>4.0%}   ->   "
              f"evaluation : PASS {mean(ev, 'pass'):>4.0%} DRIBBLE "
              f"{mean(ev, 'dribble'):>4.0%} SHOOT {mean(ev, 'shoot'):>4.0%}")

    print("\n4) Evaluation sur des episodes separes, sans apprentissage :")
    print(f"   graines d'entrainement {TRAIN_SEEDS.start}-{TRAIN_SEEDS.stop - 1}, "
          f"d'evaluation {EVAL_SEEDS.start}-{EVAL_SEEDS.stop - 1} ; learning=False, "
          f"epsilon=0 ; table Q identique avant/apres l'evaluation (verifie)")

    print("\nPolitique apprise (meilleure action par etat, avec ballon) :")
    for s in learner.known_states():
        if s[2]:
            available = ["PASS", "DRIBBLE", "SHOOT"] if s[3] else ["DRIBBLE", "SHOOT"]
            values = "  ".join(f"{a}={learner.value(s, a):+.1f}" for a in available)
            print(f"   {str(s):<40} -> {learner.policy(s, available):<8} ({values})")


def tp5_scenario_d():
    title("TP5 / Scenario D - Apprentissage d'un mauvais raccourci\n"
          "La passe va au coequipier libre le PLUS PROCHE (passe facile).\n"
          "1. toute passe reussie vaut +10, la progression n'est pas recompensee\n"
          "2. fonction corrigee (table du sujet) : seule la passe qui progresse vaut +3")
    print(SUMMARY_HEADER)
    _, _, bad = train_and_evaluate("scenarioD_shortcut", "decay", "pass_shortcut",
                                   pass_target="nearest")
    print(summary_line("raccourci EVAL", bad))
    _, _, good = train_and_evaluate("scenarioD_fixed", "decay", "default",
                                    pass_target="nearest")
    print(summary_line("corrige EVAL", good))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "tp1"):
        tp1_reactive_inhibits()
        tp1_rule_conflict()
    if which in ("all", "tp2"):
        tp2_bdi_five_ticks()
        tp2_belief_revision()
        tp2_desire_not_retained()
        tp2_intention_dropped()
        tp2_commitment_comparison()
    if which in ("all", "tp3"):
        tp3_plan_generation()
        tp3_replan_scenario_c()
        tp3_multiagent_execution()
        tp3_resource_conflict()
        tp3_open_vs_closed()
    if which in ("all", "tp4"):
        tp4_protocol()
        tp4_comparison()
    if which in ("all", "tp5"):
        tp5_tests(tp5_exploration())
        tp5_scenario_d()
