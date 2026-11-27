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
