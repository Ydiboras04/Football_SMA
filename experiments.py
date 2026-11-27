"""Scenarios controles pour valider les TP.

Chaque scenario place les joueurs a la main, donne le ballon a un joueur,
puis execute UN seul cycle du joueur observe avec le journal active.

    python experiments.py
"""

from model import LEFT_TEAM, RIGHT_TEAM, FootballModel

RM = LEFT_TEAM    # attaque vers la droite (x croissant)
BA = RIGHT_TEAM   # attaque vers la gauche


def set_scene(positions, carrier, seed=0):
    """positions : {nom_joueur: (x, y)} ; carrier : nom du porteur."""
    model = FootballModel(seed=seed, verbose=True)
    for name, pos in positions.items():
        model.space.move_agent(model.get_player(name), pos)
    model.give_ball(model.get_player(carrier))
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
    }, carrier=f"{RM}3")
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
    }, carrier=f"{BA}3")
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
    }, carrier=f"{RM}3")
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
    }, carrier=f"{RM}3")
    run_one_cycle(model, f"{RM}3")


if __name__ == "__main__":
    tp1_reactive_inhibits()
    tp1_rule_conflict()
