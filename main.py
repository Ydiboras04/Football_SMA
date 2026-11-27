"""Point d'entree : lance un match, affiche le journal puis les indicateurs."""

from model import LEFT_TEAM, RIGHT_TEAM, FootballModel

TRACE_TICKS = 5      # nombre de ticks avec le journal detaille de chaque joueur
MATCH_TICKS = 600    # duree totale du match

# (cle de model.metrics(), libelle, format)
MACRO = [
    ("goals_for", "buts marques", "{:.0f}"),
    ("goals_against", "buts encaisses", "{:.0f}"),
    ("shots", "tirs", "{:.0f}"),
    ("passes", "passes tentees", "{:.0f}"),
    ("pass_success", "passes reussies", "{:.0%}"),
    ("ball_progression", "progression / possession (m)", "{:.1f}"),
    ("ball_losses", "pertes de balle", "{:.0f}"),
    ("recovery_time", "temps de recuperation (ticks)", "{:.1f}"),
    ("possession", "possession", "{:.0%}"),
    ("dispersion", "dispersion spatiale (m)", "{:.1f}"),
    ("reward", "recompense cumulee", "{:.1f}"),
]
MICRO = [
    ("belief_revisions", "revisions de croyances", "{:.0f}"),
    ("intention_changes", "changements d'intention", "{:.0f}"),
    ("dropped_rate", "intentions abandonnees (impossibles)", "{:.0%}"),
    ("replans", "replanifications", "{:.0f}"),
    ("messages", "messages envoyes", "{:.0f}"),
    ("reactive_share", "decisions de la couche reactive", "{:.0%}"),
    ("q_share", "decisions du Q-learning", "{:.0%}"),
]


def print_indicators(model):
    metrics = {team: model.metrics(team) for team in (LEFT_TEAM, RIGHT_TEAM)}
    width = max(len(label) for _, label, _ in MACRO + MICRO) + 2

    def section(title, rows):
        print(f"\n{title}")
        print(f"   {'':<{width}}{LEFT_TEAM:>14}{RIGHT_TEAM:>14}")
        for key, label, fmt in rows:
            cells = []
            for team in (LEFT_TEAM, RIGHT_TEAM):
                value = metrics[team][key]
                cells.append("-" if value != value else fmt.format(value))  # NaN
            print(f"   {label:<{width}}{cells[0]:>14}{cells[1]:>14}")

    section("Indicateurs MACRO (comportement collectif)", MACRO)
    section("Indicateurs MICRO (decisions individuelles)", MICRO)


def main():
    model = FootballModel()

    for _ in range(MATCH_TICKS):
        # Journal detaille au debut, puis seulement les evenements.
        model.trace = model.steps < TRACE_TICKS
        model.step()

    print(f"\nFin du match apres {model.steps} ticks : {model.score_text()}")
    print(f"Plans crees : {model.plans_created} | Messages : "
          f"{model.total_messages} {model.bus.by_performative}")
    print_indicators(model)


if __name__ == "__main__":
    main()
