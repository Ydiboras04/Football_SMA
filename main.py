"""Point d'entree : lance un match et affiche le journal du TP 0."""

from model import FootballModel

TRACE_TICKS = 5      # nombre de ticks avec le journal detaille de chaque joueur
MATCH_TICKS = 600    # duree totale du match


def main():
    model = FootballModel()

    for _ in range(MATCH_TICKS):
        # Journal detaille au debut, puis seulement les evenements.
        model.trace = model.steps < TRACE_TICKS
        model.step()

    print(f"\nFin du match apres {model.steps} ticks : {model.score_text()}")
    print(f"Passes reussies : {model.successful_passes} / {model.total_passes}"
          f" | Tirs : {model.total_shots}")
    print(f"Plans crees : {model.plans_created} | Replanifications : "
          f"{model.total_replans}")


if __name__ == "__main__":
    main()
