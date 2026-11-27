"""Point d'entree : lance un match et affiche le journal du TP 0."""

from model import FootballModel

TRACE_TICKS = 5      # nombre de ticks avec le journal detaille de chaque joueur
MATCH_TICKS = 600    # duree totale du match


def main():
    model = FootballModel(seed=42)

    for _ in range(MATCH_TICKS):
        # Journal detaille au debut, puis seulement les evenements.
        model.trace = model.steps < TRACE_TICKS
        model.step()

    print(f"\nFin du match apres {model.steps} ticks : "
          f"A {model.score['A']} - {model.score['B']} B")


if __name__ == "__main__":
    main()
