"""Point d'entree : lance un match court et affiche l'etat du terrain."""

from model import FootballModel


def print_players(model):
    owner = model.get_ball_owner()
    print(f"tick {model.steps} | ballon en ({model.ball.pos[0]:.1f}, "
          f"{model.ball.pos[1]:.1f}) | porteur = {owner.name if owner else None}")
    for p in sorted(model.players(), key=lambda a: a.name):
        print(f"   {p.name} ({p.role}) pos=({p.pos[0]:5.1f}, {p.pos[1]:5.1f}) "
              f"state={p.state}")


def main():
    model = FootballModel(seed=42)
    print_players(model)
    for _ in range(3):
        model.step()
        print_players(model)


if __name__ == "__main__":
    main()
