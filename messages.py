"""Messages inter-agents (TP 4).

Communication explicite, point a point ou par diffusion, avec des actes de
langage (performatifs). Un message envoye pendant le tick t est remis au
debut du tick t+1 : les informations recues ont donc toujours un peu d'age.
"""

from dataclasses import dataclass, field

# --- Performatifs (4.2) ---------------------------------------------------------
INFORM = "INFORM"      # partager une croyance            "je suis marque"
REQUEST = "REQUEST"    # demander une action              "fais une course"
PROPOSE = "PROPOSE"    # proposer une action / allocation "je peux recevoir"
CONFIRM = "CONFIRM"    # accepter : engagement coordonne  "course confirmee"
REFUSE = "REFUSE"      # refuser une requete              "je suis marque"

# --- Contenus utilises ------------------------------------------------------------
RUN_INTO_SPACE = "RUN_INTO_SPACE"   # REQUEST / CONFIRM / REFUSE
RECEIVE = "RECEIVE"                 # PROPOSE : je peux recevoir une passe
TASK_CANCELLED = "task_cancelled"   # INFORM : la tache demandee est annulee


@dataclass
class Message:
    sender_id: str               # nom de l'agent emetteur
    receiver_id: str | None      # nom du destinataire, None = diffusion equipe
    performative: str
    content: str
    payload: dict = field(default_factory=dict)
    tick: int = 0

    def describe(self):
        dest = self.receiver_id or "equipe"
        extra = ""
        if "value" in self.payload:
            extra = f"={self.payload['value']}"
        if "reason" in self.payload:
            extra += f" ({self.payload['reason']})"
        return f"{self.performative} {self.sender_id}->{dest} : {self.content}{extra}"


class MessageBus:
    """Boite aux lettres du modele : collecte les messages du tick et les
    remet aux destinataires au debut du tick suivant."""

    def __init__(self, model):
        self.model = model
        self.outgoing = []
        self.total = 0
        self.by_performative = {}

    def send(self, message):
        self.outgoing.append(message)
        self.total += 1
        self.by_performative[message.performative] = (
            self.by_performative.get(message.performative, 0) + 1)

    def deliver(self):
        for msg in self.outgoing:
            sender = self.model.get_player(msg.sender_id)
            if msg.receiver_id is None:
                receivers = [p for p in self.model.players(sender.team)
                             if p is not sender]
            else:
                receivers = [self.model.get_player(msg.receiver_id)]
            for agent in receivers:
                agent.inbox.append(msg)
        self.outgoing = []
