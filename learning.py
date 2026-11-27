"""Apprentissage par renforcement (TP 5).

5.2  Etat discret : x = (zone, pression, possession, coequipier libre).
5.3  Actions du porteur : U(x) = {PASS, DRIBBLE, SHOOT} et renforcements.
5.4  Q-learning tabulaire :
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
5.5  Exploration epsilon-gloutonne (epsilon constant ou decroissant).
"""

import random
from collections import Counter, defaultdict  # noqa: F401

ACTIONS = ("PASS", "DRIBBLE", "SHOOT")

# --- 5.2 Zones pour l'etat discret (distance au but adverse) ---------------------
def zone_label(goal_dist):
    if goal_dist < 16.5:
        return "SHOOTING"
    if goal_dist < 35.0:
        return "ATTACKING"
    if goal_dist < 60.0:
        return "MIDDLE"
    return "DEFENSIVE"


# =============================================================================
# 5.3  Fonctions de renforcement
# =============================================================================
REWARDS = {
    # table du sujet : progression recompensee, perte de balle punie
    "default": {
        "goal": 20.0,
        "progressive_pass": 3.0,   # passe reussie qui fait gagner >= 5 m
        "other_pass": 0.0,         # passe reussie laterale ou en retrait
        "shooting_zone": 2.0,      # entree dans la zone de tir
        "ball_lost": -6.0,
        "long_shot_missed": -3.0,  # tir rate de plus de 25 m
        "neutral": 0.0,
    },
    # scenario D : TOUTE passe reussie est fortement recompensee,
    # la progression ne compte pas -> raccourci a decouvrir
    "pass_shortcut": {
        "goal": 20.0,
        "progressive_pass": 10.0,
        "other_pass": 10.0,
        "shooting_zone": 0.0,
        "ball_lost": -6.0,
        "long_shot_missed": -3.0,
        "neutral": 0.0,
    },
}

PROGRESSIVE_GAIN = 5.0     # une passe "progresse" si le receveur est 5 m plus haut
LONG_SHOT = 25.0


def compute_reward(table, event):
    """event : dictionnaire decrivant ce qui s'est passe depuis la decision.
    Renvoie (recompense, liste des raisons)."""
    reward, reasons = 0.0, []

    def add(key):
        nonlocal reward
        if table[key]:
            reward += table[key]
            reasons.append(f"{key} {table[key]:+.0f}")

    if event["goal"]:
        add("goal")
    if event["pass_result"] == "PASS_OK":
        add("progressive_pass" if event["pass_gain"] >= PROGRESSIVE_GAIN
            else "other_pass")
    if event["shot_result"] == "SHOT_MISSED" and event["shot_dist"] > LONG_SHOT:
        add("long_shot_missed")
    if event["entered_shooting_zone"]:
        add("shooting_zone")
    if event["ball_lost"]:
        add("ball_lost")
    if not reasons:
        reasons.append("neutre 0")
    return reward, reasons


# =============================================================================
# 5.4  Q-learning tabulaire
# =============================================================================
class QLearner:

    def __init__(self, actions=ACTIONS, alpha=0.2, gamma=0.95, epsilon=0.2,
                 reward_table="default", pass_target="best", seed=None):
        """pass_target : "best"    -> passe au coequipier libre le plus avance
                         "nearest" -> passe au coequipier libre le plus proche
                                      (passe "facile", souvent laterale)"""
        self.actions = list(actions)
        self.pass_target = pass_target
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning = True            # False : evaluation sans apprentissage
        self.rewards = REWARDS[reward_table]
        self.q = {}                     # {(etat, action): valeur}, 0 par defaut
        self.rng = random.Random(seed)

        # statistiques (remises a zero par reset_stats)
        self.history = []               # mises a jour a recompense non nulle
        self.state_actions = defaultdict(Counter)   # actions choisies par etat
        self.reset_stats()

    def reset_stats(self):
        self.action_counts = Counter()
        self.decisions = 0
        self.total_reward = 0.0
        self.ball_losses = 0
        self.updates = 0

    def value(self, state, action):
        """Q(s, a) ; lire une valeur ne cree pas d'entree dans la table."""
        return self.q.get((state, action), 0.0)

    # --- choix epsilon-glouton, restreint aux actions possibles U(x) -------
    def choose(self, state, available=None):
        available = list(available or self.actions)
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(available)
            explored = True
        else:
            best = max(self.value(state, a) for a in available)
            # departage aleatoire entre actions de meme valeur
            action = self.rng.choice(
                [a for a in available if self.value(state, a) == best])
            explored = False
        self.action_counts[action] += 1
        self.state_actions[state][action] += 1
        self.decisions += 1
        return action, explored

    # --- mise a jour de Q ----------------------------------------------------
    def update(self, state, action, reward, next_state, next_available=None):
        self.total_reward += reward
        if not self.learning:
            return None
        old = self.value(state, action)
        next_actions = list(next_available) if next_available else []
        best_next = max((self.value(next_state, a) for a in next_actions),
                        default=0.0)      # pas d'action possible : etat terminal
        target = reward + self.gamma * best_next
        new = old + self.alpha * (target - old)
        self.q[(state, action)] = new
        self.updates += 1
        if reward != 0 and len(self.history) < 500:
            self.history.append((state, action, reward, next_state, old, new))
        return old, new

    # --- lecture ---------------------------------------------------------------
    def policy(self, state, available=None):
        available = list(available or self.actions)
        return max(available, key=lambda a: self.value(state, a))

    def proportions(self):
        total = sum(self.action_counts.values()) or 1
        return {a: self.action_counts[a] / total for a in self.actions}

    def known_states(self):
        return sorted({s for s, _ in self.q})
