"""Croyances et revision des croyances (TP 2).

Une croyance n'est plus une simple valeur : elle porte sa source,
le tick ou l'information a ete observee et un niveau de confiance.

Politique de revision (inspiree des TMS, sans TMS complet) :
  1. la confiance d'une croyance s'erode avec son age
     (confiance effective = confiance * DECAY ** age) ;
  2. une nouvelle information est acceptee si sa confiance effective
     est STRICTEMENT superieure a celle de la croyance en place ;
  3. a confiance effective egale, la plus recente l'emporte ;
  4. sinon elle est rejetee : on garde la croyance existante
     (conserver ce qui est compatible, changer le minimum necessaire).
"""

from dataclasses import dataclass

DECAY = 0.95   # perte de confiance par tick d'age
EPS = 1e-9

# Categories de croyances (grille du support de cours, section 2.2)
ENVIRONMENT = "environnementale"
RELATIONAL = "relationnelle"
SOCIAL = "sociale"
PERSONAL = "personnelle"

# Croyances de routine : grandeurs continues ou details qui changent presque
# a chaque tick. Leur mise a jour n'est pas comptee comme une "revision".
ROUTINE = {"ball_position", "dist_ball", "goal_dist", "nearest_opp_dist",
           "nearest_opponent", "mate_too_close", "last_action"}


@dataclass
class Belief:
    value: object
    source: str          # "perception", "message:<agent>", "deduction", "defaut"
    tick: int            # tick ou l'information a ete observee
    confidence: float = 1.0
    category: str = ENVIRONMENT

    def effective_confidence(self, now):
        return self.confidence * DECAY ** max(0, now - self.tick)


@dataclass
class Revision:
    """Trace d'une tentative de revision (pour le journal)."""
    key: str
    status: str          # "new", "updated", "revised", "rejected"
    old: Belief | None
    new: Belief
    reason: str


class BeliefBase:

    def __init__(self):
        self._beliefs = {}
        self.revision_count = 0      # indicateur micro : revisions effectives
        self.history = []            # revisions et rejets (hors grandeurs continues)

    # ------------------------------------------------------------ lecture
    def value(self, key, default=None):
        belief = self._beliefs.get(key)
        return default if belief is None else belief.value

    def get(self, key):
        return self._beliefs.get(key)

    def snapshot(self):
        """Dictionnaire {cle: valeur} : la memoire de travail des regles."""
        return {k: b.value for k, b in self._beliefs.items()}

    def by_category(self):
        result = {}
        for key, b in self._beliefs.items():
            result.setdefault(b.category, {})[key] = b
        return result

    # ----------------------------------------------------------- revision
    def revise(self, key, value, source, tick, now, confidence=1.0,
               category=ENVIRONMENT):
        new = Belief(value, source, tick, confidence, category)
        old = self._beliefs.get(key)

        if old is None:
            self._beliefs[key] = new
            return Revision(key, "new", None, new, "premiere information")

        old_conf = old.effective_confidence(now)
        new_conf = new.effective_confidence(now)

        if new_conf > old_conf + EPS:
            reason = (f"plus fiable ({new_conf:.2f} > {old_conf:.2f})")
        elif abs(new_conf - old_conf) <= EPS and tick >= old.tick:
            reason = (f"confiance egale ({new_conf:.2f}), "
                      f"plus recente (tick {tick} >= {old.tick})")
        else:
            reason = (f"moins fiable ({new_conf:.2f} < {old_conf:.2f})"
                      if new_conf < old_conf - EPS
                      else f"plus ancienne (tick {tick} < {old.tick})")
            revision = Revision(key, "rejected", old, new, reason)
            self.history.append(revision)
            return revision

        self._beliefs[key] = new
        routine = key in ROUTINE and old.source == new.source
        if old.value == new.value or routine:
            return Revision(key, "updated", old, new, reason)

        self.revision_count += 1
        revision = Revision(key, "revised", old, new, reason)
        self.history.append(revision)
        return revision
