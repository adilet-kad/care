"""care.verify -- the Verifier: projection of repair candidates onto F_H."""

from care.verify.projector import apply_repair, hard_violation_set
from care.verify.verifier import Verifier

__all__ = ["Verifier", "apply_repair", "hard_violation_set"]
