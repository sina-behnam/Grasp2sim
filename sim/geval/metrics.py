import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# -------------------- enums --------------------

class BodyName(str, Enum):
    """MuJoCo body names used by the evaluator."""
    HAND         = "hand"
    LEFT_FINGER  = "left_finger"
    RIGHT_FINGER = "right_finger"

    def __str__(self):
        return self.value

    @classmethod
    def fingers(cls):
        return (cls.LEFT_FINGER, cls.RIGHT_FINGER)


class JointName(str, Enum):
    """MuJoCo joint names used by the evaluator."""
    FINGER_JOINT1 = "finger_joint1"
    FINGER_JOINT2 = "finger_joint2"

    def __str__(self):
        return self.value


class Phase(str, Enum):
    """Named execution phases. Executors call mark_phase(Phase.X)."""
    INIT     = "init"
    APPROACH = "approach"
    CLOSE    = "close"
    RETREAT  = "retreat"
    LIFT     = "lift"
    DONE     = "done"

    def __str__(self):
        return self.value


class FailureMode(str, Enum):
    """Outcome classes derived from ObjState."""
    OK                  = "ok"
    NO_CONTACT_AT_CLOSE = "no_contact_at_close"
    LOST_DURING_RETREAT = "lost_during_retreat"
    SLIPPED_DURING_LIFT = "slipped_during_lift"

    def __str__(self):
        return self.value


# -------------------- data model --------------------

@dataclass
class PhaseSnapshot:
    """State captured at a phase boundary (start of phase == end of previous)."""
    t: float
    phase: str
    finger_gap: float
    n_finger_obj_contacts: int
    hand_pos: np.ndarray
    obj_pos: Optional[np.ndarray] = None
    obj_quat: Optional[np.ndarray] = None


@dataclass
class ContactTimeSeries:
    """Per-step finger <-> target-object contact instrumentation."""
    t:           List[float] = field(default_factory=list)
    phase:       List[str]   = field(default_factory=list)
    n_contacts:  List[int]   = field(default_factory=list)
    left_fn:     List[float] = field(default_factory=list)
    left_ft:     List[float] = field(default_factory=list)
    right_fn:    List[float] = field(default_factory=list)
    right_ft:    List[float] = field(default_factory=list)
    finger_gap:  List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    def append_sample(self, *, t, phase, n_contacts,
                      left_fn, left_ft, right_fn, right_ft, finger_gap):
        self.t.append(float(t))
        self.phase.append(str(phase))
        self.n_contacts.append(int(n_contacts))
        self.left_fn.append(float(left_fn))
        self.left_ft.append(float(left_ft))
        self.right_fn.append(float(right_fn))
        self.right_ft.append(float(right_ft))
        self.finger_gap.append(float(finger_gap))


@dataclass
class OrientationTimeSeries:
    """Per-step world-frame approach and binormal (finger-closing) unit axes of the hand."""
    t:        List[float]      = field(default_factory=list)
    phase:    List[str]        = field(default_factory=list)
    approach: List[np.ndarray] = field(default_factory=list)
    binormal: List[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    def append_sample(self, *, t, phase, approach, binormal):
        self.t.append(float(t))
        self.phase.append(str(phase))
        self.approach.append(np.asarray(approach, dtype=np.float64).copy())
        self.binormal.append(np.asarray(binormal, dtype=np.float64).copy())

    def as_arrays(self):
        return (np.asarray(self.t),
                np.asarray(self.approach),   # (N, 3)
                np.asarray(self.binormal))   # (N, 3)


@dataclass
class ObjState:
    """
    Three-stage state of a grasp execution w.r.t. the target object:
      seated — fingers had contact right after close + settle
      held   — contact survived the retreat motion
      lifted — at least one object crossed the lift-height threshold (success)
    """
    seated: bool
    held:   bool
    lifted: bool

    @property
    def failure_mode(self) -> FailureMode:
        if self.lifted:
            return FailureMode.OK
        if not self.seated:
            return FailureMode.NO_CONTACT_AT_CLOSE
        if not self.held:
            return FailureMode.LOST_DURING_RETREAT
        return FailureMode.SLIPPED_DURING_LIFT


