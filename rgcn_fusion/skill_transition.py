"""Stochastic, interruptible skill selection for native JSBSim rollouts.

The selector is deliberately independent of a particular JSBSim Python wrapper:
``rollout`` accepts any adapter implementing :class:`JSBSimBackend`.  Production
adapters should delegate ``step`` to JSBSim's native C++ flight-dynamics model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import random
from typing import Any, Callable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class BoundedDistribution:
    """A random parameter distribution with an enforced closed interval."""

    low: float
    high: float
    distribution: str = "uniform"
    mean: float | None = None
    stddev: float | None = None

    def sample(self, rng: random.Random) -> float:
        if not math.isfinite(self.low) or not math.isfinite(self.high) or self.low > self.high:
            raise ValueError("distribution bounds must be finite and low <= high")
        if self.distribution == "uniform":
            value = rng.uniform(self.low, self.high)
        elif self.distribution == "normal":
            mean = (self.low + self.high) / 2 if self.mean is None else self.mean
            stddev = (self.high - self.low) / 6 if self.stddev is None else self.stddev
            if stddev < 0 or not math.isfinite(stddev):
                raise ValueError("stddev must be finite and non-negative")
            value = rng.gauss(mean, stddev)
        else:
            raise ValueError(f"unsupported distribution: {self.distribution}")
        return min(self.high, max(self.low, value))


@dataclass(frozen=True)
class SkillSpec:
    name: str
    duration: BoundedDistribution
    parameters: Mapping[str, BoundedDistribution] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectorState:
    skill: str
    entered_at: float
    expires_at: float
    parameters: Mapping[str, float]
    transition_reason: str


@dataclass(frozen=True)
class FlightControls:
    """Normalized JSBSim flight-control command."""

    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 1.0


class JSBSimBackend(Protocol):
    """Minimal adapter contract for an accelerated JSBSim backend."""

    backend_name: str

    def observation(self) -> Mapping[str, Any]: ...

    def step(self, controls: FlightControls, dt: float) -> None: ...


class NativeJSBSimAdapter:
    """Adapt a ``jsbsim.FGFDMExec`` instance to :class:`JSBSimBackend`.

    ``observer`` translates FDM properties plus tactical sensor state into the
    mapping consumed by the selector. This keeps weapon/threat modelling out of
    the flight-dynamics adapter.
    """

    backend_name = "jsbsim-cpp"

    def __init__(
        self,
        fdm: Any,
        observer: Callable[[Any], Mapping[str, Any]],
    ) -> None:
        if not callable(getattr(fdm, "run", None)):
            raise TypeError("fdm must be a configured jsbsim.FGFDMExec-like object")
        self.fdm = fdm
        self.observer = observer

    def observation(self) -> Mapping[str, Any]:
        return self.observer(self.fdm)

    def step(self, controls: FlightControls, dt: float) -> None:
        self.fdm["fcs/aileron-cmd-norm"] = min(1.0, max(-1.0, controls.aileron))
        self.fdm["fcs/elevator-cmd-norm"] = min(1.0, max(-1.0, controls.elevator))
        self.fdm["fcs/rudder-cmd-norm"] = min(1.0, max(-1.0, controls.rudder))
        self.fdm["fcs/throttle-cmd-norm"] = min(1.0, max(0.0, controls.throttle))
        if callable(set_dt := getattr(self.fdm, "set_dt", None)):
            set_dt(dt)
        if self.fdm.run() is False:
            raise RuntimeError("JSBSim stopped before rollout completion")


SkillController = Callable[[Mapping[str, Any], Mapping[str, float]], FlightControls]


DEFAULT_SEQUENCE = (
    "maintain_position", "pursue_target", "launch", "crank_maneuver",
    "support_missile", "turn_cold", "recommit",
)


class SkillManager:
    """Select skills by dwell time, geometry, and safety-priority interrupts."""

    def __init__(
        self,
        skills: Mapping[str, SkillSpec],
        controllers: Mapping[str, SkillController],
        *,
        sequence: Sequence[str] = DEFAULT_SEQUENCE,
        seed: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        required = {"launch", "missile_evasion", "disengage", "search"}
        if not sequence or any(name not in skills for name in sequence):
            raise ValueError("sequence must be non-empty and reference defined skills")
        if missing := required.difference(skills):
            raise ValueError(f"missing interrupt skills: {sorted(missing)}")
        if any(name not in controllers for name in skills):
            raise ValueError("every skill requires a controller")
        self.skills = dict(skills)
        self.controllers = dict(controllers)
        self.sequence = tuple(sequence)
        self.rng = random.Random(seed)
        self.logger = logger or logging.getLogger(__name__)
        self.state: SelectorState | None = None

    def _enter(self, skill: str, now: float, reason: str) -> SelectorState:
        spec = self.skills[skill]
        duration = spec.duration.sample(self.rng)
        if duration <= 0:
            raise ValueError(f"skill {skill!r} must have a positive duration")
        self.state = SelectorState(
            skill, now, now + duration,
            {name: dist.sample(self.rng) for name, dist in spec.parameters.items()},
            reason,
        )
        self.logger.info(
            "skill_transition skill=%s reason=%s entered_at=%.3f expires_at=%.3f parameters=%s",
            skill, reason, now, now + duration, dict(self.state.parameters),
        )
        return self.state

    @staticmethod
    def _interrupt(observation: Mapping[str, Any]) -> tuple[str, str] | None:
        # Ordering is intentional: survival takes precedence over mission state.
        if observation.get("incoming_active_missile", False):
            return "missile_evasion", "incoming_active_missile"
        if observation.get("fuel_fraction", 1.0) <= observation.get("bingo_fuel_fraction", 0.15):
            return "disengage", "low_fuel"
        if observation.get("weapons_remaining", 1) <= 0:
            return "disengage", "no_weapons"
        if observation.get("target_destroyed", False):
            return "search", "target_destroyed"
        return None

    def select(self, observation: Mapping[str, Any], now: float) -> SelectorState:
        if self.state is None:
            return self._enter(self.sequence[0], now, "initial")
        interrupt = self._interrupt(observation)
        if interrupt and interrupt[0] != self.state.skill:
            return self._enter(interrupt[0], now, interrupt[1])
        # Geometry may bring a launch forward, without defeating safety interrupts.
        if (
            self.state.skill == "pursue_target"
            and observation.get("target_in_launch_envelope", False)
        ):
            return self._enter("launch", now, "target_in_launch_envelope")
        if now >= self.state.expires_at:
            try:
                index = self.sequence.index(self.state.skill)
                next_skill = self.sequence[(index + 1) % len(self.sequence)]
            except ValueError:  # emergency skills return to the tactical sequence
                next_skill = self.sequence[0]
            return self._enter(next_skill, now, "duration_elapsed")
        return self.state

    def controls(self, observation: Mapping[str, Any], now: float) -> FlightControls:
        state = self.select(observation, now)
        return self.controllers[state.skill](observation, state.parameters)

    def rollout(self, backend: JSBSimBackend, *, duration: float, dt: float) -> list[SelectorState]:
        """Run a rollout, refusing a non-native (non-C++) dynamics backend."""
        if getattr(backend, "backend_name", None) not in {"jsbsim-cpp", "jsbsim_native"}:
            raise ValueError("rollouts require the accelerated native JSBSim C++ backend")
        if duration <= 0 or dt <= 0:
            raise ValueError("duration and dt must be positive")
        history: list[SelectorState] = []
        now = 0.0
        while now < duration:
            observation = backend.observation()
            controls = self.controls(observation, now)
            assert self.state is not None
            if not history or history[-1] is not self.state:
                history.append(self.state)
            backend.step(controls, min(dt, duration - now))
            now += min(dt, duration - now)
        return history


def default_skill_specs() -> dict[str, SkillSpec]:
    """Return conservative example bounds; tune them for the simulated aircraft."""
    tactical = list(DEFAULT_SEQUENCE) + ["missile_evasion", "disengage", "search"]
    return {
        name: SkillSpec(
            name,
            BoundedDistribution(3.0, 12.0),
            {"bank_command": BoundedDistribution(-0.8, 0.8, "normal")},
        )
        for name in tactical
    }
