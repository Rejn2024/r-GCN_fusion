import logging

import pytest

from rgcn_fusion.skill_transition import (
    BoundedDistribution, FlightControls, NativeJSBSimAdapter, SkillManager,
    default_skill_specs,
)


def controllers(specs):
    return {name: lambda observation, params: FlightControls(aileron=params["bank_command"])
            for name in specs}


def test_parameters_and_duration_are_bounded_and_reproducible():
    specs = default_skill_specs()
    first = SkillManager(specs, controllers(specs), seed=4).select({}, 10.0)
    second = SkillManager(specs, controllers(specs), seed=4).select({}, 10.0)
    assert first == second
    assert 13.0 <= first.expires_at <= 22.0
    assert -0.8 <= first.parameters["bank_command"] <= 0.8


def test_dwell_geometry_and_emergency_transitions(caplog):
    specs = default_skill_specs()
    manager = SkillManager(specs, controllers(specs), seed=2)
    with caplog.at_level(logging.INFO):
        initial = manager.select({}, 0.0)
        assert manager.select({}, initial.expires_at - 0.01) is initial
        pursue = manager.select({}, initial.expires_at)
        assert pursue.skill == "pursue_target"
        assert manager.select({"target_in_launch_envelope": True}, initial.expires_at + 0.1).skill == "launch"
        evasion = manager.select({"incoming_active_missile": True}, initial.expires_at + 0.2)
    assert evasion.skill == "missile_evasion"
    assert evasion.transition_reason == "incoming_active_missile"
    assert "skill_transition" in caplog.text and "reason=incoming_active_missile" in caplog.text


@pytest.mark.parametrize("observation, skill, reason", [
    ({"fuel_fraction": .1, "bingo_fuel_fraction": .2}, "disengage", "low_fuel"),
    ({"weapons_remaining": 0}, "disengage", "no_weapons"),
    ({"target_destroyed": True}, "search", "target_destroyed"),
])
def test_mission_interrupts(observation, skill, reason):
    specs = default_skill_specs()
    manager = SkillManager(specs, controllers(specs))
    manager.select({}, 0)
    state = manager.select(observation, 1)
    assert (state.skill, state.transition_reason) == (skill, reason)


class FakeBackend:
    backend_name = "jsbsim-cpp"
    def __init__(self): self.steps = []
    def observation(self): return {}
    def step(self, controls, dt): self.steps.append((controls, dt))


def test_rollout_uses_native_backend_and_steps_to_exact_duration():
    specs = default_skill_specs()
    manager = SkillManager(specs, controllers(specs), seed=1)
    backend = FakeBackend()
    history = manager.rollout(backend, duration=1.0, dt=.3)
    assert len(backend.steps) == 4
    assert sum(dt for _, dt in backend.steps) == pytest.approx(1.0)
    assert history[0].skill == "maintain_position"
    backend.backend_name = "python"
    with pytest.raises(ValueError, match="C\\+\\+"):
        manager.rollout(backend, duration=1, dt=.1)


def test_native_adapter_writes_jsbsim_controls_and_runs_fdm():
    class FDM(dict):
        def set_dt(self, dt): self.dt = dt
        def run(self): self.ran = True

    fdm = FDM()
    adapter = NativeJSBSimAdapter(fdm, lambda current: {"altitude": current.get("h", 0)})
    adapter.step(FlightControls(aileron=2, elevator=-2, rudder=.2, throttle=2), .01)
    assert fdm["fcs/aileron-cmd-norm"] == 1
    assert fdm["fcs/elevator-cmd-norm"] == -1
    assert fdm["fcs/rudder-cmd-norm"] == .2
    assert fdm["fcs/throttle-cmd-norm"] == 1
    assert fdm.dt == .01 and fdm.ran
