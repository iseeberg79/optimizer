import numpy as np

from optimizer.optimizer import (
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
)


def _solve(battery_first, p_N):
    """
    3 flat-priced slots, battery can fully charge in a single slot (c_max covers the whole
    capacity within one step). Charging is worth exactly p_a per Wh regardless of which slot
    it happens in, so absent battery_first the timing is cost-neutral and left to the solver.
    """
    grid = GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None)
    battery = BatteryConfig(
        charge_from_grid=True, discharge_to_grid=False,
        s_capacity=1000, s_min=0, s_max=1000, s_initial=0,
        c_min=0, c_max=5000, d_max=0, p_a=0.30, battery_first=battery_first,
    )
    time_series = TimeSeriesData(
        dt=[3600, 3600, 3600], gt=[0, 0, 0], ft=[0, 0, 0],
        p_N=p_N, p_E=[0.10, 0.10, 0.10],
    )
    return Optimizer(OptimizationStrategy('none', 'none'), grid, [battery], time_series, eta_c=1.0, eta_d=1.0).solve()


def test_battery_first_prefers_earliest_slot():
    """With battery_first, a cost-neutral charge is placed in the first available slot."""
    result = _solve(battery_first=True, p_N=[0.30, 0.30, 0.30])

    assert result['status'] == 'Optimal'
    charge = result['batteries'][0]['charging_power']
    assert np.isclose(charge[0], 1000, atol=1.0), "should charge fully in the first slot"
    assert np.isclose(charge[1], 0, atol=1.0)
    assert np.isclose(charge[2], 0, atol=1.0)


def test_battery_first_is_cost_neutral():
    """The tie-break must not change the real economic outcome: same total energy charged at
    the same (flat) price either way. Not compared via get_clean_objective_value() - its
    s[T-1] - s[0] baseline (not s_initial) drops the value of energy charged in slot 0 itself,
    a pre-existing quirk unrelated to battery_first (see evcc-io/optimizer#125/#126)."""
    deferred = _solve(battery_first=False, p_N=[0.30, 0.30, 0.30])
    early = _solve(battery_first=True, p_N=[0.30, 0.30, 0.30])

    assert np.isclose(sum(deferred['batteries'][0]['charging_power']),
                       sum(early['batteries'][0]['charging_power']), atol=1.0)
    assert np.isclose(deferred['batteries'][0]['state_of_charge'][-1],
                       early['batteries'][0]['state_of_charge'][-1], atol=1.0)


def test_battery_first_yields_to_real_price_difference():
    """battery_first is a tie-break only: a real price advantage in a later slot still wins."""
    result = _solve(battery_first=True, p_N=[0.50, 0.10, 0.50])

    assert result['status'] == 'Optimal'
    charge = result['batteries'][0]['charging_power']
    assert np.isclose(charge[1], 1000, atol=1.0), "cheap slot should still win over early slots"
    assert np.isclose(charge[0], 0, atol=1.0)
    assert np.isclose(charge[2], 0, atol=1.0)


def test_battery_first_inactive_by_default():
    """battery_first defaults to False and must not implicitly bias timing."""
    grid = GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None)
    battery = BatteryConfig(
        charge_from_grid=True, discharge_to_grid=False,
        s_capacity=1000, s_min=0, s_max=1000, s_initial=0,
        c_min=0, c_max=5000, d_max=0, p_a=0.30,
    )
    assert battery.battery_first is False


def _solve_short_first_slot(dt0, battery_first, n_slots=8):
    """
    Mirrors evcc's real request shape: dt[0] is the short remainder of the ongoing 15-min slot
    (shrinks toward 0 as the ~2-minute update loop re-solves within the same quarter-hour),
    dt[1:] are full 900s slots, all at the same flat price. s_capacity (3000 Wh) needs at least
    two slots at full c_max (6000 W * 900s = 1500 Wh each) to fill, but the flat window offers
    more slots than that (8), leaving genuine slack in *which* of the equally-cheap slots get
    used at max power - that slack is exactly what battery_first is meant to resolve in favour
    of the earliest ones. p_a (0.35) sits above p_N (0.30), so charging fully is worthwhile, but
    nothing distinguishes *when* within the flat window - without that gap, charging would never
    be profitable regardless of dt0 or battery_first, making the test vacuous.
    """
    grid = GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None)
    dt = [dt0] + [900] * (n_slots - 1)
    n = len(dt)
    battery = BatteryConfig(
        charge_from_grid=True, discharge_to_grid=False,
        s_capacity=3000, s_min=0, s_max=3000, s_initial=0,
        c_min=0, c_max=6000, d_max=0, p_a=0.35, battery_first=battery_first,
    )
    time_series = TimeSeriesData(dt=dt, gt=[0] * n, ft=[0] * n, p_N=[0.30] * n, p_E=[0.10] * n)
    return Optimizer(OptimizationStrategy('none', 'none'), grid, [battery], time_series, eta_c=1.0, eta_d=1.0).solve()


def test_battery_first_keeps_short_first_slot_at_full_power():
    """
    evcc's core/site_optimizer.go used to read the charge intent from ChargingPower[1] instead
    of [0] because the first slot's charge was unreliable (see evcc-io/optimizer companion
    fork commit 8a69a797e): under a locally flat price, the solver could arbitrarily defer all
    charging past the short first slot into a later, full-length one. That workaround has been
    reverted in favour of battery_first, which must therefore keep the first slot's charge
    *power* (not just its energy, which necessarily shrinks with a shorter slot) at the full
    available rate throughout the slot's whole lifetime - reproducing the repeated re-solves
    (~every 2 minutes) that happen in real operation as dt[0] shrinks from a full slot down to
    a few seconds before the wall-clock quarter-hour boundary is crossed.
    """
    for dt0 in (900, 780, 540, 300, 60, 5):
        result = _solve_short_first_slot(dt0, battery_first=True)
        assert result['status'] == 'Optimal'

        energy0 = result['batteries'][0]['charging_power'][0]
        power0 = energy0 / (dt0 / 3600)
        assert np.isclose(power0, 6000, atol=1.0), (
            f"dt0={dt0}s: slot 0 should charge at the full available rate, got {power0:.1f} W"
        )


def test_battery_first_off_still_defers_short_first_slot():
    """Without battery_first, the degeneracy this feature fixes is still present - guards
    against the fix above becoming vacuous if something else started masking it."""
    result = _solve_short_first_slot(120, battery_first=False)
    assert result['status'] == 'Optimal'
    assert np.isclose(result['batteries'][0]['charging_power'][0], 0, atol=1.0)
