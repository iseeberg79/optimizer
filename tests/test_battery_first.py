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
