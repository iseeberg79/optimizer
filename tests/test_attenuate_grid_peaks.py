import numpy as np

from optimizer.optimizer import (
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
)


def _solve_shaping(charging_strategy, p_a):
    """Solar hump ft=[0, 500, 1500, 500, 0] (total 2500), no load, a 1000 Wh battery."""
    grid = GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None)
    battery = BatteryConfig(
        charge_from_grid=False, discharge_to_grid=False,
        s_capacity=1000, s_min=0, s_max=1000, s_initial=0,
        c_min=0, c_max=5000, d_max=0, p_a=p_a,
    )
    time_series = TimeSeriesData(
        dt=[3600, 3600, 3600, 3600, 3600], gt=[0, 0, 0, 0, 0], ft=[0, 500, 1500, 500, 0],
        p_N=[0.3, 0.3, 0.3, 0.3, 0.3], p_E=[0.1, 0.1, 0.1, 0.1, 0.1],
    )
    strategy = OptimizationStrategy(charging_strategy, 'none')
    return Optimizer(strategy, grid, [battery], time_series, eta_c=1.0, eta_d=1.0).solve()


def test_attenuate_flattens_feed_in_and_fills():
    """attenuate_grid_peaks fills the 1000 Wh battery from surplus and flattens the residual feed-in:
    2500 Wh surplus - 1000 stored = 1500 exported, spread toward the ~500 mean so the raw 1500 Wh
    export peak is cut well below half (PWL granularity leaves a small residual)."""
    attenuated = _solve_shaping('attenuate_grid_peaks', p_a=0.1)

    assert attenuated['status'] == 'Optimal'
    assert np.isclose(attenuated['batteries'][0]['state_of_charge'][-1], 1000, atol=1e-2), "battery is filled"
    assert np.max(attenuated['grid_export']) <= 600, "residual feed-in peak is strongly flattened"


def test_attenuate_shaping_does_not_override_economics():
    """The shaping is a secondary tie-breaker: when exporting is strictly better (p_a < p_E), it
    must not force uneconomic storing - the battery stays empty and the peak is exported."""
    result = _solve_shaping('attenuate_grid_peaks', p_a=0.05)

    assert result['status'] == 'Optimal'
    assert np.isclose(result['batteries'][0]['state_of_charge'][-1], 0, atol=1e-2), "filling not forced"
    assert np.isclose(np.max(result['grid_export']), 1500, atol=1e-2), "peak exported when cheaper"


def test_attenuate_absorbs_curtailment_under_export_limit():
    """A 13 kW midday peak against a 10 kW export limit would curtail 2.5 kWh. Under
    attenuate_grid_peaks the battery absorbs the over-limit surplus, driving the curtailment to
    zero and storing the energy instead of losing it (the SellToGridLimit behaviour)."""
    grid = GridConfig(p_max_imp=None, p_max_exp=10000, prc_p_exc_imp=None)
    pa = min([0.25e-3] * 5) * 0.9 * 0.99
    battery = BatteryConfig(
        charge_from_grid=False, discharge_to_grid=False,
        s_capacity=10000, s_min=0, s_max=10000, s_initial=0,
        c_min=0, c_max=3000, d_max=0, p_a=pa,
    )
    time_series = TimeSeriesData(
        dt=[3600] * 5, gt=[500] * 5, ft=[2000, 8000, 13000, 8000, 2000],
        p_N=[0.25e-3] * 5, p_E=[0.08e-3] * 5,
    )
    strategy = OptimizationStrategy('attenuate_grid_peaks', 'none')
    result = Optimizer(strategy, grid, [battery], time_series, eta_c=0.9, eta_d=0.9).solve()

    assert result['status'] == 'Optimal'
    assert np.max(result['grid_export_overshoot']) <= 1e-2, "over-limit peak absorbed, not curtailed"
    assert np.max(result['grid_export']) <= 10000 + 1e-2, "feed-in stays within the export limit"
