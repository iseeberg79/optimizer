import numpy as np

from optimizer.optimizer import (
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
)


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
