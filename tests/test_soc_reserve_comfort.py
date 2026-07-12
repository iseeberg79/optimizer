import numpy as np

from optimizer.optimizer import (
    BatteryConfig,
    GridConfig,
    OptimizationStrategy,
    Optimizer,
    TimeSeriesData,
)


def _solve(prc_dpl_soc_low, p_a, p_E):
    """
    1000 Wh battery at 600 Wh (60%), floor s_min=50 (5%), no load, may export. The only choice is
    how far to discharge to grid. The low-SOC reserve-comfort cost ramps from zero at 20% (200 Wh)
    to full at s_min, so it discourages draining through the 20% band - but only softly.
    """
    grid = GridConfig(p_max_imp=None, p_max_exp=None, prc_p_exc_imp=None)
    battery = BatteryConfig(
        charge_from_grid=False, discharge_to_grid=True,
        s_capacity=1000, s_min=50, s_max=1000, s_initial=600,
        c_min=0, c_max=0, d_max=5000, p_a=p_a, prc_dpl_soc_low=prc_dpl_soc_low,
    )
    time_series = TimeSeriesData(dt=[3600, 3600], gt=[0, 0], ft=[0, 0], p_N=[0.3, 0.3], p_E=[p_E, p_E])
    return Optimizer(OptimizationStrategy('none', 'none'), grid, [battery], time_series, eta_c=1.0, eta_d=1.0).solve()


def test_low_soc_comfort_holds_the_band():
    """Exporting is only marginally better than holding (below the in-band comfort cost): the battery
    discharges down to the 20% floor and then holds - it is not drained to s_min for a tiny gain."""
    result = _solve(prc_dpl_soc_low=0.02, p_a=0.10, p_E=0.10001)

    assert result['status'] == 'Optimal'
    soc = result['batteries'][0]['state_of_charge']
    assert np.isclose(soc[-1], 200, atol=1.0), "battery should hold the 20% comfort floor, not drain to s_min"


def test_low_soc_comfort_yields_to_real_value():
    """Exporting is clearly profitable (p_E >> p_a): the small comfort cost must not block real
    value - the battery still drains all the way to s_min."""
    result = _solve(prc_dpl_soc_low=0.002, p_a=0.10, p_E=0.12)

    assert result['status'] == 'Optimal'
    soc = result['batteries'][0]['state_of_charge']
    assert np.isclose(soc[-1], 50, atol=1.0), "real export value should still drain to s_min"


def test_low_soc_comfort_inactive_when_zero():
    """With the cost disabled (0) the band is not defended: the marginally-better export drains it."""
    result = _solve(prc_dpl_soc_low=0.0, p_a=0.10, p_E=0.10001)

    assert result['status'] == 'Optimal'
    soc = result['batteries'][0]['state_of_charge']
    assert np.isclose(soc[-1], 50, atol=1.0), "without the comfort cost the battery drains to s_min"
