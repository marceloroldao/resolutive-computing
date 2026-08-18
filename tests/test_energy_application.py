import numpy as np

from resolutive.applications.energy import (
    decode_duty,
    mppt_clear_sky_cost,
    mppt_partial_shading_cost,
    synthetic_pv_power,
)


def test_decode_duty_maps_normalized_interval():
    assert decode_duty(np.array([-1.0])) == 0.05
    assert decode_duty(np.array([1.0])) == 0.95


def test_clear_sky_objective_is_negative_power():
    x = np.array([0.0])
    duty = decode_duty(x)
    assert mppt_clear_sky_cost(x) == -synthetic_pv_power(duty)


def test_partial_shading_surface_differs_from_clear_sky():
    x = np.array([0.4])
    assert mppt_partial_shading_cost(x) != mppt_clear_sky_cost(x)


def test_synthetic_power_is_deterministic_and_nonnegative():
    first = synthetic_pv_power(0.42, irradiance=0.8, temperature_c=40.0, partial_shading=True)
    second = synthetic_pv_power(0.42, irradiance=0.8, temperature_c=40.0, partial_shading=True)
    assert first == second
    assert first >= 0.0
