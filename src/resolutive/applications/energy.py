"""Synthetic energy-system benchmarks for black-box optimization.

The models in this module are deterministic engineering test problems, not
high-fidelity photovoltaic device simulators. They are intended to test whether
an optimizer can locate high-power operating points under smooth and partially
shaded response surfaces while respecting the same objective-call budget used
elsewhere in the project.
"""

from __future__ import annotations

import numpy as np


def decode_duty(x: np.ndarray) -> float:
    """Map a normalized scalar decision variable in [-1, 1] to duty in [0.05, 0.95]."""
    x = np.asarray(x, dtype=float)
    if x.shape != (1,):
        raise ValueError("MPPT decision vector must have shape (1,)")
    z = float(np.clip((x[0] + 1.0) / 2.0, 0.0, 1.0))
    return 0.05 + 0.90 * z


def synthetic_pv_power(
    duty: float,
    *,
    irradiance: float = 1.0,
    temperature_c: float = 25.0,
    partial_shading: bool = False,
) -> float:
    """Return a deterministic synthetic PV power proxy.

    The duty cycle is mapped to an effective operating voltage. The clear-sky
    surface has one dominant maximum. The partial-shading surface adds a second
    local maximum to create a more difficult MPPT search landscape.
    """
    duty = float(np.clip(duty, 0.05, 0.95))
    irradiance = float(np.clip(irradiance, 0.05, 1.2))
    temperature_c = float(temperature_c)

    # Higher duty maps to lower effective panel voltage for this synthetic
    # converter convention.
    voltage = 42.0 * (1.0 - 0.82 * duty)
    temperature_factor = max(0.65, 1.0 - 0.0035 * (temperature_c - 25.0))

    # Main operating lobe around the nominal maximum-power region.
    main = 235.0 * np.exp(-0.5 * ((voltage - 30.0) / 5.5) ** 2)
    shoulder = 28.0 * np.exp(-0.5 * ((voltage - 20.0) / 8.0) ** 2)
    power = irradiance * temperature_factor * (main + shoulder)

    if partial_shading:
        # A competing local maximum emulates the multi-peak character often
        # encountered in partially shaded arrays without claiming device-level
        # physical fidelity.
        shaded_peak = 0.78 * 235.0 * np.exp(-0.5 * ((voltage - 17.0) / 3.6) ** 2)
        power = max(power * 0.72, irradiance * temperature_factor * shaded_peak)

    return float(max(0.0, power))


def mppt_clear_sky_cost(x: np.ndarray) -> float:
    """Minimization objective equivalent to maximizing clear-sky PV power."""
    duty = decode_duty(x)
    return -synthetic_pv_power(duty)


def mppt_partial_shading_cost(x: np.ndarray) -> float:
    """Minimization objective for a synthetic multi-peak partial-shading case."""
    duty = decode_duty(x)
    return -synthetic_pv_power(duty, irradiance=0.82, temperature_c=38.0, partial_shading=True)
