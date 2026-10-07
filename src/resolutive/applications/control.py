"""Small reproducible control problem for black-box optimizer evaluation.

This module is an engineering benchmark, not a hardware controller. A normalized
3-vector is mapped to PID gains and evaluated on a deterministic second-order
plant simulation with actuator saturation.
"""

from __future__ import annotations

import numpy as np


def decode_pid(x: np.ndarray) -> tuple[float, float, float]:
    """Map normalized decision variables in [-1, 1] to PID gains."""
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError("PID decision vector must have shape (3,)")
    z = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
    kp = 12.0 * z[0]
    ki = 6.0 * z[1]
    kd = 3.0 * z[2]
    return float(kp), float(ki), float(kd)


def pid_step_response_cost(x: np.ndarray) -> float:
    """Evaluate PID gains on a saturated second-order plant.

    Plant: y'' + 1.4 y' + y = u, integrated by semi-implicit Euler.
    Cost combines tracking error, overshoot, settling tail, and control effort.
    Lower is better.
    """
    kp, ki, kd = decode_pid(x)
    dt = 0.01
    steps = 800
    reference = 1.0
    max_u = 6.0

    y = 0.0
    velocity = 0.0
    integral = 0.0
    previous_error = reference
    absolute_error = 0.0
    control_effort = 0.0
    overshoot = 0.0
    tail_error = 0.0

    for step in range(steps):
        error = reference - y
        integral = np.clip(integral + error * dt, -3.0, 3.0)
        derivative = (error - previous_error) / dt
        u = kp * error + ki * integral + kd * derivative
        u = float(np.clip(u, -max_u, max_u))

        acceleration = u - 1.4 * velocity - y
        velocity += acceleration * dt
        y += velocity * dt

        absolute_error += abs(error) * dt
        control_effort += (u / max_u) ** 2 * dt
        overshoot = max(overshoot, max(0.0, y - reference))
        if step >= steps - 100:
            tail_error += abs(error) * dt
        previous_error = error

    return float(
        absolute_error
        + 4.0 * overshoot**2
        + 2.0 * tail_error
        + 0.03 * control_effort
    )
