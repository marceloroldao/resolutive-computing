# Applications of Resolutive Computing

Resolutive Computing targets computational problems in which a system must search a large space of possible configurations while minimizing the number or cost of evaluations.

## Target application domains

- Engineering design and parameter optimization
- Industrial process and energy optimization
- Simulation-based optimization
- Expensive black-box optimization, where each evaluation may require significant computation or a physical experiment
- Parameter tuning for software and computational systems
- Resource allocation and scheduling
- Planning and trajectory optimization for autonomous systems
- Network and distributed-system optimization
- Multi-agent optimization
- Scientific parameter estimation and computational experiments

## Future application domains

Routing, robotics, IoT and embedded optimization are candidate future applications after the corresponding algorithms and implementations have been independently benchmarked. The current project priority is PC/server execution and reproducible validation.

## Conceptual optimization loop

```text
Problem
  |
  v
Candidate configurations
  |
  v
Resolutive Optimizer
  |
  v
Evaluate candidates
  |
  v
Feedback
  |____ repeat ____|
  |
  v
Best configuration found
```

These are target application domains, not claims of demonstrated superiority. Current evidence is limited to reproducible benchmarks published in this repository. Results should be compared under equal evaluation budgets, multiple random seeds and recorded configurations.
