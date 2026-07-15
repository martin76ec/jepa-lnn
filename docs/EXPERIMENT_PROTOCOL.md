# Experiment Protocol

This repository implements the plan in the parent LeWM workspace: evaluate stateful CfC and LTC predictors against MLP and causal Transformer predictors without changing the encoder, latent space, dataset split, objective, or evaluation procedure.

The primary outcome is normalized latent error in a closed-loop rollout at the pre-registered horizon. One-step teacher-forced error is diagnostic only. Secondary outcomes include rollout curves, divergence rate, control performance, robustness under pre-defined perturbations, latent-structure diagnostics, parameters, latency, throughput, and memory.

The local profile is restricted to a small data fraction and exists only to prove data flow, numerical correctness, logging, and test coverage. All reported metrics must use the full H200 profile with the shared seed set and pre-registered configuration.

The predictor interface must represent recurrent state explicitly:

```python
state = predictor.init_state(batch_size, device)
z_next, state = predictor.step(z_t, action_t, state, dt)
z_pred, final_state = predictor.rollout(z_0, actions, state=state, dt=dt)
```

Before training candidates, finalize the environments, split manifests, rollout horizons, divergence threshold, capacity-matching bounds, compute budget, and stopping criterion.
