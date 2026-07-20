# Experiment Inputs

## Supplied baseline reference

| Field | Value |
| --- | --- |
| Repository | `https://github.com/lucas-maes/le-wm.git` |
| Commit | `c8a44170b22dfe15b0b1924e2845da97af926c42` |
| Candidate predictors | MLP, Transformer, CfC, LTC |
| Selected environment | PushT |
| Upstream training dataset identifier | `pusht_expert_train.lance` |

The working assumption from the implementation plan is direct predictor substitution: retain the encoder, latent space, data splits, and training objective while replacing the dynamics predictor. This must be verified against the pinned upstream source before implementation begins.

The pinned upstream training config identifies a Lance dataset and uses `frameskip: 5`.
The adapter preserves this behavior by flattening five raw action rows per downsampled
visual step and discarding the terminal action block that has no following observation.

CfC and LTC use the maintained `ncps` source pinned at commit
`695ec22d7d1831f2ced67346ef7d2f08c525bd47` (version 1.0.1).

## Inputs required before full experimentation

### Dataset and representation

- Download the upstream PushT Lance archive from the LeWM Hugging Face data collection and store `pusht_expert_train.lance` under `$STABLEWM_HOME`.
- H200 dataset storage/mount path and download command.
- Trajectory-level train/validation/test policy, including held-out initial conditions or dynamics regimes.
- Sequence length, action encoding and shape, latent dimension, and timestep `dt`.

### Evaluation protocol

- Closed-loop rollout horizons: `H = [1, 5, 10, 20, 50]` predictor steps; `H=20` is the primary horizon.
- Divergence: normalized latent error greater than `10.0`, or any NaN/Inf value; these rules are fixed before test evaluation.
- Planning/control metrics and perturbation suites with severity levels.

### Training protocol

- Batch size, maximum epochs, optimizer, scheduler, and early-stopping criterion.
- Five shared random seeds.
- Parameter-count matching tolerance and equal hyperparameter-search budget.
- H200 determinism requirement and permitted trade-offs with throughput.

### H200 environment

- CUDA version: 12.8.
- Access method, PyTorch build, and approved package versions.
- Dataset storage/mount path and any download or access credentials.
- Experiment tracking requirements such as Weights & Biases project/entity, or an offline-only policy.

### Deliverables

- Required training and evaluation CLI commands.
- Reproduction scripts and configuration manifests.
- Required metrics tables and plots.
- Whether the final handoff is runnable code only or also includes a paper-style report.
