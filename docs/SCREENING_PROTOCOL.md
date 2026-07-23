# Budgeted Predictor Screening Protocol

This screening study tests predictor dynamics under a fixed compute budget. It is not the full
reported LeWM training protocol.

## Fixed representation

Every candidate uses an identical frozen copy of the encoder, projector, and action encoder from
the pinned official PushT LeWM checkpoint. Only predictor parameters are optimized. This gives all
candidates the same latent targets and prevents predictor-dependent encoder drift or collapse.

The official representation was pretrained on random windows from the same source dataset,
including episodes assigned to the screening test split. The study is therefore a transductive
comparison of predictors in a fixed pretrained latent space, not a test of representation
generalization to unseen episodes.

## Candidate fitting

- deterministic episode-level 80/10/10 split with seed 7;
- deterministic 10% sample of training episodes;
- 10 epochs and batch size 128;
- shared official raw-action normalization;
- shared next-latent MSE objective;
- seeds 7, 19, and 43;
- LeWM-AR, MLP, Transformer, CfC, and LTC predictors.

## Evaluation

The primary screening metric is normalized latent MSE at closed-loop horizon 20. It is computed
per trajectory and then averaged across trajectories. One-step error, other rollout horizons,
divergence rate, and median first-divergence time are diagnostics.

The complete official LeWM checkpoint is evaluated separately as an in-dataset pretrained
reference. It is not a held-out baseline because its original training split used windows from the
same source episodes.

Retrieval galleries exclude every frame from the query episode. Their middle column is a nearest
real-frame proxy for a predicted latent, not a generated image.
