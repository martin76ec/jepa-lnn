# Post-hoc Projected-CLS Decoder

The decoder is a visualization-only model trained after predictor screening. It maps the frozen
official LeWM projected CLS latent to a 224x224 RGB reconstruction. Predictor parameters and the
official checkpoint are never optimized by this workflow.

The architecture follows the decoder described in Appendix D of the LeWM paper: a projected
global latent conditions 196 learned patch queries through cross-attention blocks. Each query is
projected to a 16x16 RGB patch and the patches are rearranged into the output image. The input is
the projected CLS-derived latent rather than the raw ViT CLS token because that is the space
predicted by every saved screening model.

## Artifact isolation

Predictor checkpoints remain under:

```text
runs/h200-screen/<variant>/run_<timestamp>_seed<seed>/system.pt
```

Decoder artifacts use a separate namespace:

```text
runs/h200-decoder/run_<timestamp>_seed<seed>/
  decoder.pt
  source_checkpoints.json
  galleries/render_<timestamp>/<variant>/<predictor-run>/
```

The decoder saver only accepts the filename `decoder.pt`, writes through `decoder.pt.tmp`, and
refuses to overwrite an existing file. Gallery rendering loads `system.pt` files for inference
only and writes outside the predictor run tree.

## H200 commands

Train only the decoder:

```bash
CUDA_VISIBLE_DEVICES=2 make train-h200-decoder
```

Render decoded galleries from all 15 existing predictor checkpoints:

```bash
CUDA_VISIBLE_DEVICES=2 make render-h200-decoder-galleries
```

The existing nearest-neighbor retrieval galleries remain unchanged. Decoded galleries are actual
outputs of the post-hoc decoder, but they still reflect decoder reconstruction error as well as
predictor error.
