# Post-hoc Projected-CLS Decoder

The decoder is a visualization-only model trained after predictor screening. It maps the frozen
official LeWM projected CLS latent to a 224x224 RGB reconstruction. Predictor parameters and the
official checkpoint are never optimized by this workflow.

The architecture follows the decoder described in Appendix D of the LeWM paper: a projected
global latent conditions 196 learned patch queries through cross-attention blocks. Each query is
projected to a 16x16 RGB patch and the patches are rearranged into the output image. The input is
the projected CLS-derived latent rather than the raw ViT CLS token because that is the space
predicted by every saved screening model.

The H200 decoder uses the complete predictor-training split and optimizes a combined L1 and LPIPS
perceptual objective for 100 fixed epochs. LPIPS uses its frozen VGG network and receives RGB
images scaled to `[-1, 1]` as required by the reference implementation. A linear warmup over the
first 1% of optimizer steps is followed by cosine learning-rate decay. Decoder optimization never
updates the encoder or any predictor. The resolved VGG checkpoint is SHA-256 recorded alongside
the official LeWM checkpoint.

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

The first `make sync` after this change installs `lpips==0.1.4`. Initializing VGG LPIPS may download
its pretrained torchvision weights if they are not already cached. Training fails rather than
silently falling back to another objective when those weights are unavailable.

Render decoded galleries from all 15 existing predictor checkpoints:

```bash
CUDA_VISIBLE_DEVICES=2 make render-h200-decoder-galleries
```

The existing nearest-neighbor retrieval galleries remain unchanged. Decoded galleries are actual
outputs of the post-hoc decoder, but they still reflect decoder reconstruction error as well as
predictor error.
