# LLaDA Implementation Log

## Files Created

- `networks/llada_encoder.py`
  - Added the COGITAO-adapted LLaDA masked diffusion encoder.
  - Ported RoPE, RMS/layer norm variants, LLaMA-style SwiGLU blocks, bidirectional SDPA attention, activation checkpointing hooks, stochastic target masking, and iterative denoising generation.
  - Added an internal LM head so the encoder returns full-sequence logits directly.

- `models/llada_model.py`
  - Added a standalone PyTorch Lightning module for LLaDA.
  - Implements target-only masked diffusion training, generated validation/test metrics, teacher-forced masked validation/test loss, optimizer parameter grouping, warmup, sample logging, and JSON test output handling.

- `configs/network/encoder/llada_encoder.yaml`
  - Added Hydra encoder architecture and diffusion defaults.

- `configs/model/llada_model.yaml`
  - Added Hydra model defaults for the LLaDA Lightning wrapper.

- `configs/sweeps/exp_sweep_llada.yaml`
  - Added a LLaDA WandB sweep entry matching the existing sweep style.

## Files Modified

- `models/model_helpers.py`
  - Registered `llada_encoder` in `_NETWORK_REGISTRY`.

- `main.py`
  - Imported `LLaDAModel`.
  - Added `llada_model` to `MODEL_MAP`.

## Deviations From Plan

- The LLaDA head uses `cfg.model.output_vocab_size` rather than the full input vocabulary by default. This follows the plan's mitigation for input/output vocabulary mismatch and prevents task/mask tokens from being normal training targets. If `sage_thinking=true`, the output vocabulary is expanded to include `thinking_token_id` because that mode can place the thinking token in `target_for_loss`.
- `predict_eos=false` remains the default. If enabled later, `LLaDAModel` expands `output_vocab_size` so the configured EOS token ID is representable.
- The sweep keeps `network/decoder=mlp_decoder` only to satisfy the existing Hydra defaults. `LLaDAModel` does not instantiate or use a decoder.
- No data module changes were made.
- No SLURM script was added because the local Python path and synthetic training checks were sufficient for this pass; the plan listed the SLURM script as optional after local validation.

## Verification

- Python compile check:
  - `uv run --project /cluster/raid/home/yassine/model_training_cogitao/compgen-reasoning python -m py_compile ...`
  - Result: passed.

- Hydra composition smoke test:
  - Composed `model=llada_model network/encoder=llada_encoder`.
  - Result: resolved `llada_model` and `llada_encoder`.

- Synthetic CPU forward/generation smoke test:
  - Used a tiny 2x2 batch with one LLaDA layer and `diffusion.steps=2`.
  - Result: full logits shape `(2, 11, 12)`, generated prediction shape `(2, 4)`, finite masked loss.

- Synthetic Lightning `fast_dev_run`:
  - Ran `Trainer(..., fast_dev_run=True, accelerator="cpu")` on a tiny in-memory DataLoader.
  - Result: passed training and validation, including iterative denoising validation.

- `torch.compile` compatibility smoke test:
  - Instantiated `LLaDAModel` with `training.use_torch_compile=true`.
  - Result: compiled encoder still exposes custom `mask_input_sequence()` and `generate_masked_sequence()` methods.

## Commits

- `38482e6 Add LLaDA masked diffusion model`
- `9237670 Handle LLaDA EOS vocabulary sizing`
- `be1f886 Document LLaDA implementation`
- Follow-up commit: `Update LLaDA verification log`
