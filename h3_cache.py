from __future__ import annotations

import logging
import types
from typing import Any

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax import model as minimax_model

REUSE_THRESHOLD = 0.15
START_PERCENT = 0.15
END_PERCENT = 0.95
MAX_STEPS = 2
DEVICE = "auto"
VERBOSE = True


class _Cache:
    def __init__(self):
        self.reuse_threshold = REUSE_THRESHOLD
        self.start_percent = START_PERCENT
        self.end_percent = END_PERCENT
        self.max_steps = MAX_STEPS
        self.device = DEVICE
        self.verbose = VERBOSE
        self.total_steps = 1
        self.reset()

    def reset(self):
        self.cached_residual = None
        self.previous_feature_signature = None
        self.layout_signature = None
        self.last_seen_timestep = None
        self.step_counter = 0
        self.accumulated_relative_l1 = 0.0
        self.consecutive_skips = 0
        self.run_count = 0
        self.skip_count = 0

    def begin(self, total_steps):
        self.reset()
        self.total_steps = max(1, total_steps)

    def finish(self):
        if self.verbose and self.run_count + self.skip_count:
            total = self.run_count + self.skip_count
            logging.info(
                "[H3 Cache] Skipped %s/%s block-stack executions (%.2fx).",
                self.skip_count, total, total / max(1, self.run_count),
            )
        self.reset()

    @staticmethod
    def _feature_signature(hidden_states, cache_ranges):
        max_dim = min(64, hidden_states.shape[-1])
        signatures = []
        for start, end in cache_ranges:
            length = end - start
            if length <= 0:
                continue
            stride = max(1, length // 100)
            signatures.append(hidden_states[start:end:stride, :max_dim].detach().abs().mean(dim=-1))
        if not signatures:
            stride = max(1, hidden_states.shape[0] // 100)
            return hidden_states[::stride, :max_dim].detach().abs().mean(dim=-1).clone()
        return torch.cat(signatures).clone()

    @staticmethod
    def _timestep_value(timestep):
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 0:
                return None
            return float(timestep.detach().flatten()[0].item())
        if isinstance(timestep, (int, float)):
            return float(timestep)
        return None

    def _store_residual(self, residual):
        if self.device == "cuda" and residual.device.type != "cuda":
            raise ValueError("H3 cache device is cuda but model is not running on CUDA.")
        try:
            if self.device == "cpu":
                self.cached_residual = residual.detach().to("cpu", copy=True)
            else:
                self.cached_residual = residual.detach().clone()
        except torch.OutOfMemoryError:
            if self.device == "cuda":
                raise
            self.cached_residual = residual.detach().to("cpu", copy=True)

    def _apply_residual(self, hidden_states):
        residual = self.cached_residual
        if residual is None:
            return hidden_states
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(device=hidden_states.device, dtype=hidden_states.dtype, non_blocking=True)
        return hidden_states + residual

    def __call__(self, args, extra_options):
        original_block = extra_options["original_block"]
        hidden_states = args["img"]
        cache_ranges = tuple(tuple(pair) for pair in args.get("cache_ranges", ()))
        current_layout = (
            tuple(hidden_states.shape), hidden_states.dtype, hidden_states.device,
            args.get("block_count"), cache_ranges,
        )

        if self.layout_signature is None:
            self.layout_signature = current_layout
        elif self.layout_signature != current_layout:
            total_steps = self.total_steps
            self.reset()
            self.total_steps = total_steps
            self.layout_signature = current_layout

        timestep = self._timestep_value(args.get("timestep"))
        if timestep is None:
            return original_block(args)
        if self.last_seen_timestep != timestep:
            self.last_seen_timestep = timestep
            self.step_counter += 1

        progress = self.step_counter / self.total_steps
        in_cache_range = self.start_percent <= progress <= self.end_percent
        skip_reason = "initial step"

        if self.cached_residual is not None and self.previous_feature_signature is not None:
            current_float = self._feature_signature(hidden_states, cache_ranges).float()
            previous_float = self.previous_feature_signature.float()
            difference = (current_float - previous_float).abs().mean().item()
            denominator = previous_float.abs().mean().item() + 1e-6
            self.accumulated_relative_l1 += difference / denominator

            below_threshold = self.accumulated_relative_l1 < self.reuse_threshold
            below_skip_limit = self.consecutive_skips < self.max_steps
            if below_threshold and below_skip_limit and in_cache_range:
                self.skip_count += 1
                self.consecutive_skips += 1
                if self.verbose:
                    logging.info(
                        "[H3 Cache] Step %s SKIP (relative L1 %.4f < %.4f).",
                        self.step_counter, self.accumulated_relative_l1, self.reuse_threshold,
                    )
                return {"img": self._apply_residual(hidden_states)}

            reasons = []
            if not below_threshold:
                reasons.append("threshold reached")
            if not below_skip_limit:
                reasons.append("max consecutive skips reached")
            if not in_cache_range:
                reasons.append("outside cache range")
            skip_reason = ", ".join(reasons)

        if self.verbose:
            logging.info("[H3 Cache] Step %s RUN (%s).", self.step_counter, skip_reason)

        self.run_count += 1
        self.consecutive_skips = 0
        self.cached_residual = None
        self.previous_feature_signature = self._feature_signature(hidden_states, cache_ranges)
        start_hidden_states = hidden_states.clone()
        result = original_block(args)
        self._store_residual(result["img"] - start_hidden_states)
        self.accumulated_relative_l1 = 0.0
        return result


class _SamplingScope:
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, sample_fn, *args, **kwargs):
        sigmas = kwargs.get("sigmas")
        if sigmas is None and len(args) > 3:
            sigmas = args[3]
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
            self.cache.reset()
            raise ValueError("H3 cache could not read the sampler sigma schedule.")
        self.cache.begin(len(sigmas) - 1)
        try:
            return sample_fn(*args, **kwargs)
        finally:
            self.cache.finish()


def _run_blocks(model, hidden_states, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    end = len(model.blocks) if end is None else end
    blocks = list(model.blocks[start:end])
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(blocks, hidden_states.device, transformer_options)
    for index in range(start, end):
        block = model.blocks[index]
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, hidden_states.device, block)
        if ("double_block", index) in blocks_replace:
            def block_wrapper(block_args, block=block):
                return {"img": block(
                    block_args["img"], block_args["t_emb"], block_args["mod_segments"],
                    block_args["rope_freqs"], transformer_options=block_args["transformer_options"],
                )}
            hidden_states = blocks_replace[("double_block", index)](
                {"img": hidden_states, "t_emb": t_emb, "mod_segments": mod_segments,
                 "rope_freqs": rope_freqs, "transformer_options": transformer_options},
                {"original_block": block_wrapper},
            )["img"]
        else:
            hidden_states = block(hidden_states, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, hidden_states.device, None)
    return hidden_states


def _block_patch_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
    del kwargs
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2:5]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2:5]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = minimax_model.PackedLayout(
            text_len, latent_t, lat_h, lat_w, audio_t,
            keyframes=payload.get("keyframes"), refs=payload.get("refs"),
        )

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP))
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
    seg_t = {
        "text": t_v, "video": t_v, "audio": t_a,
        "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug), "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set()) | ({seg_t["ref_audio"]} if has_aud_cond else set()))
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, end, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, end - start + 1):
                if index == end - start or tags[index] != tags[run_start]:
                    mod_segments.append((start + run_start, start + index, row_base + int(tags[run_start])))
                    run_start = index
        else:
            mod_segments.append((start, end, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(self.condition_proj(text_states), transformer_options=transformer_options)

    hidden_states = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    video_offset = audio_offset = 0
    for start, end, kind in layout.segments:
        length = end - start
        if kind == "text":
            hidden_states[start:end] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden_states[start:end] = video_embed[video_offset:video_offset + length]
            video_offset += length
        else:
            hidden_states[start:end] = audio_embed[audio_offset:audio_offset + length]
            audio_offset += length

    t_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        position = t_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    else:
        t_emb = self.time_embedder(t_values).to(dtype)

    rope_freqs = minimax_model.rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    cache_ranges = tuple((start, end) for start, end, kind in layout.segments if kind in ("audio", "video"))

    if ("block_loop", 0) in blocks_replace:
        def block_loop_wrapper(block_args):
            return {"img": _run_blocks(
                self, block_args["img"], block_args["t_emb"], block_args["mod_segments"],
                block_args["rope_freqs"], block_args["transformer_options"],
                block_args.get("start", 0), block_args.get("end"),
            )}
        hidden_states = blocks_replace[("block_loop", 0)](
            {
                "img": hidden_states, "timestep": timestep, "t_emb": t_emb,
                "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                "transformer_options": transformer_options, "cache_ranges": cache_ranges,
                "target_ranges": tuple((start, end, kind) for start, end, kind in layout.segments if kind in ("audio", "video")),
                "block_count": len(self.blocks),
            },
            {"original_block": block_loop_wrapper},
        )["img"]
    else:
        hidden_states = _run_blocks(self, hidden_states, t_emb, mod_segments, rope_freqs, transformer_options)

    video_seg = next((start, end, t_row[seg_t["video"]]) for start, end, kind in layout.segments if kind == "video")
    audio_seg = next((start, end, t_row[seg_t["audio"]]) for start, end, kind in layout.segments if kind == "audio")
    video_result, audio_result = self.final_layer(
        hidden_states, t_emb, video_seg, audio_seg, sigma_v,
        transformer_options.get("sample_sigmas"), (shift_v, shift_a),
    )
    video_out = minimax_model.unpatchify_video(video_result, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = minimax_model.unpack_audio(audio_result)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def patch_h3_cache(model: Any) -> Any:
    patched_model = model.clone()
    diffusion_model = patched_model.model.diffusion_model
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError(f"H3 Cache requires a MiniMax H3 diffusion model; received {diffusion_model.__class__.__name__}.")

    cache = _Cache()
    patched_model.add_object_patch("diffusion_model._forward", types.MethodType(_block_patch_forward, diffusion_model))
    patched_model.set_model_patch_replace(cache, "dit", "block_loop", 0)
    patched_model.add_wrapper(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _SamplingScope(cache))
    return patched_model
