"""Move unused frozen decoder weights off GPU during visual encoding.

This preserves the complete vision sequence and its original kernel shapes.
The decoder is restored before the encoder returns; no trainable weights move.
"""
import os
from types import MethodType

import torch


def _cuda_available_bytes(device):
    free, _ = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    cached = (stats.get("reserved_bytes.all.current", 0)
              - stats.get("active_bytes.all.current", 0)
              - stats.get("inactive_split_bytes.all.current", 0))
    return free + max(0, cached)


def offload_frozen_decoder_for_vision(model, keep_decoder_free_gib=0):
    decoder = model._prem_text_model
    if any(parameter.requires_grad for parameter in decoder.parameters()):
        raise ValueError("Decoder offloading requires a fully frozen language model")
    original = model._prem_encode_video
    release_cache = os.environ.get("PREM_RELEASE_PINNED_HOST_CACHE", "0")
    if release_cache not in {"0", "1"}:
        raise ValueError("PREM_RELEASE_PINNED_HOST_CACHE must be 0 or 1")
    empty_host_cache = None
    if release_cache == "1":
        # PyTorch 2.11 has no public host-cache release API. This releases only
        # unused pinned blocks; saved tensors belonging to live graphs survive.
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_host_cache is None:
            raise RuntimeError("This PyTorch build cannot release the pinned host cache")
        print("[memory] pinned_host_cache=release_before_visual_encode "
              "api=torch._C._host_emptyCache", flush=True)

    def encode(self, pixels, grid):
        device = next(decoder.parameters()).device
        if device.type != "cuda":
            return original(pixels, grid)
        if empty_host_cache is not None:
            empty_host_cache()
        if (keep_decoder_free_gib
                and _cuda_available_bytes(device) >= keep_decoder_free_gib * 1024**3):
            return original(pixels, grid)
        decoder.to(device="cpu")
        try:
            encoded = original(pixels, grid)
        finally:
            decoder.to(device=device)
        return encoded

    model._prem_encode_video = MethodType(encode, model)


def save_answer_activations(device, gpu_budget_gib=0):
    """Keep bounded contiguous copies on CUDA, then use the original CPU packer."""
    cpu_offload = torch.autograd.graph.save_on_cpu(pin_memory=True)
    device = torch.device(device)
    if not gpu_budget_gib or device.type != "cuda":
        return cpu_offload

    remaining = None

    def pack(tensor):
        nonlocal remaining
        if tensor.device.type != "cuda" or tensor.layout != torch.strided:
            return cpu_offload.pack_hook(tensor)
        if remaining is None:
            free_bytes = _cuda_available_bytes(tensor.device)
            remaining = min(gpu_budget_gib * 1024**3, max(0, free_bytes - 20 * 1024**3))
        size_bytes = tensor.numel() * tensor.element_size()
        if size_bytes > remaining:
            return cpu_offload.pack_hook(tensor)

        # Match save_on_cpu's fresh, contiguous snapshot, including for views.
        stored = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        stored.copy_(tensor)
        remaining -= size_bytes
        return tensor.device, stored

    return torch.autograd.graph.saved_tensors_hooks(pack, cpu_offload.unpack_hook)
