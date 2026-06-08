from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from common import AgentLog, ROOT, SOURCE_ROOT, first_existing_path


def resolve_inputs(log: AgentLog) -> tuple[Path, Path, dict[str, Any]]:
    model_config_path = first_existing_path(
        "/target/model_config.json",
        ROOT / "target" / "model_config.json",
        SOURCE_ROOT / "target" / "model_config.json",
    )
    weight_dir = first_existing_path(
        "/target/weights",
        ROOT / "target" / "weights",
        SOURCE_ROOT / "target" / "weights",
    )
    if model_config_path is None:
        raise FileNotFoundError("model_config.json not found under /target or target")
    if weight_dir is None:
        raise FileNotFoundError("weight directory not found under /target or target")

    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    ensure_weight_alias(weight_dir, log)
    log.log("probe", "resolved model inputs", {
        "model_config_path": model_config_path,
        "weight_dir": weight_dir,
    })
    return model_config_path, weight_dir, model_config


def ensure_weight_alias(weight_dir: Path, log: AgentLog) -> None:
    canonical = weight_dir / "model.pt"
    if canonical.exists():
        return
    alternatives = [
        weight_dir / "model_weights.pt",
        weight_dir / "pytorch_model.bin",
        weight_dir / "weights.pt",
    ]
    alternatives.extend(sorted(weight_dir.glob("*.pt")))
    source = next((p for p in alternatives if p.exists()), None)
    if source is None:
        log.log("probe", "no weight alias source found", {"weight_dir": weight_dir})
        return

    try:
        os.link(source, canonical)
        log.log("probe", "created hardlink weight alias", {"source": source, "alias": canonical})
        return
    except OSError as exc:
        log.log("probe", "hardlink weight alias failed", {"error": str(exc)})
    try:
        os.symlink(source.name, canonical)
        log.log("probe", "created symlink weight alias", {"source": source, "alias": canonical})
        return
    except OSError as exc:
        log.log("probe", "symlink weight alias failed", {"error": str(exc)})
    try:
        shutil.copy2(source, canonical)
        log.log("probe", "copied weight alias", {"source": source, "alias": canonical})
    except OSError as exc:
        log.log("probe", "could not create weight alias", {"error": str(exc), "source": source})


def find_weight_file(weight_dir: Path) -> Path | None:
    candidates = [
        weight_dir / "model.pt",
        weight_dir / "model_weights.pt",
        weight_dir / "pytorch_model.bin",
        weight_dir / "weights.pt",
    ]
    candidates.extend(sorted(weight_dir.glob("*.pt")))
    return next((p for p in candidates if p.exists()), None)


def load_state_dict_cpu(path: Path) -> dict[str, Any]:
    import torch

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "weights"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                obj = nested
                break
    if not isinstance(obj, dict):
        raise TypeError(f"weight file {path} did not contain a state dict")
    return obj


def probe_environment(model_config: dict[str, Any], weight_dir: Path, log: AgentLog) -> dict[str, Any]:
    summary: dict[str, Any] = {"model": {}, "weights": {}, "device": {}}
    hidden_size = int(model_config.get("hidden_size", 0) or 0)
    num_heads = int(model_config.get("num_attention_heads", 0) or 0)
    head_dim = int(model_config.get("head_dim", hidden_size // num_heads if num_heads else 0) or 0)
    summary["model"] = {
        "num_hidden_layers": model_config.get("num_hidden_layers"),
        "hidden_size": model_config.get("hidden_size"),
        "num_attention_heads": model_config.get("num_attention_heads"),
        "num_key_value_heads": model_config.get("num_key_value_heads"),
        "head_dim": head_dim,
        "vocab_size": model_config.get("vocab_size"),
        "torch_dtype": model_config.get("torch_dtype"),
        "intermediate_size": model_config.get("intermediate_size"),
        "max_position_embeddings": model_config.get("max_position_embeddings"),
    }

    try:
        import torch

        summary["device"]["torch_version"] = str(torch.__version__)
        summary["device"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            summary["device"].update({
                "cuda_device": torch.cuda.get_device_name(0),
                "cuda_free_gb": round(free / 1024**3, 3),
                "cuda_total_gb": round(total / 1024**3, 3),
            })
        try:
            import triton  # type: ignore

            summary["device"]["triton_available"] = True
            summary["device"]["triton_version"] = getattr(triton, "__version__", "unknown")
        except Exception:
            summary["device"]["triton_available"] = False

        weight_path = find_weight_file(weight_dir)
        if weight_path is not None:
            state_dict = load_state_dict_cpu(weight_path)
            tensor_summaries = []
            total_params = 0
            dtype_counts: dict[str, int] = {}
            for name, tensor in state_dict.items():
                shape = tuple(int(x) for x in tensor.shape)
                numel = int(tensor.numel())
                dtype = str(tensor.dtype)
                total_params += numel
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + numel
                tensor_summaries.append({
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "numel": numel,
                })
            summary["weights"] = {
                "weight_path": weight_path,
                "tensor_count": len(tensor_summaries),
                "total_params": total_params,
                "dtype_param_counts": dtype_counts,
                "first_tensors": tensor_summaries[:24],
            }
    except Exception as exc:
        summary["device"]["probe_error"] = repr(exc)
        log.log("probe", "environment probe hit an error", {"error": repr(exc)})

    log.log("probe", "environment and model summary", summary)
    return summary
