from __future__ import annotations


def extract_runtime_spec() -> list[str]:
    return [
        "/workspace/engine.py must define create_engine(model_config, weight_dir, device='cuda').",
        "create_engine must return an object with prefill(request_ids, input_ids), decode(request_ids, token_ids), and remove(request_ids).",
        "prefill creates or replaces only the listed request states and returns last-token logits in request_ids order.",
        "decode only accepts already-prefilled requests, appends one token per request, and returns logits in request_ids order.",
        "remove releases the listed request states without disturbing unrelated requests.",
        "All model dimensions and dtype choices must be read dynamically from model_config and weights.",
        "Correctness is a hard gate; performance is considered only for candidates that pass correctness and stress tests.",
        "The final engine.py must be copied from the best passing candidate, never from an untested final iteration.",
    ]
