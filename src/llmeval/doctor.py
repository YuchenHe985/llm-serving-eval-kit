"""Diagnose LLM-serving startup failures from logs, using a catalog of signatures.

The catalog is built from failures met while deploying SGLang on 4x RTX 4090 (PCIe) and
4x A100 (NVLink) machines; ``docs/failure-catalog.md`` lists each with the root cause and
the fix that worked. Messages captured verbatim in the run logs are used as they were
(CUDA image vs driver, removed flags, NCCL, OOM, wedged GPU, permissions); patterns for
symptoms that were only described (disk full, port in use, stalled downloads) use the
standard system messages. The catalog is data: add an entry and a test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Signature:
    id: str
    severity: str          # "error" | "warning" | "info"
    pattern: str
    title: str
    cause: str
    fix: str


CATALOG: list[Signature] = [
    Signature(
        "cuda-image-newer-than-driver", "error",
        r"unsatisfied condition: cuda>=(\d+(?:\.\d+)?)",
        "Container image needs a newer CUDA than the driver supports",
        "The image was built for a newer CUDA runtime than the host driver can run "
        "(a `latest` tag moved to a CUDA 13 build while the driver supported up to CUDA 12.9).",
        "Pin an image tag built for your driver's CUDA (see `CUDA Version` in nvidia-smi), or update the driver.",
    ),
    Signature(
        "flag-removed-prefix-caching", "error",
        r"unrecognized arguments:.*--enable-prefix-caching",
        "SGLang no longer accepts --enable-prefix-caching",
        "Removed in newer SGLang (seen on 0.5.16, present on 0.5.10): the radix prefix cache is on by default.",
        "Delete the flag from the launch command or compose file; prefix caching still works.",
    ),
    Signature(
        "flag-renamed-expert-parallel", "error",
        r"unrecognized arguments:.*--enable-expert-parallel",
        "SGLang no longer accepts --enable-expert-parallel",
        "Replaced by an explicit size flag in newer SGLang (seen on 0.5.16).",
        "Use `--ep-size N` (for example `--ep-size 4`).",
    ),
    Signature(
        "nccl-init-failure", "error",
        r"NCCL error: unhandled system error|ncclCommInitRank",
        "NCCL failed to initialise the multi-GPU communicator",
        "Common in containers: too little shared memory, and unreliable peer-to-peer on PCIe consumer GPUs. "
        "Data-parallel replicas (one GPU each) never build a communicator, so they start fine.",
        "Add `ipc: host` (or a large --shm-size); set NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1. "
        "Expect slower all-reduce / all-to-all; on NVLink machines leave P2P enabled.",
    ),
    Signature(
        "gpu-oom-at-startup", "error",
        r"torch\.OutOfMemoryError|CUDA out of memory",
        "GPU out of memory while starting the server",
        "--mem-fraction-static too high once the engine's auxiliary processes are counted "
        "(0.85 overflowed 40 GB A100s; 80 GB cards did not).",
        "Lower --mem-fraction-static (0.75 worked on 40 GB) and set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.",
    ),
    Signature(
        "gpu-wedged", "error",
        r"cudaErrorDevicesUnavailable",
        "A GPU became unusable (dead CUDA context)",
        "Starting several servers at the same instant can race during CUDA initialisation and leave a dead context "
        "that a container cannot clear.",
        "Start replicas about 15 s apart. If `nvidia-smi -r` reports Insufficient Permissions, stop/start the "
        "instance; if that fails, replace the machine.",
    ),
    Signature(
        "gpu-reset-denied", "warning",
        r"Insufficient Permissions",
        "GPU reset is not permitted in this environment",
        "The container is not privileged, so the GPU cannot be reset from inside it.",
        "Restart the instance from the provider's console, or move to a machine with privileged access.",
    ),
    Signature(
        "port-in-use", "error",
        r"Address already in use|address already in use|port \d+ is already in use",
        "Listen port is already taken",
        "Another service (for example the provider's Jupyter on 8080) holds the port.",
        "Pick another gateway port (8081 worked) and point clients at it.",
    ),
    Signature(
        "disk-full", "error",
        r"No space left on device",
        "Disk is full (model cache)",
        "Several large checkpoints in one cache volume (a 32B bf16 model is about 64 GB; a 119 GB disk holds one or two).",
        "Clear the Hugging Face cache between models, or use a bigger volume. Check `echo $HF_HOME` for the real cache path.",
    ),
    Signature(
        "download-incomplete", "warning",
        r"\.incomplete",
        "Partial model download files present",
        "Two download processes wrote the same shards, or a download was interrupted and restarted repeatedly.",
        "Kill duplicate downloaders, delete the `.incomplete` shards, then download once.",
    ),
    Signature(
        "multimem-allgather-disabled", "info",
        r"multimem all-gather disabled",
        "Multimem all-gather unavailable (benign)",
        "The GPUs do not support NVLink multimem; the engine falls back to a normal all-gather.",
        "No action needed.",
    ),
]

_CUDA_VERSION = re.compile(r"CUDA Version:\s*(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str
    title: str
    evidence: str
    cause: str
    fix: str


def diagnose(text: str, catalog: list[Signature] = CATALOG) -> list[Finding]:
    """Return one finding per matching signature, first matching line as evidence."""
    lines = text.splitlines()
    out = []
    for sig in catalog:
        rx = re.compile(sig.pattern)
        for line in lines:
            if rx.search(line):
                out.append(Finding(sig.id, sig.severity, sig.title, line.strip()[:200], sig.cause, sig.fix))
                break
    order = {"error": 0, "warning": 1, "info": 2}
    out.sort(key=lambda f: order[f.severity])
    return out


def parse_driver_cuda(nvidia_smi_output: str):
    """Highest CUDA version the installed driver supports, from the nvidia-smi banner."""
    m = _CUDA_VERSION.search(nvidia_smi_output)
    return float(m.group(1)) if m else None


def check_cuda_compat(driver_cuda: float, image_cuda: float) -> Finding | None:
    """Preflight: an image built for CUDA X cannot run on a driver that supports less than X."""
    if image_cuda <= driver_cuda:
        return None
    sig = next(s for s in CATALOG if s.id == "cuda-image-newer-than-driver")
    return Finding(sig.id, sig.severity, sig.title,
                   f"image needs CUDA {image_cuda}, driver supports up to {driver_cuda}", sig.cause, sig.fix)


def catalog_markdown(catalog: list[Signature] = CATALOG) -> str:
    """Render the signature catalog (the source of docs/failure-catalog.md)."""
    out = ["# Failure catalog", "",
           "Built from failures met while deploying SGLang on 4x RTX 4090 (PCIe) and 4x A100 (NVLink) machines. "
           "Signatures for the CUDA image, removed flags, NCCL, OOM, wedged-GPU and permission entries are the messages "
           "seen in the logs; disk-full, port-in-use and stalled-download signatures use the standard system messages. "
           "Generated by `llmeval catalog`; the catalog itself lives in `src/llmeval/doctor.py`.", ""]
    for sig in catalog:
        out += [f"## {sig.title}", "", f"- **id:** `{sig.id}` ({sig.severity})", f"- **signature:** `{sig.pattern}`",
                f"- **cause:** {sig.cause}", f"- **fix:** {sig.fix}", ""]
    return "\n".join(out)
