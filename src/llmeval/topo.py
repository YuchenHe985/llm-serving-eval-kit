"""Parse ``nvidia-smi topo -m`` and say what the GPU interconnect means for TP / EP."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LINK_CLASS = [
    (re.compile(r"^NV\d+$"), "nvlink"),
    (re.compile(r"^(PIX|PXB)$"), "pcie_switch"),
    (re.compile(r"^PHB$"), "pcie_host_bridge"),
    (re.compile(r"^(NODE|SYS)$"), "cross_socket"),
]


def classify(code: str) -> str:
    for rx, cls in _LINK_CLASS:
        if rx.match(code):
            return cls
    return "self" if code == "X" else "unknown"


@dataclass
class Topology:
    gpus: list[str]
    links: dict[tuple[int, int], str] = field(default_factory=dict)  # (i, j) -> raw code

    def classes(self) -> set[str]:
        return {classify(c) for (i, j), c in self.links.items() if i != j}

    @property
    def interconnect(self) -> str:
        cls = self.classes()
        if not cls:
            return "single_gpu"
        if cls == {"nvlink"}:
            return "nvlink"
        if "nvlink" not in cls:
            return "pcie"
        return "mixed"

    def advice(self) -> list[str]:
        ic = self.interconnect
        if ic == "nvlink":
            return ["All GPU pairs are NVLink-connected: tensor and expert parallelism are well supported."]
        if ic == "pcie":
            return [
                "No NVLink between GPUs: all-reduce (TP) and all-to-all (EP) go over PCIe.",
                "Prefer data parallelism (one replica per GPU) for models that fit on one card.",
                "If NCCL fails to initialise, try NCCL_P2P_DISABLE=1 (slower) and ipc: host in containers.",
            ]
        if ic == "mixed":
            return ["Only some GPU pairs are NVLink-connected: place TP groups inside the NVLink islands."]
        return ["Single GPU: no interconnect to consider."]


_LINK_CODE = re.compile(r"^(X|NV\d+|PIX|PXB|PHB|NODE|SYS)$")


def parse_topo(text: str) -> Topology:
    lines = text.splitlines()
    header, start = None, 0
    for idx, line in enumerate(lines):
        toks = line.split()
        if toks and toks[0] == "GPU0" and all(re.match(r"^GPU\d+$", t) for t in toks[:2]):
            header = [t for t in toks if re.match(r"^GPU\d+$", t)]
            start = idx + 1
            break
    if not header:
        raise ValueError("no GPU header row found: is this `nvidia-smi topo -m` output?")
    n = len(header)
    links: dict[tuple[int, int], str] = {}
    for line in lines[start:]:
        toks = line.split()
        if len(toks) < n + 1 or not re.match(r"^GPU\d+$", toks[0]):
            continue
        entries = toks[1:1 + n]
        if not all(_LINK_CODE.match(e) for e in entries):
            continue
        i = int(toks[0][3:])
        for j, code in enumerate(entries):
            links[(i, j)] = code
    if len(links) != n * n:
        raise ValueError(f"expected a {n}x{n} matrix, found {len(links)} cells")
    return Topology(header, links)
