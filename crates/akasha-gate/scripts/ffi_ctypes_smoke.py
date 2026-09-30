#!/usr/bin/env python3
"""ctypes smoke for the akasha-gate cdylib (no tool execution)."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "contracts" / "fixtures" / "path_a_ready_request.json"


def _lib_path() -> Path:
    target = ROOT / "target"
    for profile in ("debug", "release"):
        for name in (
            f"{profile}/libakasha_gate.so",
            f"{profile}/libakasha_gate.dylib",
            f"{profile}/akasha_gate.dll",
        ):
            path = target / name
            if path.exists():
                return path
    raise SystemExit(
        "libakasha_gate not found under target/{debug,release}. "
        "Run: cargo build -p akasha-gate"
    )


def main() -> int:
    lib = ctypes.CDLL(str(_lib_path()))
    lib.akasha_gate_abi_version.restype = ctypes.c_char_p
    lib.akasha_gate_contract_version.restype = ctypes.c_uint32
    lib.akasha_gate_evaluate_json.argtypes = [ctypes.c_char_p]
    lib.akasha_gate_evaluate_json.restype = ctypes.c_void_p
    lib.akasha_gate_string_free.argtypes = [ctypes.c_void_p]
    lib.akasha_gate_string_free.restype = None

    assert lib.akasha_gate_abi_version() == b"1"
    assert lib.akasha_gate_contract_version() == 1

    request = FIXTURE.read_bytes()
    ptr = lib.akasha_gate_evaluate_json(request)
    if not ptr:
        raise SystemExit("evaluate_json returned NULL")
    try:
        raw = ctypes.cast(ptr, ctypes.c_char_p).value
        assert raw is not None
        payload = json.loads(raw.decode("utf-8"))
    finally:
        lib.akasha_gate_string_free(ptr)

    assert payload["ok"] is True
    assert payload["status"] == "ready"
    assert payload["tool_name"] == "fs.read"
    assert payload["request_id"] == "req-ready-001"
    print("ffi ctypes smoke ok:", payload["status"], payload["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
