# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pytest harness for the pure-Python Autellix policy core.

The modules under test live under ``vllm/v1/core/sched/autellix/`` but import
nothing from vLLM or torch. Importing them through the normal package path
would execute ``vllm/__init__.py``, which imports torch and therefore fails in
a minimal environment. When torch is unavailable we register lightweight
stand-in packages for the ``vllm`` chain in ``sys.modules`` with ``__path__``
pointing at the real directories, so ``vllm.v1.core.sched.autellix.*`` loads
the real (pure) modules while bypassing the heavy parent initialisation. When
torch is present the real package is used unchanged.
"""

import importlib.util
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _register_stub_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


if importlib.util.find_spec("torch") is None:
    for _name, _relpath in (
        ("vllm", "vllm"),
        ("vllm.v1", "vllm/v1"),
        ("vllm.v1.core", "vllm/v1/core"),
        ("vllm.v1.core.sched", "vllm/v1/core/sched"),
    ):
        if _name not in sys.modules:
            _register_stub_package(_name, _REPO_ROOT / _relpath)
