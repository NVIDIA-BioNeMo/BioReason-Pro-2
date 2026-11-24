# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Keep TorchMetrics' Extended Edit Distance (EED) module from loading.

`pytorch_lightning` imports `torchmetrics` at start-up, and `torchmetrics` imports all of its
metrics, including `torchmetrics/functional/text/eed.py`. That one file is under the RWTH EED
license (non-commercial use only) rather than the package's Apache-2.0, and this project never
uses the metric. Importing this module registers a stand-in under that module's name, so the
file is never read. Import it before anything that can import Lightning or TorchMetrics.
"""

import importlib.machinery
import sys
import types
from typing import Any, NoReturn

EED_MODULE = "torchmetrics.functional.text.eed"

#: The names TorchMetrics imports from the EED module.
EED_NAMES = ("extended_edit_distance", "_eed_compute", "_eed_update")

UNAVAILABLE = "TorchMetrics' Extended Edit Distance metric is not available in this project."


def _unavailable(*args: Any, **kwargs: Any) -> NoReturn:
    """Stand in for every EED function.

    Args:
        *args: ignored.
        **kwargs: ignored.

    Raises:
        RuntimeError: always.
    """
    raise RuntimeError(UNAVAILABLE)


def install() -> None:
    """Register the stand-in EED module, unless one is already registered.

    A stand-in has no `__file__`, so a copy of this guard elsewhere (GO-GPT has one) counts.

    Raises:
        RuntimeError: the real EED module was imported before this guard.
    """
    loaded = sys.modules.get(EED_MODULE)
    if loaded is not None:
        if getattr(loaded, "__file__", None) is not None:
            raise RuntimeError(
                f"{EED_MODULE} was imported before {__name__}; import the guard before "
                "pytorch_lightning or torchmetrics."
            )
        return

    stub = types.ModuleType(EED_MODULE, UNAVAILABLE)
    stub.__spec__ = importlib.machinery.ModuleSpec(EED_MODULE, loader=None)
    for name in EED_NAMES:
        setattr(stub, name, _unavailable)
    sys.modules[EED_MODULE] = stub


install()
