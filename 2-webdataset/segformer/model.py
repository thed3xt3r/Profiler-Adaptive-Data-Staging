"""Re-exports the real ArcheoModel from 3-pads/segformer/model.py instead of
carrying a fourth stale copy of the model/loss/metric definitions.

This matters for correctness, not just DRY: evaluation method 3 (WebDataset)
is only a valid comparison against methods 1/2/5 if the model, loss and
metrics are byte-identical and only the data path differs (Section
evaluation-plan's own stated principle). Importing the same model.py PADS
itself trains with -- already updated with precision/recall/MCC -- guarantees
that rather than hoping a hand-maintained duplicate stays in sync.

Loaded via importlib with an explicit file path and a non-"model" module
name, not via sys.path + `import model`: this file is ALSO named model.py,
so a plain `import model` after a sys.path.insert would resolve to this
already-executing module (sys.modules caches it under the name "model")
instead of 3-pads/segformer/model.py -- a self-import, not the intended one.
"""
import importlib.util
import os
import sys

_PADS_SEGFORMER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "PADS", "segformer")

# gpu_monitor.py (imported separately by train.py) lives alongside the real
# model.py; put 3-pads/segformer on sys.path too so that plain import finds it.
if _PADS_SEGFORMER_DIR not in sys.path:
    sys.path.insert(0, _PADS_SEGFORMER_DIR)

_spec = importlib.util.spec_from_file_location(
    "pads_segformer_model", os.path.join(_PADS_SEGFORMER_DIR, "model.py")
)
_pads_segformer_model = importlib.util.module_from_spec(_spec)
sys.modules["pads_segformer_model"] = _pads_segformer_model
_spec.loader.exec_module(_pads_segformer_model)

ArcheoModel = _pads_segformer_model.ArcheoModel
