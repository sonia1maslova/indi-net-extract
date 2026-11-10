"""Template-guided extraction from already prepared resting-state fMRI."""
from ._version import __version__
from ._types import (BatchExecutionError, BatchResult, BinaryMapsResult,
                     DiceResult, NetworkResult)
from .batch import extract_batch
from .postprocessing import postprocess_maps
from .evaluation import overlap_fraction, template_dice

__all__ = [
    "extract_batch", "postprocess_maps", "template_dice", "overlap_fraction",
    "NetworkResult", "BinaryMapsResult", "DiceResult", "BatchResult",
    "BatchExecutionError", "__version__",
]
