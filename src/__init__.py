"""
PACO algorithm package (Flasseur et al. 2018).

Public API
----------
PACO      : Algorithm 1 (called full PACO in the paper, to distinguish it
            from the fast variant), with a shared-memory statistics cache
            and radius-ring traversal; optional GPU support
FastPACO  : Algorithm 2, statistics on the pixel grid with interpolated
            per-pixel terms
PACOBase  : shared base class (and serial reference implementation)

FullPACOv2 and FastPACOv2, the former names of PACO and FastPACO, remain
available as deprecated aliases.
"""
from .fast_paco import FastPACO, FastPACOv2
from .full_paco import PACO, FullPACOv2
from .paco_base import PACOBase

__all__ = ['PACO', 'FastPACO', 'PACOBase', 'FullPACOv2', 'FastPACOv2']
