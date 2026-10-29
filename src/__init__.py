"""
PACO algorithm package.

Public API
----------
FastPACOv2 : memory-efficient batch variant with map interpolation
FullPACOv2  : per-pixel caching variant with optional GPU support
"""
from .fast_paco import FastPACOv2
from .full_paco import FullPACOv2

__all__ = ['FastPACOv2', 'FullPACOv2']
