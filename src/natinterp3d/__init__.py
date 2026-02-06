"""Natural neighbor interpolation in 3D.

This library builds upon the code of interpolate3d by Ross
Hemsley :footcite:`hemsley2009interpolation`.
"""

from natinterp3d.natinterp3d import Interpolator, interpolate, get_weights

try:
    from ._version import version as __version__
except ImportError:
    __version__ = '0.0.0'
