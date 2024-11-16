"""
PyChladni - A Python library for generating Chladni patterns
==========================================================

PyChladni implements physical simulations of plate vibrations to generate
Chladni patterns - the geometric patterns that emerge when plates vibrate
at specific frequencies.

Main Classes
-----------
PhysicalChladni : Basic Chladni pattern generation
AudioChladni : Audio-driven pattern generation
FrequencyPlanner : Frequency analysis and transition planning
ChladniSweepGenerator : Generate frequency sweep animations

Examples
--------
Basic pattern generation:
>>> from pychladni import PhysicalChladni
>>> chladni = PhysicalChladni(size=500)
>>> pattern, _ = chladni.compute_response(440)  # Generate 440Hz pattern

Audio-driven visualization:
>>> from pychladni import AudioChladni
>>> chladni = AudioChladni()
>>> chladni.generate_frame_patterns(
...     audio_file="music.mp3",
...     output_dir="frames",
...     save_video=True
... )
"""

__version__ = '0.1.0'
__author__ = 'Clayton Rabideau'
__email__ = 'claytonrabideau@gmail.com'

from .PhysicalChladni import PhysicalChladni, PlateProperties
from .AudioChladni import (
    AudioChladni,
    FrequencyFrame,
    FrequencyPlan,
    FrequencyPlanner,
    StableSection
)
from IdealizedChladni import ChladniSymmetryBreaking
from FrequencyChladni import FrequencyChladni, FrequencyComponent

__all__ = [
    'PhysicalChladni',
    'PlateProperties',
    'AudioChladni',
    'FrequencyComponent',
    'FrequencyFrame',
    'FrequencyPlan',
    'FrequencyPlanner',
    'StableSection',
    'ChladniSymmetryBreaking'
]