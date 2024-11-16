<p align="center">
  <img src="chladni_small_animation.gif" alt="Chladni Pattern Animation">
</p>

# PyChladni

PyChladni is a Python library for generating physically accurate Chladni patterns - the geometric patterns that emerge when plates vibrate at specific frequencies. The library implements Ritz's method with symmetry breaking, providing accurate simulations of both idealized and real-world plate behavior.

## Overview

Chladni patterns form when sand or fine particles on a vibrating plate move to the nodal lines (points of zero displacement). While simple in concept, these patterns reveal complex physics:

- Each pattern emerges from "eigenmodes" - natural resonances determined by two integers and the plate's properties
- The patterns are scale-invariant - they look identical regardless of plate size, only the frequencies change
- Small asymmetries can drive significant pattern changes through mode competition
- Multiple frequencies create complex superpositions of patterns

The library supports:
- Single and multi-frequency pattern generation
- Audio-driven pattern generation
- Frequency sweeps
- Symmetry breaking effects
- Vector field visualization

## Installation

```bash
pip install pychladni
```

### System Requirements
- Python 3.7+
- FFmpeg (required for video generation)

## Quick Start

### Generate a Simple Pattern

```python
from pychladni import ChladniSymmetryBreaking

# Initialize with symmetry breaking
chladni = ChladniSymmetryBreaking(size=100, delta=0.022)

# Compute and visualize a mode
mode, U, V, L = chladni.compute_mode(33.2, max_modes=15)
chladni.plot_field(U, V, plot_type='stream')
```

### Audio-Driven Patterns

```python
from pychladni import AudioChladni

# Initialize
chladni = AudioChladni()

# Generate visualization from audio
patterns = chladni.generate_frame_patterns(
    audio_file="example.mp3",
    start_time=0,
    end_time=30,
    fps=30.0,
    output_dir="output_frames",
    save_video=True
)
```

## Examples

The `examples` folder contains several demonstration scripts:

- `simple_example.py`: Basic pattern generation and mode visualization
- `multi_frequency.py`: Complex patterns from frequency combinations
- `chladni_sweep.py`: Frequency sweep animations
- `audio_driven.py`: Audio-driven pattern generation

## Technical Background

### Ritz Method
The simulation uses Ritz's method to solve the plate equation, approximating the solution as a series of basis functions. This provides an efficient way to compute eigenmodes and their interactions.

### Symmetry Breaking
Real plates exhibit small asymmetries that break the perfect symmetry assumed in idealized models. When modes have similar energies, they compete through these asymmetries, leading to pattern instabilities at certain frequencies. Our implementation includes these effects based on recent research.

## Citations & References

The key references for this work include:

### Historical Development 
- Chladni, E.F.F. :  the O.G.

### Mathematical Foundations
- Ritz, W. : The foundation establishing the mathematical theory of plate vibrations & Established the Ritz method that enables numerical solution of plate vibrations.

### Modern Physical Understanding
- Gander, M.J. & Wanner, G. (2012). ["From Euler, Ritz, and Galerkin to Modern Computing"](https://doi.org/10.1137/100804036). *SIAM Review*, 54(4), 627-666. Comprehensive review connecting historical developments to modern computational methods.
- Miljković, D. (2021). ["Cymatics for Visual Representation of Aircraft Engine Noise"](https://doi.org/10.23919/MIPRO52101.2021.9597165). *44th International Convention on Information, Communication and Electronic Technology (MIPRO)*, 1064-1069. Modern applications and analysis techniques.
- Tuan, P.H., et al. (2018). ["Point-driven modern Chladni figures with symmetry breaking"](https://doi.org/10.1038/s41598-018-29244-6). *Scientific Reports*, 8, 10844. Recent advances in understanding symmetry breaking effects.

This code builds on these foundational works in plate vibration theory and modern computational methods to create an accessible Python implementation for generating and analyzing Chladni patterns.

For additional references and historical context, consult the papers linked above and their associated citations.
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use PyChladni in your research, please cite:

```bibtex
@software{pychladni2024rabideau,
  author = {Rabideau, Clayton},
  title = {PyChladni: A Python Library for Generating Chladni Patterns},
  year = {2024},
  url = {https://github.com/clayton-r/pychladni}
}
```

## Contact

Clayton Rabideau - claytonrabideau@gmail.com

Project Link: [https://github.com/clayton-r/pychladni](https://github.com/clayton-r/pychladni)