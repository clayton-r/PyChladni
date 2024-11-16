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
pip install git+https://github.com/clayton-r/pychladni.git
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

## Detailed Examples

### Physical Chladni Pattern Generation

The `PhysicalChladni` class lets you generate patterns based on direct physical simulation of plate vibrations. Here's how to use it:

```python
import numpy as np
import matplotlib.pyplot as plt
from pychladni import PhysicalChladni, PlateProperties

# Set up plate properties
plate_props = PlateProperties(
    # Material properties
    youngs_modulus=70e9,  # Young's modulus (Pa) - aluminum ≈ 70 GPa
    poisson_ratio=0.33,   # Poisson ratio - aluminum ≈ 0.33
    density=2700,         # Density (kg/m³) - aluminum ≈ 2700
    thickness=0.002,      # Plate thickness (m) - 2mm
    
    # Geometry
    length=0.2,          # Plate length (m) - 20cm
    width=0.2           # Plate width (m) - 20cm
)

# Initialize the Chladni simulator
chladni = PhysicalChladni(
    size=500,           # Resolution of simulation grid
    delta=0.022,        # Asymmetry coefficient (required to make the patterns interesting)
    omega_o=104,        # (rad/s)
    gamma=16.64,        # Driving force coefficient
    properties=plate_props
)

# Generate patterns for different frequencies
frequencies = [50, 100, 200, 400]  # Hz
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for idx, freq in enumerate(frequencies):
    # Compute response for this frequency
    pattern, _ = chladni.compute_response(freq)
    
    # Plot
    ax = axs[idx//2, idx%2]
    im = ax.imshow(np.abs(pattern), cmap='viridis')
    ax.set_title(f'{freq} Hz')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Generate a single pattern with more detail
frequency = 440  # Hz (A4 note)
pattern, contributions = chladni.compute_response(frequency)

plt.figure(figsize=(8, 8))
plt.imshow(np.abs(pattern), cmap='viridis')
plt.colorbar(label='Amplitude')
plt.title(f'Chladni Pattern at {frequency} Hz')
plt.axis('off')
plt.show()

# Analyze modal contributions
for n1, n2, f, w in contributions[:5]:  # Show top 5 contributing modes
    print(f"Mode ({n1}, {n2}): {f:.1f} Hz, Weight: {w:.3f}")
```

This example produces Chladni patterns for different frequencies, showing how the patterns become more complex at higher frequencies. The plate parameters can be adjusted to match different materials and geometries:

Common material properties:
- Aluminum: E = 70 GPa, ν = 0.33, ρ = 2700 kg/m³
- Steel: E = 200 GPa, ν = 0.30, ρ = 7800 kg/m³
- Brass: E = 105 GPa, ν = 0.35, ρ = 8500 kg/m³
- Glass: E = 70 GPa, ν = 0.22, ρ = 2500 kg/m³

The `compute_response()` method returns both the pattern and a list of contributing modes, allowing you to analyze which eigenmodes are most important for each frequency.

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
