import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ModeContribution:
    n1: int
    n2: int
    frequency: float
    weight: float
    eigenfunction: np.ndarray = None

    def __str__(self):
        return f"Mode ({self.n1},{self.n2}): {self.frequency:.1f}Hz, weight={self.weight:.3f}"


class ChladniBlender:
    def __init__(self, size=500, delta=0.022, omega_o=104):
        self.size = size
        self.delta = delta
        self.omega_o = omega_o

        # Set up coordinate grid
        self.x = np.linspace(-1, 1, size)
        self.y = np.linspace(-1, 1, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def compute_eigenfunction(self, n1: int, n2: int) -> np.ndarray:
        """Compute eigenfunction for given mode numbers"""
        return (2 / np.pi) * np.cos(n1 * np.pi * self.X / 2) * np.cos(n2 * np.pi * self.Y / 2)

    def natural_frequency(self, n1: int, n2: int) -> float:
        """Compute natural frequency for mode numbers"""
        eigenvalue = (np.pi / 2) ** 2 * ((1 - self.delta) * n1 ** 2 + (1 + self.delta) * n2 ** 2)
        return self.omega_o * np.sqrt(eigenvalue)

    def find_mode_contributions(self, freq: float, max_modes: int = 10) -> List[ModeContribution]:
        """Find contributions of natural modes to a given frequency"""
        contributions = []
        driving_omega = 2 * np.pi * freq
        damping = 0.01

        for n1 in range(1, max_modes + 1):
            for n2 in range(1, max_modes + 1):
                natural_freq = self.natural_frequency(n1, n2)
                natural_omega = 2 * np.pi * natural_freq

                # Compute response magnitude
                denominator = (natural_omega ** 2 - driving_omega ** 2) ** 2 + \
                              (2 * damping * natural_omega * driving_omega) ** 2
                weight = 1 / np.sqrt(denominator)

                # Compute eigenfunction
                eigenfunction = self.compute_eigenfunction(n1, n2)

                contributions.append(ModeContribution(n1, n2, natural_freq, weight, eigenfunction))

        # Normalize weights
        total_weight = sum(c.weight for c in contributions)
        for c in contributions:
            c.weight /= total_weight

        return sorted(contributions, key=lambda x: x.weight, reverse=True)

    def compute_blended_pattern(self, freq: float, max_modes: int = 10) -> Tuple[np.ndarray, List[ModeContribution]]:
        """Compute the blended pattern and return contributions"""
        contributions = self.find_mode_contributions(freq, max_modes)

        # Initialize pattern with zeros
        pattern = np.zeros_like(self.X, dtype=complex)

        # Sum weighted eigenfunctions
        for c in contributions:
            pattern += c.weight * c.eigenfunction

        return pattern, contributions

    def visualize_pattern_composition(self, freq: float, max_modes: int = 10, top_n: int = 4):
        """Visualize how individual modes combine to create the final pattern"""
        pattern, contributions = self.compute_blended_pattern(freq, max_modes)

        # Set up plot
        n_plots = top_n + 2  # top modes + weighted sum + final pattern
        fig = plt.figure(figsize=(10, 3 * ((n_plots + 1) // 2)))

        # Create custom colormap
        colors = [(0, 0, 0), (1, 1, 1)]
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)

        # Plot top contributing modes
        for i, c in enumerate(contributions[:top_n]):
            ax = plt.subplot(((n_plots + 1) // 2), 2, i + 1)
            im = ax.imshow(np.abs(c.eigenfunction), cmap=custom_cmap,
                           extent=[-1, 1, -1, 1], interpolation='bilinear')
            ax.set_title(f'Mode ({c.n1},{c.n2}): {c.frequency:.1f}Hz\nWeight: {c.weight:.3f}')
            ax.axis('off')

        # Plot weighted sum
        ax = plt.subplot(((n_plots + 1) // 2), 2, top_n + 1)
        weighted_sum = np.zeros_like(self.X)
        for c in contributions[:top_n]:
            weighted_sum += c.weight * np.abs(c.eigenfunction)
        im = ax.imshow(weighted_sum, cmap=custom_cmap,
                       extent=[-1, 1, -1, 1], interpolation='bilinear')
        ax.set_title(f'Weighted Sum of Top {top_n} Modes')
        ax.axis('off')

        # Plot final pattern (all modes)
        ax = plt.subplot(((n_plots + 1) // 2), 2, top_n + 2)
        im = ax.imshow(np.abs(pattern), cmap=custom_cmap,
                       extent=[-1, 1, -1, 1], interpolation='bilinear')
        ax.set_title(f'Final Pattern at {freq}Hz\n(All {max_modes}² modes)')
        ax.axis('off')

        plt.tight_layout()
        return fig


def main():
    blender = ChladniBlender(size=200)

    # Test frequencies
    test_frequencies = [440, 657, 823]

    for freq in test_frequencies:
        print(f"\nAnalyzing {freq} Hz:")

        # Compute and visualize pattern
        pattern, contributions = blender.compute_blended_pattern(freq)

        print("\nTop 5 contributing modes:")
        for c in contributions[:5]:
            print(c)

        # Visualize composition
        blender.visualize_pattern_composition(freq)
        plt.show()
        plt.savefig('mode_blend.png', pad_inches=0, dpi=300)


if __name__ == "__main__":
    main()