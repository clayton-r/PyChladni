import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class ModeContribution:
    n1: int
    n2: int
    frequency: float
    weight: float  # Contribution strength

    def __str__(self):
        return f"Mode ({self.n1},{self.n2}): {self.frequency:.1f}Hz, weight={self.weight:.3f}"


class ChladniModeAnalysis:
    def __init__(self, delta=0.022, omega_o=104):
        self.delta = delta
        self.omega_o = omega_o

    def compute_natural_frequency(self, n1: int, n2: int) -> float:
        """Compute natural frequency for integer mode numbers"""
        eigenvalue = (np.pi / 2) ** 2 * ((1 - self.delta) * n1 ** 2 + (1 + self.delta) * n2 ** 2)
        return self.omega_o * np.sqrt(eigenvalue)

    def find_mode_contributions(self, freq: float, max_modes: int = 10) -> List[ModeContribution]:
        """
        Find contributions of natural modes to a given driving frequency.
        Uses resonance formula to determine relative weights.
        """
        contributions = []
        driving_omega = 2 * np.pi * freq
        damping = 0.01  # Damping ratio

        # Check all integer mode combinations
        for n1 in range(1, max_modes + 1):
            for n2 in range(1, max_modes + 1):
                natural_freq = self.compute_natural_frequency(n1, n2)
                natural_omega = 2 * np.pi * natural_freq

                # Compute response magnitude using resonance formula
                denominator = (natural_omega ** 2 - driving_omega ** 2) ** 2 + \
                              (2 * damping * natural_omega * driving_omega) ** 2
                weight = 1 / np.sqrt(denominator)

                contributions.append(ModeContribution(n1, n2, natural_freq, weight))

        # Normalize weights
        total_weight = sum(c.weight for c in contributions)
        for c in contributions:
            c.weight /= total_weight

        # Sort by contribution strength
        return sorted(contributions, key=lambda x: x.weight, reverse=True)

    def visualize_mode_contributions(self, freq: float, max_modes: int = 6):
        """Visualize how different integer modes contribute to arbitrary frequency"""
        contributions = self.find_mode_contributions(freq, max_modes)

        # Create mode grid visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Plot 1: Mode weights as scatter
        weights = np.array([c.weight for c in contributions])
        scatter = ax1.scatter([c.n1 for c in contributions],
                              [c.n2 for c in contributions],
                              c=weights,
                              s=1000 * weights,
                              cmap='viridis',
                              alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Contribution Weight')

        # Add mode labels
        for c in contributions[:5]:  # Label top 5 contributors
            ax1.annotate(f'({c.n1},{c.n2})\n{c.frequency:.0f}Hz\n{c.weight:.3f}',
                         (c.n1, c.n2),
                         xytext=(5, 5),
                         textcoords='offset points')

        ax1.set_xlabel('n₁')
        ax1.set_ylabel('n₂')
        ax1.set_title(f'Mode Contributions for {freq}Hz')
        ax1.grid(True)

        # Plot 2: Frequency spectrum
        freqs = np.array([c.frequency for c in contributions])
        weights = np.array([c.weight for c in contributions])

        ax2.vlines(freqs, 0, weights, alpha=0.5)
        ax2.plot(freqs, weights, 'o')
        ax2.axvline(freq, color='r', linestyle='--', label='Target frequency')

        # Highlight nearest modes
        nearest_idx = np.argmin(np.abs(freqs - freq))
        ax2.plot(freqs[nearest_idx], weights[nearest_idx], 'ro')

        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Contribution Weight')
        ax2.set_title('Mode Contributions vs Frequency')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig


def main():
    analyzer = ChladniModeAnalysis()

    # Test with some arbitrary frequencies
    test_frequencies = [440, 657, 823]  # Hz

    for freq in test_frequencies:
        print(f"\nAnalyzing frequency: {freq} Hz")
        contributions = analyzer.find_mode_contributions(freq)

        print("\nTop 5 contributing modes:")
        for c in contributions[:5]:
            print(c)

        # Visualize
        analyzer.visualize_mode_contributions(freq)
        plt.show()
        plt.savefig('mode_contribution.png', bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == "__main__":
    main()