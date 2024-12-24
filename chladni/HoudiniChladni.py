import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter
from chladni.FrequencyChladni import FrequencyChladni, FrequencyComponent, PlateProperties


class HoudiniStyleChladni(FrequencyChladni):
    """
    Chladni pattern generator that mimics Houdini's FEM-based visualization.
    Uses gradient-based nodal detection for smooth, physically-based patterns.
    """

    def __init__(self, size=500, delta=0.022, omega_o=104, gamma=16.64, properties=None,
                 gradient_weight=1.0, smoothing_sigma=1.0):
        super().__init__(size, delta, omega_o, gamma, properties)
        self.gradient_weight = gradient_weight
        self.smoothing_sigma = smoothing_sigma
        # Use a perceptually uniform colormap
        self.custom_cmap = plt.cm.magma

    def compute_gradient_magnitude(self, field: np.ndarray) -> np.ndarray:
        """
        Compute the magnitude of the gradient field.

        Parameters
        ----------
        field : np.ndarray
            2D scalar field

        Returns
        -------
        np.ndarray
            Gradient magnitude field
        """
        # Compute gradients using central differences
        dy, dx = np.gradient(field)
        return np.sqrt(dx ** 2 + dy ** 2)

    def get_pattern_from_response(self, response: np.ndarray) -> np.ndarray:
        """
        Convert complex response to pattern using gradient-based nodal detection.

        This method:
        1. Computes amplitude field
        2. Calculates gradient magnitude
        3. Combines amplitude and gradient information
        4. Applies smoothing for better visualization

        Parameters
        ----------
        response : np.ndarray
            Complex response array from compute_multi_frequency_response

        Returns
        -------
        np.ndarray
            Pattern field with values in [0, 1]
        """
        # Get amplitude field
        amplitude = np.abs(response)
        if np.max(amplitude) > 0:
            amplitude /= np.max(amplitude)

        # Compute gradient magnitude of amplitude
        grad_mag = self.compute_gradient_magnitude(amplitude)
        if np.max(grad_mag) > 0:
            grad_mag /= np.max(grad_mag)

        # Combine amplitude and gradient information
        # High gradient + low amplitude = likely nodal region
        pattern = (1 - amplitude) * (self.gradient_weight * grad_mag + 1)

        # Normalize
        if np.max(pattern) > 0:
            pattern /= np.max(pattern)

        # Apply Gaussian smoothing for better visualization
        if self.smoothing_sigma > 0:
            pattern = gaussian_filter(pattern, sigma=self.smoothing_sigma)

        # Enhance contrast
        pattern = np.power(pattern, 1.5)

        return pattern

    def plot_multi_frequency_pattern(self, frequencies: List[FrequencyComponent]) -> None:
        """
        Plot Houdini-style Chladni pattern for multiple frequencies.
        """
        response, contributions = self.compute_multi_frequency_response(frequencies)
        pattern = self.get_pattern_from_response(response)

        plt.figure(figsize=(10, 10))
        plt.imshow(pattern,
                   cmap=self.custom_cmap,
                   extent=[-1, 1, -1, 1],
                   interpolation='bilinear')

        freq_str = ", ".join([f"{f.frequency:.1f}Hz" for f in frequencies])
        plt.title(f'Houdini-Style Pattern: {freq_str}')
        plt.axis('off')

        self.analyze_contributions(contributions)
        plt.show()

    def plot_detailed_analysis(self, frequencies: List[FrequencyComponent]) -> None:
        """
        Plot detailed analysis showing amplitude, gradient, and final pattern.
        """
        response, _ = self.compute_multi_frequency_response(frequencies)

        # Get individual components
        amplitude = np.abs(response)
        amplitude /= np.max(amplitude)

        grad_mag = self.compute_gradient_magnitude(amplitude)
        grad_mag /= np.max(grad_mag)

        pattern = self.get_pattern_from_response(response)

        # Create subplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot amplitude
        axes[0].imshow(amplitude, cmap='viridis', extent=[-1, 1, -1, 1])
        axes[0].set_title('Amplitude')
        axes[0].axis('off')

        # Plot gradient magnitude
        axes[1].imshow(grad_mag, cmap='viridis', extent=[-1, 1, -1, 1])
        axes[1].set_title('Gradient Magnitude')
        axes[1].axis('off')

        # Plot final pattern
        axes[2].imshow(pattern, cmap=self.custom_cmap, extent=[-1, 1, -1, 1])
        axes[2].set_title('Final Pattern')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()


def main():
    # Create Houdini-style pattern generator
    chladni = HoudiniStyleChladni(
        size=500,
        gradient_weight=1.2,  # Adjust to control pattern definition
        smoothing_sigma=0.8  # Adjust to control smoothness
    )

    # Example: F4 chord components
    chord_freqs = [
        FrequencyComponent(171, 1.0),  # F4
        FrequencyComponent(339, 1.0),  # E5
        FrequencyComponent(342, 0.5),  # F5
    ]

    # Plot both standard and detailed analysis
    chladni.plot_multi_frequency_pattern(chord_freqs)
    chladni.plot_detailed_analysis(chord_freqs)


if __name__ == "__main__":
    main()