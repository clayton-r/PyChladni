import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from scipy.ndimage import binary_dilation

from chladni.FrequencyChladni import FrequencyChladni, FrequencyComponent, PlateProperties


class NodalChladni(FrequencyChladni):
    """
    Chladni pattern generator that identifies true nodal lines.

    This implementation finds actual nodal lines by detecting zero crossings
    in the real and imaginary parts of the complex response. This matches
    the physical phenomenon where sand/powder accumulates along lines of
    zero displacement.
    """

    def __init__(self, size=500, delta=0.022, omega_o=104, gamma=16.64, properties=None, tolerance=1e-10):
        super().__init__(size, delta, omega_o, gamma, properties)
        self.tolerance = tolerance
        self.colors = [(1, 1, 1), (0, 0, 0)]  # White background, black lines
        self.custom_cmap = plt.cm.binary

    def find_nodal_lines(self, response: np.ndarray) -> np.ndarray:
        """
        Find true nodal lines by identifying zero crossings in both real
        and imaginary parts of the response.

        A point is on a nodal line if both:
        1. The real part crosses zero
        2. The imaginary part crosses zero

        We detect these by checking neighboring points for sign changes.

        Parameters
        ----------
        response : np.ndarray
            Complex response array from compute_multi_frequency_response

        Returns
        -------
        np.ndarray
            Boolean array where True represents nodal lines
        """
        # Separate real and imaginary parts
        real_part = response.real
        imag_part = response.imag

        # Normalize both parts
        if np.max(np.abs(real_part)) > 0:
            real_part = real_part / np.max(np.abs(real_part))
        if np.max(np.abs(imag_part)) > 0:
            imag_part = imag_part / np.max(np.abs(imag_part))

        # Initialize arrays for zero crossings
        zero_crossings_real = np.zeros_like(real_part, dtype=bool)
        zero_crossings_imag = np.zeros_like(imag_part, dtype=bool)

        # Find zero crossings in x and y directions for real part
        zero_crossings_real[:, :-1] |= (real_part[:, :-1] * real_part[:, 1:] <= 0)
        zero_crossings_real[:-1, :] |= (real_part[:-1, :] * real_part[1:, :] <= 0)

        # Find zero crossings in x and y directions for imaginary part
        zero_crossings_imag[:, :-1] |= (imag_part[:, :-1] * imag_part[:, 1:] <= 0)
        zero_crossings_imag[:-1, :] |= (imag_part[:-1, :] * imag_part[1:, :] <= 0)

        # A point is on a nodal line if it's near zero in both real and imaginary parts
        nodal_lines = zero_crossings_real & zero_crossings_imag

        # Slightly dilate the lines to make them more visible
        return binary_dilation(nodal_lines, iterations=8)


    def get_pattern_from_response(self, response: np.ndarray) -> np.ndarray:
        """
        Convert complex response to pattern by finding nodal lines.

        Parameters
        ----------
        response : np.ndarray
            Complex response array from compute_multi_frequency_response

        Returns
        -------
        np.ndarray
            Boolean array where True represents nodal lines
        """
        return self.find_nodal_lines(response)

    def plot_multi_frequency_pattern(self, frequencies: List[FrequencyComponent]) -> None:
        """
        Plot Chladni pattern showing true nodal lines for multiple frequencies.
        """
        response, contributions = self.compute_multi_frequency_response(frequencies)
        pattern = self.get_pattern_from_response(response)

        plt.figure(figsize=(10, 10))
        plt.imshow(pattern,
                   cmap=self.custom_cmap,
                   extent=[-1, 1, -1, 1],
                   interpolation='nearest')

        freq_str = ", ".join([f"{f.frequency:.1f}Hz" for f in frequencies])
        plt.title(f'Nodal Pattern: {freq_str}')
        plt.axis('off')

        self.analyze_contributions(contributions)
        plt.show()


def main():
    # Create nodal pattern generator
    chladni = NodalChladni()

    # Example: F4 chord components
    chord_freqs = [
        FrequencyComponent(171, 1.0),  # F4
        FrequencyComponent(339, 1.0),  # E5
        FrequencyComponent(342, 0.5),  # F5
    ]
    chladni.plot_multi_frequency_pattern(chord_freqs)


if __name__ == "__main__":
    main()