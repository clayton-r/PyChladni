from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from chladni.PhysicalChladni import PhysicalChladni, PlateProperties


@dataclass
class FrequencyComponent:
    """
    Represents a single frequency component in a Chladni pattern.

    Following Tuan et al. (2018), each frequency component contributes to
    the overall plate response according to the biharmonic equation:
        ∆²w = λw
    where λ is related to the frequency by the dispersion relation.

    Parameters
    ----------
    frequency : float
        Frequency in Hz
    amplitude : float
        Relative amplitude in [0,1] for superposition
    """
    frequency: float  # Hz
    amplitude: float  # Relative amplitude


class FrequencyChladni(PhysicalChladni):
    """
    Base class for frequency-driven Chladni pattern generation.

    Extends PhysicalChladni to handle multiple frequency inputs and their
    superposition. The total response is computed as:
        w_total = Σ A_i * w_i
    where w_i is the response at frequency f_i and A_i is its amplitude.

    Inherits the plate parameters and computation methods from PhysicalChladni
    while adding frequency-specific functionality.
    """

    def __init__(self, size=500, delta=0.022, omega_o=104, gamma=16.64, properties=None):
        """
        Initialize FrequencyDrivenChladni with plate parameters.

        Parameters match parent PhysicalChladni class.
        """
        super().__init__(size, delta, omega_o, gamma, properties)

        # Set up default visualization parameters
        self.colors = [(0, 0, 0), (1, 1, 1)]
        self.custom_cmap = LinearSegmentedColormap.from_list("custom", self.colors, N=100)

    def compute_multi_frequency_response(self,
                                         frequencies: List[FrequencyComponent]
                                         ) -> Tuple[np.ndarray, List[Tuple[int, int, float, float]]]:
        """
        Compute the superposed response for multiple frequency components.

        From Tuan et al. (2018), the total response is a weighted sum of
        individual modal responses at each frequency.

        Parameters
        ----------
        frequencies : List[FrequencyComponent]
            List of frequency components with their amplitudes

        Returns
        -------
        tuple
            - Complex response array of shape (size, size)
            - List of (n1, n2, freq, weight) tuples describing modal contributions
        """
        total_response = np.zeros((self.size, self.size), dtype=complex)
        total_contributions = []

        for freq_comp in frequencies:
            if freq_comp.frequency > 0:  # Skip DC/negative frequencies
                response, contributions = self.compute_response(freq_comp.frequency)
                total_response += freq_comp.amplitude * response
                # Scale contributions by component amplitude
                total_contributions.extend([
                    (n1, n2, f, w * freq_comp.amplitude)
                    for n1, n2, f, w in contributions
                ])

        return total_response, total_contributions

    def get_pattern_from_response(self, response: np.ndarray) -> np.ndarray:
        """
        Convert complex response to normalized pattern amplitudes.

        Parameters
        ----------
        response : np.ndarray
            Complex response array from compute_multi_frequency_response

        Returns
        -------
        np.ndarray
            Normalized pattern amplitudes in [0,1]
        """
        pattern = np.abs(response)
        if np.max(pattern) > 0:
            pattern /= np.max(pattern)
        return pattern

    def analyze_contributions(self, contributions: List[Tuple[int, int, float, float]]) -> None:
        """
        Analyze and print modal contributions to the pattern.

        Parameters
        ----------
        contributions : List[Tuple]
            List of (n1, n2, freq, weight) tuples from compute_multi_frequency_response
        """
        contributions.sort(key=lambda x: x[3], reverse=True)
        total_weight = sum(c[3] for c in contributions)

        print("\nTop contributing modes:")
        for n1, n2, f_nat, weight in contributions[:5]:
            rel_weight = weight / total_weight * 100
            print(f"Mode ({n1},{n2}): {f_nat:.1f} Hz, contribution: {rel_weight:.1f}%")


