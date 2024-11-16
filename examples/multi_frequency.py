import matplotlib.pyplot as plt
from typing import List, Tuple

from chladni.FrequencyChladni import FrequencyChladni, FrequencyComponent, PlateProperties


class AudioChladni(FrequencyChladni):
    """
    Chladni pattern generator for audio frequency analysis.
    Inherits core frequency response computation from FrequencyDrivenChladni.
    """

    def __init__(self, size=500, delta=0.022, omega_o=104, gamma=16.64, properties=None):
        super().__init__(size, delta, omega_o, gamma, properties)

    def plot_multi_frequency_pattern(self, frequencies: List[FrequencyComponent]) -> None:
        """
        Plot Chladni pattern for multiple frequencies.
        Uses parent class's compute_multi_frequency_response method.
        """
        # Use parent class method for computation
        response, contributions = self.compute_multi_frequency_response(frequencies)
        pattern = self.get_pattern_from_response(response)

        plt.figure(figsize=(10, 10))
        plt.imshow(pattern, cmap=self.custom_cmap,
                   extent=[-1, 1, -1, 1],
                   interpolation='bilinear')

        # Create title showing all frequencies
        freq_str = ", ".join([f"{f.frequency:.1f}Hz" for f in frequencies])
        plt.title(f'Combined Pattern: {freq_str}')
        plt.axis('off')

        # Analyze and print contributions
        self.analyze_contributions(contributions)
        plt.show()


def main():
    # Create plate with default settings
    chladni = AudioChladni()

    # Example 1: A440 chord (A4 + E4 + A5)
    chord_freqs = [
        FrequencyComponent(440, 1.0),  # A4 (full amplitude)
        FrequencyComponent(330, 0.7),  # E4 (70% amplitude)
        FrequencyComponent(880, 0.5),  # A5 (50% amplitude)
    ]
    chladni.plot_multi_frequency_pattern(chord_freqs)

    # Example 2: Two close frequencies creating beats
    beat_freqs = [
        FrequencyComponent(440, 1.0),
        FrequencyComponent(444, 1.0),
    ]
    chladni.plot_multi_frequency_pattern(beat_freqs)

    # Example 3: Custom plate properties
    custom_props = PlateProperties()
    custom_chladni = AudioChladni(
        size=800,
        delta=0.05,
        properties=custom_props
    )

    custom_chladni.plot_multi_frequency_pattern(chord_freqs)


if __name__ == "__main__":
    main()
