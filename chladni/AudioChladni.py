from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import signal
from p_tqdm import p_map
import os
import subprocess
import tempfile
import soundfile as sf
from PIL import Image

from chladni.FrequencyChladni import FrequencyChladni, FrequencyComponent
from chladni.PhysicalChladni import PlateProperties


@dataclass
class FrequencyFrame:
    """Represents frequencies and amplitudes at one time point"""
    time: float
    frequencies: List[float]
    amplitudes: List[float]




class AudioChladni(FrequencyChladni):
    """
    Audio-driven Chladni pattern generation and visualization.

    Extends FrequencyDrivenChladni to handle audio input and analysis,
    using FrequencyPlanner for time-based frequency transitions.
    """

    def __init__(self, size=500, delta=0.022, omega_o=104, gamma=16.64, properties=None):
        super().__init__(size, delta, omega_o, gamma, properties)
        self.planner = FrequencyPlanner()
        self.params = {
            'size': size,
            'delta': delta,
            'omega_o': omega_o,
            'gamma': gamma
        }

    def generate_frame_patterns(self, audio_file: str,
                                start_time: float = None,
                                end_time: float = None,
                                fps: float = 30.0,
                                output_dir: str = "chladni_frames",
                                save_video: bool = True) -> List[Tuple[float, np.ndarray]]:
        """Generate Chladni patterns from audio with smooth transitions"""
        os.makedirs(output_dir, exist_ok=True)

        # Get frequency plan
        plan = self.planner.detect_changes(
            audio_file,
            start_time=start_time,
            end_time=end_time
        )

        # Generate frames at target FPS
        duration = plan.duration
        frame_times = np.arange(0, duration, 1 / fps)

        frames = [
            self.planner.get_frequency_at_time(t, plan)
            for t in frame_times
        ]

        # Generate patterns in parallel
        print("\nGenerating Chladni patterns in parallel...")
        patterns = p_map(lambda i_frame: self.generate_single_frame(i_frame[0], i_frame[1], output_dir),
                         enumerate(frames))

        if save_video:
            audio_data, sr = librosa.load(audio_file, sr=None)
            if start_time is not None:
                audio_data = audio_data[int(start_time * sr):]
            if end_time is not None:
                audio_data = audio_data[:int((end_time - (start_time or 0)) * sr)]

            self._create_video(output_dir, audio_file, audio_data, sr, fps, patterns)

        return patterns

    def generate_single_frame(self, i: int, frame: FrequencyFrame, output_dir: str) -> Tuple[float, np.ndarray]:
        """Generate and save a single frame's Chladni pattern"""
        print(f"\nGenerating Frame {i + 1} at {frame.time:.2f}s")

        # Convert to FrequencyComponents
        freq_components = [
            FrequencyComponent(freq, amp)
            for freq, amp in zip(frame.frequencies, frame.amplitudes)
        ]

        # Use parent class method to compute pattern
        response, _ = self.compute_multi_frequency_response(freq_components)
        pattern = np.abs(response)
        if np.max(pattern) > 0:
            pattern /= np.max(pattern)

        # Save visualization
        plt.figure(figsize=(10, 10), facecolor='black')
        ax_pattern = plt.axes([0.1, 0.1, 0.8, 0.8])
        ax_pattern.imshow(pattern, cmap=self.custom_cmap,
                          extent=[-1, 1, -1, 1],
                          interpolation='bilinear')
        ax_pattern.axis('off')

        freq_text = "Frequencies: " + " | ".join([f"{freq:.1f} Hz" for freq in frame.frequencies])
        plt.figtext(0.5, 0.95, freq_text, color='white', ha='center', va='center', fontsize=10)

        params_text = (f"Size: {self.params['size']} | "
                       f"δ: {self.params['delta']:.3f} | "
                       f"ω₀: {self.params['omega_o']} | "
                       f"γ: {self.params['gamma']:.2f}")
        plt.figtext(0.5, 0.02, params_text, color='white', ha='center', va='center', fontsize=10)

        frame_file = os.path.join(output_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, bbox_inches='tight', pad_inches=0.1, facecolor='black')
        plt.close()

        # Ensure even dimensions
        with Image.open(frame_file) as img:
            width, height = img.size
            if width % 2 != 0 or height % 2 != 0:
                new_width = width if width % 2 == 0 else width + 1
                new_height = height if height % 2 == 0 else height + 1
                img = img.resize((new_width, new_height))
                img.save(frame_file)

        return frame.time, pattern

    def _create_video(self, frame_dir: str, audio_file: str,
                      audio_data: np.ndarray, sr: int, fps: float,
                      patterns: List[Tuple[float, np.ndarray]]) -> None:
        """Create video from frames with synchronized audio"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = os.path.join(temp_dir, "temp_audio.wav")
            sf.write(temp_audio, audio_data, sr)

            frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
            output_video = os.path.join(frame_dir, "chladni_visualization.mp4")

            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', frame_pattern,
                '-i', temp_audio,
                '-c:v', 'libx264',
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-shortest',
                output_video
            ]

            subprocess.run(cmd, check=True)
            print(f"\nVideo saved to {output_video}")


@dataclass
class StableSection:
    start_time: float
    end_time: float
    frequencies: List[float]
    amplitudes: List[float]

@dataclass
class FrequencyPlan:
    sections: List[StableSection]
    fps: float
    duration: float


class FrequencyPlanner:
    def __init__(self,
                 min_section_duration: float = 0.5,
                 max_section_duration: float = 4.0,
                 transition_duration: float = 0.5,
                 n_frequency_bands: int = 3,
                 drift_rate: float = 0.005):  # 0.5% change per second
        self.min_section_duration = min_section_duration
        self.max_section_duration = max_section_duration
        self.transition_duration = transition_duration
        self.n_bands = n_frequency_bands
        self.drift_rate = drift_rate

    def cubic_ease_out(self, t: float) -> float:
        """Cubic easing function - starts fast, then settles"""
        return 1 - (1 - t) ** 3

    def apply_frequency_drift(self, frequency: float, time_in_section: float,
                              section_duration: float) -> float:
        """Apply a very gradual drift to the frequency"""
        # Determine drift direction based on section midpoint
        mid_point = section_duration / 2
        if time_in_section < mid_point:
            # First half: drift up
            drift_factor = 1.0 + (self.drift_rate * time_in_section)
        else:
            # Second half: drift down
            drift_factor = 1.0 + (self.drift_rate * (section_duration - time_in_section))

        return frequency * drift_factor

    def detect_changes(self, audio_file: str,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> FrequencyPlan:
        """Analyze audio with fixed edge sections"""
        # Load and trim audio
        y, sr = librosa.load(audio_file)
        if start_time is not None:
            start_frame = int(start_time * sr)
            y = y[start_frame:]
        if end_time is not None:
            end_frame = int((end_time - (start_time or 0)) * sr)
            y = y[:end_frame]

        duration = len(y) / sr

        # Compute STFT
        D = librosa.stft(y)
        frequencies = np.fft.fftfreq(D.shape[0], 1.0 / sr)[:D.shape[0] // 2]
        D = D[:D.shape[0] // 2]
        times_freq = librosa.times_like(D)

        def get_frequencies_at_time(time_point):
            """Get frequencies at a specific time point"""
            time_idx = int(time_point * sr / D.shape[1])
            time_idx = np.clip(time_idx, 0, D.shape[1] - 1)
            frame = np.abs(D[:, time_idx])

            band_freqs = []
            band_amps = []
            bands = [(20, 150), (150, 800), (800, 1500)]

            for low, high in bands:
                mask = (frequencies >= low) & (frequencies < high)
                band_frame = frame[mask]
                band_freqs_subset = frequencies[mask]

                if len(band_frame) > 0 and np.max(band_frame) > 0:
                    max_idx = np.argmax(band_frame)
                    band_freqs.append(float(band_freqs_subset[max_idx]))
                    band_amps.append(float(band_frame[max_idx] / np.max(frame)))
                else:
                    band_freqs.append(float((low + high) / 2))
                    band_amps.append(0.1)

            return band_freqs, band_amps

        def get_average_frequencies(start_t, end_t):
            """Get average frequencies over a time window"""
            start_idx = int(start_t * sr / D.shape[1])
            end_idx = int(end_t * sr / D.shape[1])
            start_idx = np.clip(start_idx, 0, D.shape[1] - 1)
            end_idx = np.clip(end_idx, 0, D.shape[1] - 1)

            frame = np.mean(np.abs(D[:, start_idx:end_idx + 1]), axis=1)

            band_freqs = []
            band_amps = []
            bands = [(20, 150), (150, 800), (800, 1500)]

            for low, high in bands:
                mask = (frequencies >= low) & (frequencies < high)
                band_frame = frame[mask]
                band_freqs_subset = frequencies[mask]

                if len(band_frame) > 0 and np.max(band_frame) > 0:
                    max_idx = np.argmax(band_frame)
                    band_freqs.append(float(band_freqs_subset[max_idx]))
                    band_amps.append(float(band_frame[max_idx] / np.max(frame)))
                else:
                    band_freqs.append(float((low + high) / 2))
                    band_amps.append(0.1)

            return band_freqs, band_amps

        # Detect onset strength for middle sections
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_times = librosa.times_like(onset_env, sr=sr)

        peaks = signal.find_peaks(onset_env,
                                  distance=int(sr * self.min_section_duration / 512),
                                  prominence=np.mean(onset_env) * 1.5)[0]

        # Filter peaks to exclude edge regions
        edge_duration = 1.0  # Half second at edges
        middle_peaks = [p for p in peaks if edge_duration < onset_times[p] < (duration - edge_duration)]

        # Create sections list starting with edge section
        sections = []

        # Add initial section using average of first 0.5 seconds
        initial_freqs, initial_amps = get_average_frequencies(0, edge_duration)
        sections.append(StableSection(
            start_time=0.0,
            end_time=edge_duration,
            frequencies=initial_freqs,
            amplitudes=initial_amps
        ))

        # Add middle sections
        change_times = [onset_times[p] for p in middle_peaks]
        if change_times:
            last_time = edge_duration
            for change_time in change_times:
                if change_time - last_time >= self.min_section_duration:
                    freqs, amps = get_frequencies_at_time(change_time)
                    sections.append(StableSection(
                        start_time=float(last_time),
                        end_time=float(change_time),
                        frequencies=freqs,
                        amplitudes=amps
                    ))
                    last_time = change_time

            # Add section from last change to start of final edge section
            if duration - edge_duration - last_time >= self.min_section_duration:
                freqs, amps = get_frequencies_at_time((duration - edge_duration + last_time) / 2)
                sections.append(StableSection(
                    start_time=float(last_time),
                    end_time=float(duration - edge_duration),
                    frequencies=freqs,
                    amplitudes=amps
                ))

        # Add final section using average of last 0.5 seconds
        final_freqs, final_amps = get_average_frequencies(duration - edge_duration, duration)
        sections.append(StableSection(
            start_time=float(duration - edge_duration),
            end_time=float(duration),
            frequencies=final_freqs,
            amplitudes=final_amps
        ))

        return FrequencyPlan(sections=sections, fps=30, duration=float(duration))

    def interpolate_frequency(self, t: float, start_freq: float, end_freq: float) -> float:
        """Interpolate between frequencies using cubic easing"""
        eased_t = self.cubic_ease_out(t)
        return start_freq + (end_freq - start_freq) * eased_t

    def get_frequency_at_time(self, time: float, plan: FrequencyPlan) -> FrequencyFrame:
        """Get interpolated frequencies and amplitudes at a specific time"""
        if not plan.sections:
            return FrequencyFrame(time=time,
                                  frequencies=[0.0] * self.n_bands,
                                  amplitudes=[0.0] * self.n_bands)

        # Find current section
        current_section = None
        for section in plan.sections:
            if section.start_time <= time <= section.end_time:
                current_section = section
                break

        if current_section is None:
            return FrequencyFrame(time=time,
                                  frequencies=[0.0] * self.n_bands,
                                  amplitudes=[0.0] * self.n_bands)

        # Get index of current section
        section_idx = plan.sections.index(current_section)

        # Calculate time within current section
        time_in_section = time - current_section.start_time
        section_duration = current_section.end_time - current_section.start_time

        # If we're at a section start and there's a previous section, we need to transition
        if section_idx > 0:  # If there's a previous section
            # If we're within transition duration of the section start
            if time_in_section < self.transition_duration:
                prev_section = plan.sections[section_idx - 1]
                # Calculate transition progress (0 to 1)
                t = time_in_section / self.transition_duration
                eased_t = self.cubic_ease_out(t)

                # Interpolate from previous section to current
                frequencies = [
                    prev_f + (curr_f - prev_f) * eased_t
                    for prev_f, curr_f in zip(prev_section.frequencies, current_section.frequencies)
                ]

                amplitudes = [
                    prev_a + (curr_a - prev_a) * eased_t
                    for prev_a, curr_a in zip(prev_section.amplitudes, current_section.amplitudes)
                ]

                return FrequencyFrame(
                    time=time,
                    frequencies=frequencies,
                    amplitudes=amplitudes
                )

        # If not in transition, apply drift to current section frequencies
        frequencies = [
            self.apply_frequency_drift(f, time_in_section, section_duration)
            for f in current_section.frequencies
        ]

        return FrequencyFrame(
            time=time,
            frequencies=frequencies,
            amplitudes=current_section.amplitudes
        )

    def visualize_plan(self, plan: FrequencyPlan, output_file: str = "frequency_plan.png"):
        """Visualize the frequency change plan with smooth transitions"""
        plt.figure(figsize=(15, 10))

        # Plot frequency bands
        colors = ['blue', 'orange', 'green']
        labels = ['Bass (20-150Hz)', 'Mid (150-800Hz)', 'Treble (800-1500Hz)']

        # Create dense time points for smooth curve visualization
        times = np.linspace(0, plan.duration, int(plan.duration * 200))  # Increased resolution for smoother curves

        for band_idx in range(self.n_bands):
            frequencies = []
            for t in times:
                freq_frame = self.get_frequency_at_time(t, plan)
                frequencies.append(freq_frame.frequencies[band_idx])

            plt.plot(times, frequencies, color=colors[band_idx], label=labels[band_idx])

            # Plot vertical lines at section changes
            for section in plan.sections:
                plt.axvline(x=section.start_time, color='gray', linestyle='--', alpha=0.3)

        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Frequency Change Plan with Smooth Transitions')
        plt.grid(True, alpha=0.3)

        plt.savefig(output_file)
        plt.close()


