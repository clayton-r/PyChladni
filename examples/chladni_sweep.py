import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import subprocess
from PIL import Image
import tempfile
import soundfile as sf
from scipy import signal
from tqdm import tqdm

from chladni.PhysicalChladni import PhysicalChladni


class ChladniSweepGenerator:
    def __init__(self, size=500, properties=None):
        self.chladni = PhysicalChladni(size=size, properties=properties)

    def generate_sweep(self,
                       start_freq: float = 36.8,
                       end_freq: float = 1500.0,
                       duration: float = 10.0,
                       fps: float = 30.0,
                       output_dir: str = "sweep_frames",
                       sample_rate: int = 44100):
        """Generate a logarithmic frequency sweep with visuals and audio"""
        os.makedirs(output_dir, exist_ok=True)

        # Calculate frames
        n_frames = int(duration * fps)
        frame_times = np.linspace(0, duration, n_frames)

        # Use logarithmic frequency scaling for more musical sweep
        frequencies = np.exp(
            np.linspace(
                np.log(start_freq),
                np.log(end_freq),
                n_frames
            )
        )

        # Generate audio
        t_audio = np.linspace(0, duration, int(duration * sample_rate))
        # Logarithmic frequency sweep
        inst_freq = np.exp(
            np.linspace(
                np.log(start_freq),
                np.log(end_freq),
                len(t_audio)
            )
        )

        # Generate chirp signal
        phase = np.cumsum(2 * np.pi * inst_freq / sample_rate)
        audio = np.sin(phase)

        # Apply envelope to avoid clicks
        envelope = np.ones_like(audio)
        fade_samples = int(0.01 * sample_rate)  # 10ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= envelope

        # Normalize audio
        audio = audio * 0.9 / np.max(np.abs(audio))

        # Generate frames
        patterns = []
        print("\nGenerating Chladni patterns...")
        for i, (t, freq) in enumerate(tqdm(zip(frame_times, frequencies), total=n_frames)):
            # Compute pattern
            response, _ = self.chladni.compute_response(freq)
            pattern = np.abs(response)
            if np.max(pattern) > 0:
                pattern /= np.max(pattern)

            # Save frame
            plt.figure(figsize=(10, 10))
            colors = [(0, 0, 0), (1, 1, 1)]
            custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)
            plt.imshow(pattern, cmap=custom_cmap, extent=[-1, 1, -1, 1], interpolation='bilinear')
            plt.axis('off')

            frame_file = os.path.join(output_dir, f"frame_{i:04d}.png")
            plt.savefig(frame_file, bbox_inches='tight', pad_inches=0)
            plt.close()

            patterns.append((t, pattern))

        # Create video with audio
        self._create_video(output_dir, audio, sample_rate, fps)

        return patterns, audio, sample_rate

    def _create_video(self, frame_dir: str, audio: np.ndarray,
                      sample_rate: int, fps: float) -> None:
        """Create video from frames with synchronized audio"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save audio
            temp_audio = os.path.join(temp_dir, "temp_audio.wav")
            sf.write(temp_audio, audio, sample_rate)

            # Define frame pattern
            frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
            output_video = os.path.join(frame_dir, "chladni_sweep.mp4")

            # FFmpeg command
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


def main():
    # Create generator
    generator = ChladniSweepGenerator()

    # Generate sweep from 36.8 Hz to 1500 Hz over 10 seconds
    patterns, audio, sr = generator.generate_sweep(
        start_freq=36.8,
        end_freq=1500.0,
        duration=10.0,
        fps=30.0,
        output_dir="../render/chladni_sweep"
    )


if __name__ == "__main__":
    main()