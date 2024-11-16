from chladni.AudioChladni import AudioChladni


def main():
    # Create AudioChladni instance with default settings
    chladni = AudioChladni()

    # Process audio file
    audio_file = "example.mp3"

    # Analyze audio and create frequency plan
    plan = chladni.planner.detect_changes(
        audio_file,
        start_time=30,  # Optional time range
        end_time=47
    )

    # Visualize frequency analysis
    chladni.planner.visualize_plan(plan, "frequency_plan.png")

    # Generate Chladni animation with synchronized audio
    patterns = chladni.generate_frame_patterns(
        audio_file,
        start_time=30,
        end_time=47,
        fps=30.0,
        output_dir="../render/example",
        save_video=True
    )


if __name__ == "__main__":
    main()