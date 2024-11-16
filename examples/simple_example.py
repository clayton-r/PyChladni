import matplotlib.pyplot as plt

from chladni.IdealizedChladni import ChladniSymmetryBreaking


def main():
    """
    Example usage demonstrating mode computation, visualization, and animation.
    """
    # Create instance with symmetry breaking
    chladni = ChladniSymmetryBreaking(size=100, delta=0.022)

    # # Create animation with progress bar
    # print("Creating Chladni pattern animation...")
    # anim = chladni.create_animation(
    #     mode_start=30.00,
    #     mode_end=34.00,
    #     step=0.01,
    #     fps=10,
    #     filename='chladni_animation_30_34.gif'
    # )
    # print("\nAnimation complete!")

    # # Display animation in notebook
    # try:
    #     from IPython.display import HTML
    #     display(HTML(anim.to_jshtml()))
    # except ImportError:
    #     plt.show()

    mode, U, V, L = chladni.compute_mode(33.2, max_modes=15)
    chladni.plot_field(U, V, plot_type='stream')  # 'stream' or 'quiver'
    plt.show()

    anim = chladni.create_field_animation(
        mode_start=30.00,
        mode_end=34.00,
        step=0.01,
        fps=10,
        filename='chladni_streamplot.gif',
        plot_type='stream'  # stream or 'quiver'
    )


if __name__ == "__main__":
    main()