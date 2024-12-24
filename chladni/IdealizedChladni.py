import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp2d

# from IPython.display import HTML
import matplotlib.animation as animation
from tqdm.auto import tqdm


class ChladniSymmetryBreaking:
    """
    Class for generating Chladni patterns with symmetry breaking using the Ritz method.

    References
    ----------
    [1] Tuan, P. H., et al. "Point-driven modern Chladni figures with symmetry breaking."
        Scientific reports 8.1 (2018): 10844.

    [2] Ritz, W. "Theorie der Transversalschwingungen einer quadratischen Platte mit freien Rändern."
        Annalen der Physik 333.4 (1909): 737-786.

    [3] Gander, M. J., and Wanner, G. "From Euler, Ritz, and Galerkin to Modern Computing."
        SIAM Review 54.4 (2012): 627-666.

    The system is governed by the anisotropic Kirchhoff-Love equation [1]:

    (1-δ)∂⁴w/∂x⁴ + (1+δ)∂⁴w/∂y⁴ - k⁴w = 0

    where δ is the symmetry breaking parameter and k is the wave number.

    For the point-driven case, the response wave function Ψ(x,y;ω) satisfies [1]:

    ∇⁴Ψ - (ρh/D)ω²Ψ = (mₐ/D)Q[δ(x-x')δ(y-y') - αΨ(x',y';ω)]
    """

    def __init__(self, size=500, delta=0.022, omega_o=104, gamma=16.64):
        """
        Initialize the Chladni pattern generator.

        Parameters:
        -----------
        size : int
            Resolution of the output pattern
        delta : float
            Symmetry breaking parameter. δ=0 for isotropic case
        omega_o : float
            Base frequency scaling factor
        gamma : float
            Damping coefficient
        """
        self.size = size
        self.delta = delta
        self.omega_o = omega_o
        self.gamma = gamma

        self.x = np.linspace(-1, 1, size)
        self.y = np.linspace(-1, 1, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Driving point position (center)
        self.xd = 0
        self.yd = 0

        plt.style.use('dark_background')

    def compute_eigenfunction(self, n1, n2, X, Y):
        """
        Compute eigenfunction ψ_{n1,n2} according to equation:

        ψ_{n1,n2}(x,y) = (2/π) cos(n₁πx/2) cos(n₂πy/2)

        Parameters:
        -----------
        n1, n2 : int
            Mode numbers
        X, Y : ndarray
            Coordinate grids

        Returns:
        --------
        ndarray
            The eigenfunction values over the grid
        """
        return (2 / np.pi) * np.cos(n1 * np.pi * X / 2) * np.cos(n2 * np.pi * Y / 2)

    def compute_eigenvalue(self, n1, n2):
        """
        Compute eigenvalue with symmetry breaking according to:

        K_{n1,n2} = (π/2)² [(1-δ)n₁² + (1+δ)n₂²]

        Parameters:
        -----------
        n1, n2 : int
            Mode numbers

        Returns:
        --------
        float
            The eigenvalue for the given mode numbers
        """
        return (np.pi / 2) ** 2 * ((1 - self.delta) * n1 ** 2 + (1 + self.delta) * n2 ** 2)

    def compute_response(self, omega, max_modes=10):
        """
        Compute response wave function Ψ according to:

        Ψ(x,y;ω) = Σ_{n1,n2} C_{n1,n2}(ω) ψ_{n1,n2}(x,y)

        where:

        C_{n1,n2}(ω) = (4Q/mp) / [1 + Ξ(x',y';ω)] *
                       ψ_{n1,n2}(x',y') / [K_{n1,n2} - (ω/ωₒ)² - iγ/ω]

        and:

        Ξ(x',y';ω) = 4Σ_{n1,n2} |ψ_{n1,n2}(x',y')|² /
                     [K_{n1,n2} - (ω/ωₒ)² - iγ/ω]

        Parameters:
        -----------
        omega : float
            Driving frequency
        max_modes : int
            Maximum number of modes to include in summation

        Returns:
        --------
        ndarray
            Complex response wave function over the grid
        """
        response = np.zeros((self.size, self.size), dtype=complex)
        U = np.zeros((self.size, self.size), dtype=complex)
        V = np.zeros((self.size, self.size), dtype=complex)
        laplacian = np.zeros((self.size, self.size), dtype=complex)  # Laplacian field

        # First, compute Ξ(x',y';ω) normalization factor
        Xi = 0
        for n1 in range(1, max_modes + 1):
            for n2 in range(1, max_modes + 1):
                eigenvalue = self.compute_eigenvalue(n1, n2)
                eigenfunction = self.compute_eigenfunction(n1, n2, self.xd, self.yd)
                Xi += 4 * eigenfunction ** 2 / (eigenvalue - (omega / self.omega_o) ** 2 - 1j * self.gamma / omega)

        # Second, compute the full response and directional components
        for n1 in range(1, max_modes + 1):
            for n2 in range(1, max_modes + 1):
                eigenvalue = self.compute_eigenvalue(n1, n2)
                eigenfunction_d = self.compute_eigenfunction(n1, n2, self.xd, self.yd)
                eigenfunction = self.compute_eigenfunction(n1, n2, self.X, self.Y)

                # Analytical derivatives of eigenfunction
                dpsi_dx = -(n1 * np.pi / 2) * np.sin(n1 * np.pi * self.X / 2) * np.cos(n2 * np.pi * self.Y / 2)
                dpsi_dy = -(n2 * np.pi / 2) * np.cos(n1 * np.pi * self.X / 2) * np.sin(n2 * np.pi * self.Y / 2)

                # Laplacian of the eigenfunction
                laplacian_psi = -((n1 * np.pi / 2) ** 2 + (n2 * np.pi / 2) ** 2) * eigenfunction

                # Compute the response coefficient for this mode
                denominator = eigenvalue - (omega / self.omega_o) ** 2 - 1j * self.gamma / omega
                C_n1_n2 = eigenfunction_d / (denominator * (1 + Xi))

                # Accumulate the response and directional components
                response += C_n1_n2 * eigenfunction
                U += C_n1_n2 * dpsi_dx
                V += C_n1_n2 * dpsi_dy
                laplacian += C_n1_n2 * laplacian_psi

        return response, U, V, laplacian

    @staticmethod
    def complex_to_real(z):
        """
        Convert complex values to real while preserving magnitude and sign information.
        For each point, returns sqrt(real² + imag²) * sign(real) to maintain:
        1. Total magnitude of the complex number (√(real² + imag²))
        2. Direction information from the real component's sign

        Parameters:
        -----------
        z : ndarray
            Complex array to convert

        Returns:
        --------
        ndarray
            Real-valued array with preserved magnitude and sign information
        """
        return np.sign(np.real(z)) * np.sqrt(np.real(z) ** 2 + np.imag(z) ** 2)

    def compute_mode(self, mode_number, max_modes=10, normalize=False, keep_complex=True):
        """
        Compute a specific mode shape with optional normalization.
        Preserves full complex information unless specifically requested not to.
        """
        # Compute base frequency and scale by mode number
        base_freq = self.omega_o * np.sqrt(self.compute_eigenvalue(1, 1))
        omega = base_freq * (1 + 0.2 * mode_number)

        # Get the raw complex fields
        response, U, V, L = self.compute_response(omega, max_modes)

        if normalize:
            # Calculate normalization factor using complex magnitude
            magnitude = np.sqrt(U * np.conjugate(U) + V * np.conjugate(V))
            max_mag = np.max(magnitude)

            if max_mag > 0:
                # Scale fields while preserving complex information
                scale_factor = 1.0 / max_mag
                U = U * scale_factor
                V = V * scale_factor
                response = response * scale_factor
                L = L * scale_factor

        if not keep_complex:
            # Convert to real values while preserving magnitude and direction
            response = self.complex_to_real(response)
            U = self.complex_to_real(U)
            V = self.complex_to_real(V)
            L = self.complex_to_real(L)

        return response, U, V, L

    def plot_field(self, U, V, plot_type='stream', title=None, ax=None, normalize=True):
        """
        Plot the displacement/flow field of the plate vibration.
        Converts complex values to real only at visualization stage if needed.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 12), dpi=150, facecolor='black')
            ax = fig.add_subplot(111)

        # Convert to real for visualization if still complex
        if np.iscomplexobj(U):
            magnitude_viz = np.sqrt(np.abs(U * np.conjugate(U) + V * np.conjugate(V)))
            U_viz = self.complex_to_real(U)
            V_viz = self.complex_to_real(V)
        else:
            magnitude_viz = np.sqrt(U**2 + V**2)
            U_viz = U
            V_viz = V

        ax.imshow(magnitude_viz, cmap='binary',
                  extent=[-1, 1, -1, 1],
                  interpolation='bilinear',
                  alpha=0.5)

        scale = 50 if normalize else 150

        if plot_type == 'stream':
            field_plot = ax.streamplot(self.x, self.y,
                                     U_viz, V_viz,
                                     color=magnitude_viz,
                                     density=10,
                                     linewidth=0.6,
                                     arrowsize=0.4,
                                     arrowstyle='->',
                                     minlength=0.1,
                                     cmap='viridis',
                                     integration_direction='both')
        else:  # quiver
            field_plot = ax.quiver(self.X, self.Y,
                                 U_viz, V_viz,
                                 magnitude_viz,
                                 cmap='viridis',
                                 scale=scale,
                                 width=0.003,
                                 headwidth=4,
                                 headlength=5)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if title:
            ax.set_title(title, pad=20, fontsize=14, color='white')

        ax.set_facecolor('black')
        if ax.get_figure() is not None:
            ax.get_figure().set_facecolor('black')

        return field_plot

    def create_field_animation(self, mode_start=33.00, mode_end=33.50, step=0.01,
                               max_modes=15, fps=10, density=80, plot_type='stream',
                               filename=None, normalize=True):
        """
        Create an animation of the displacement/flow field visualization.
        Added normalize parameter to control field normalization.
        """
        fig = plt.figure(figsize=(12, 12), dpi=150, facecolor='black')
        ax = fig.add_subplot(111)

        mode, U, V, L = self.compute_mode(mode_start, max_modes)
        field_plot = self.plot_field(U, V, plot_type=plot_type, ax=ax, normalize=normalize)
        frames = int((mode_end - mode_start) / step) + 1
        pbar = tqdm(total=frames, desc=f"Generating {plot_type} field frames")

        def update(frame):
            ax.clear()
            mode_num = mode_start + frame * step
            mode, U, V, L = self.compute_mode(mode_num, max_modes)
            self.plot_field(U, V, plot_type=plot_type, ax=ax, normalize=normalize)
            ax.set_title(f'Mode {mode_num:.2f} (δ={self.delta})',
                         pad=20, fontsize=14, color='white')
            pbar.update(1)
            return ax.get_children()

        anim = animation.FuncAnimation(fig, update, frames=frames,
                                       interval=1000 / fps)

        if filename:
            plt.rcParams['savefig.facecolor'] = 'black'
            plt.rcParams['text.color'] = 'white'
            anim.save(filename, writer='pillow', fps=fps)

        pbar.close()
        return anim

    def plot_mode(self, mode_shape, title=None, ax=None):
        """
        Plot the mode shape as a Chladni pattern.

        Parameters:
        -----------
        mode_shape : ndarray
            Complex mode shape to plot
        title : str, optional
            Title for the plot
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure

        Returns:
        --------
        matplotlib.image.AxesImage
            The plotted image
        """

        if ax is None:
            fig = plt.figure(figsize=(12, 12), dpi=150, facecolor='black')
            ax = fig.add_subplot(111)

        pattern = np.abs(mode_shape)
        pattern = pattern / np.max(pattern)

        # Create custom colormap (black to white)
        colors = [(0, 0, 0), (1, 1, 1)]
        n_bins = 100
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

        im = ax.imshow(pattern, cmap=custom_cmap,
                       extent=[-1, 1, -1, 1],
                       interpolation='bilinear')

        # Remove all borders and ticks
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if title:
            ax.set_title(title, pad=20, fontsize=14, color='white')

        # Set background color
        ax.set_facecolor('black')
        if ax.get_figure() is not None:
            ax.get_figure().set_facecolor('black')

        return im

    def create_animation(self, mode_start=33.00, mode_end=33.50, step=0.01,
                         max_modes=15, fps=10, filename=None):
        """
        Create an animation of Chladni patterns over a range of mode numbers.

        Parameters remain the same as before.
        """
        # Create figure and axis for animation
        fig = plt.figure(figsize=(12, 12), dpi=100, facecolor='black')
        ax = fig.add_subplot(111)

        # Generate first frame
        mode, U, V, L = self.compute_mode(mode_start, max_modes)
        im = self.plot_mode(mode, ax=ax)

        # Calculate total frames
        frames = int((mode_end - mode_start) / step) + 1

        # Create progress bar
        pbar = tqdm(total=frames, desc="Generating frames")

        # Function to update frame
        def update(frame):
            mode_num = mode_start + frame * step
            mode, U, V, L = self.compute_mode(mode_num, max_modes)
            pattern = np.abs(mode)
            pattern = pattern / np.max(pattern)
            im.set_array(pattern)
            ax.set_title(f'Mode {mode_num:.2f} (δ={self.delta})',
                         pad=20, fontsize=14, color='white')
            pbar.update(1)
            return [im]

        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=frames,
                                       interval=1000 / fps, blit=True)

        # Save if filename provided
        if filename:
            # Ensure white text and transparent background in saved file
            plt.rcParams['savefig.facecolor'] = 'black'
            plt.rcParams['text.color'] = 'white'
            anim.save(filename, writer='pillow', fps=fps)

        # Close progress bar
        pbar.close()

        return anim


def main():
    """
    Example usage demonstrating mode computation, visualization, and animation.
    """
    # Create instance with symmetry breaking
    chladni = ChladniSymmetryBreaking(size=100, delta=0.022)
    # plt.figure(figsize=(39, 18), dpi=400)  # dpi=300 for high resolution output
    # mode, U, V, L = chladni.compute_mode(mode_number=33.2, max_modes=15)
    # chladni.plot_mode(mode)
    # plt.show()
    # plt.savefig('chladni_pattern.png',
    #             bbox_inches='tight',  # Removes excess whitespace
    #             facecolor='black',  # Ensures black background in saved file
    #             edgecolor='none',  # No edge color
    #             )  # Slight padding around the image
    # plt.close()

    # # Create animation with progress bar
    print("Creating Chladni pattern animation...")
    anim = chladni.create_animation(
        mode_start=30.00,
        mode_end=34.00,
        step=0.01,
        fps=10,
        filename='chladni_animation_30_34_iphone_small.gif'
    )
    print("\nAnimation complete!")

    # # Display animation in notebook
    # try:
    #     from IPython.display import HTML
    #     display(HTML(anim.to_jshtml()))
    # except ImportError:
    #     plt.show()

    # mode, U, V, L = chladni.compute_mode(33.2, max_modes=15, normalize=True, keep_complex=False)
    # chladni.plot_field(U, V, plot_type='quiver')  # 'stream' or 'quiver'
    # plt.show()
    #
    # anim = chladni.create_field_animation(
    #     mode_start=30.00,
    #     mode_end=34.00,
    #     step=0.01,
    #     fps=10,
    #     filename='chladni_quiverplot_normed.gif',
    #     plot_type='quiver'  # stream or 'quiver'
    # )


if __name__ == "__main__":
    main()