import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


@dataclass
class PlateProperties:
    """
    Physical properties for plate vibration simulation.

    Based on brass plate parameters from Tuan et al. (2018):

    Parameters
    ----------
    Ex : float
        Young's modulus in x direction [Pa]
        For brass: 107.7 GPa

    Ey : float
        Young's modulus in y direction [Pa]
        For brass: 126.5 GPa

    poisson_ratio : float
        Poisson's ratio
        For brass: 0.33

    density : float
        Material density [kg/m³]
        For brass: 8500

    thickness : float
        Plate thickness [m]
        Typical value: 0.8 mm

    width, length : float
        Plate dimensions [m]

    The flexural rigidities Dx, Dy are given by:

    D_x,y = E_x,y * h³ / (12(1-ν²))

    where ν is the Poisson ratio and h is thickness.
    """
    Ex: Q_ = Q_(107.7, 'GPa')  # Young's modulus x
    Ey: Q_ = Q_(126.5, 'GPa')  # Young's modulus y
    poisson_ratio: float = 0.33
    density: Q_ = Q_(8500, 'kg/m^3')
    shear_modulus: Q_ = Q_(40.15, 'GPa')
    thickness: Q_ = Q_(0.8, 'mm')
    width: Q_ = Q_(280, 'mm')
    length: Q_ = Q_(280, 'mm')

    @property
    def Dx(self) -> Q_:
        """Flexural rigidity x [N⋅m²]"""
        h = self.thickness.to('m').magnitude
        Ex = self.Ex.to('Pa').magnitude
        return Q_(Ex * h ** 3 / (12 * (1 - self.poisson_ratio ** 2)), 'N*m^2')

    @property
    def Dy(self) -> Q_:
        """Flexural rigidity y [N⋅m²]"""
        h = self.thickness.to('m').magnitude
        Ey = self.Ey.to('Pa').magnitude
        return Q_(Ey * h ** 3 / (12 * (1 - self.poisson_ratio ** 2)), 'N*m^2')

    @property
    def mass_per_area(self) -> Q_:
        """Mass per unit area [kg/m²]"""
        return self.density * self.thickness.to('m')

    @property
    def delta(self) -> float:
        """Asymmetry parameter"""
        return abs(self.width.magnitude - self.length.magnitude) / max(self.width.magnitude, self.length.magnitude)


class PhysicalChladni:
    """
    Simulation of Chladni figures based on plate vibration physics.

    The class implements solutions to the biharmonic eigenvalue equation. For a square
    plate with free edges, the eigenfunctions can be approximated as:

    w(x,y) = Σ A_mn * φ_m(x)φ_n(y)

    where φ_m, φ_n are beam functions satisfying:

    d⁴φ/dx⁴ = k⁴φ

    with free-edge boundary conditions.

    Parameters
    ----------
    size : int
        Grid size for discretization

    delta : float
        Asymmetry parameter quantifying elastic anisotropy
        From Tuan et al: δ = (B(θ+π/2)/B(θ))^(1/4) - 1
        where B(θ) describes orientation-dependent elasticity

    omega_o : float
        Reference frequency [rad/s]

    gamma : float
        Damping coefficient

    properties : PlateProperties
        Physical properties of the plate
    """
    def __init__(self, size=500, delta=0.022, omega_o=104, gamma=16.64, properties=None):
        self.size = size
        self.delta = delta if properties is None else properties.delta
        self.omega_o = omega_o
        self.gamma = gamma
        self.properties = properties or PlateProperties()

        # Grid setup
        self.x = np.linspace(-1, 1, size)
        self.y = np.linspace(-1, 1, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.xd, self.yd = 0, 0  # Center driving point

    def compute_eigenfunction(self, n1, n2, X, Y):
        """
        Compute eigenfunction for given mode numbers (n1,n2).

        For a square plate with free edges, the eigenfunctions are:

        φ_mn(x,y) = (2/π) * cos(n1πx/2) * cos(n2πy/2)

        Parameters
        ----------
        n1, n2 : int
            Mode numbers in x and y directions
        X, Y : ndarray
            Coordinate grids

        Returns
        -------
        ndarray
            Eigenfunction values on grid
        """
        return (2 / np.pi) * np.cos(n1 * np.pi * X / 2) * np.cos(n2 * np.pi * Y / 2)

    def compute_eigenvalue(self, n1, n2):
        """
        Compute eigenvalue for given mode numbers (n1,n2).

        For a square plate with free edges:

        λ_mn = (π/2)² * ((1-δ)n1² + (1+δ)n2²)

        where δ is the asymmetry parameter.

        Parameters
        ----------
        n1, n2 : int
            Mode numbers

        Returns
        -------
        float
            Eigenvalue
        """
        return (np.pi / 2) ** 2 * ((1 - self.delta) * n1 ** 2 + (1 + self.delta) * n2 ** 2)

    def natural_frequency(self, n1, n2):
        """
        Calculate natural frequency for given mode numbers (n1,n2).

        From Tuan et al. (2018), the natural frequency is related to eigenvalues by:

        ω_mn = ω_o * sqrt(λ_mn)

        where ω_o = sqrt(B*h)/(ρπ/a) is a reference frequency determined by
        plate properties B (elastic modulus), h (thickness), ρ (density),
        and a (plate dimension).

        Parameters
        ----------
        n1, n2 : int
            Mode numbers in x and y directions

        Returns
        -------
        float
            Natural frequency in Hz
        """
        return self.omega_o * np.sqrt(self.compute_eigenvalue(n1, n2))

    def find_mode_range(self, target_freq, margin=1.2):
        """
        Determine required mode numbers for accurate response calculation.

        Following Tuan et al. (2018), the mode range needs to extend beyond
        the target frequency to ensure proper convergence. The mode range is
        determined by solving:

        ω_max = margin * target_freq

        where ω_max is the natural frequency for mode (max_mode, max_mode).

        Parameters
        ----------
        target_freq : float
            Target frequency in Hz
        margin : float, optional
            Safety factor for mode range, default 1.2

        Returns
        -------
        int
            Maximum mode number required
        """
        omega = 2 * np.pi * target_freq
        max_mode = 1
        while True:
            freq = self.natural_frequency(max_mode, max_mode) / (2 * np.pi)
            if freq > target_freq * margin:
                break
            max_mode += 1
        return max_mode

    def compute_response(self, frequency):
        """
        Compute plate response at given driving frequency.

        The response is given by:

        w(x,y,ω) = Σ C_mn(ω) * φ_mn(x,y)

        where the coefficients are:

        C_mn = α*Q*(m_d/m_p) * φ_mn(x',y') /
               (λ_mn - (ω/ω_o)² - iγ/ω)*(1 + Ξ(x',y';ω))

        Parameters
        ----------
        frequency : float
            Driving frequency [Hz]

        Returns
        -------
        ndarray
            Complex response amplitudes on grid
        tuple
            Mode contributions (n1, n2, natural_freq, weight)
        """
        omega = 2 * np.pi * frequency

        # Find required mode range
        max_mode = self.find_mode_range(frequency)
        print(f"Using modes up to {max_mode} for {frequency} Hz")

        response = np.zeros((self.size, self.size), dtype=complex)

        # First compute Xi - KEEP ORIGINAL RESONANCE FORMULA
        Xi = 0
        for n1 in range(1, max_mode + 1):
            for n2 in range(1, max_mode + 1):
                eigenvalue = self.compute_eigenvalue(n1, n2)
                eigenfunction = self.compute_eigenfunction(n1, n2, self.xd, self.yd)
                denominator = eigenvalue - (omega / self.omega_o) ** 2 - 1j * self.gamma / omega
                Xi += 4 * eigenfunction ** 2 / denominator

        contributions = []

        # Then compute response - KEEP ORIGINAL RESONANCE FORMULA
        for n1 in range(1, max_mode + 1):
            for n2 in range(1, max_mode + 1):
                eigenvalue = self.compute_eigenvalue(n1, n2)
                eigenfunction_d = self.compute_eigenfunction(n1, n2, self.xd, self.yd)
                eigenfunction = self.compute_eigenfunction(n1, n2, self.X, self.Y)

                denominator = eigenvalue - (omega / self.omega_o) ** 2 - 1j * self.gamma / omega
                C_n1_n2 = eigenfunction_d / (denominator * (1 + Xi))

                response += C_n1_n2 * eigenfunction

                natural_freq = self.natural_frequency(n1, n2) / (2 * np.pi)
                weight = abs(C_n1_n2)

                if weight > 1e-10:  # Only store significant contributions
                    contributions.append((n1, n2, natural_freq, weight))

        return response, contributions

    def analyze_frequency(self, frequency: float) -> None:
        """
        Analyze response and mode contributions at given frequency.

        Following the modal analysis approach in Tuan et al. (2018), this method:
        1. Computes total response using eigenfunction expansion
        2. Calculates relative contributions of each mode
        3. Identifies dominant modes based on weighting coefficients

        The weighting coefficient for mode (m,n) is:

        C_mn = φ_mn(x',y') /
               (λ_mn - (ω/ω_o)² - iγ/ω)*(1 + Ξ(x',y';ω))

        Parameters
        ----------
        frequency : float
            Analysis frequency in Hz
        """
        print(f"\nAnalyzing frequency: {frequency} Hz")
        print("------------------------")

        _, contributions = self.compute_response(frequency)
        contributions.sort(key=lambda x: x[3], reverse=True)

        f11 = self.natural_frequency(1, 1) / (2 * np.pi)
        print(f"Fundamental (1,1) frequency: {f11:.1f} Hz")
        print(f"Input frequency is {frequency / f11:.2f} × fundamental")

        print("\nTop contributing modes:")
        total_weight = sum(c[3] for c in contributions)
        for n1, n2, f_nat, weight in contributions[:5]:
            rel_weight = weight / total_weight * 100
            print(f"Mode ({n1},{n2}): {f_nat:.1f} Hz, contribution: {rel_weight:.1f}%")

    def plot_pattern(self, frequency: float) -> None:
        """
        Plot Chladni figure pattern at given frequency.

        As described in Miljković (2021), Chladni patterns are formed
        by vibrational nodes where |w(x,y)| = 0. For each frequency,
        pattern is normalized to maximum amplitude:

        pattern = |w(x,y)| / max|w(x,y)|

        The nodes appear as dark lines where sand/powder would accumulate
        in physical experiments.

        Parameters
        ----------
        frequency : float
            Frequency in Hz for pattern generation
        """
        response, _ = self.compute_response(frequency)
        pattern = np.abs(response)

        if np.max(pattern) > 0:
            pattern /= np.max(pattern)

        plt.figure(figsize=(10, 10))
        colors = [(0, 0, 0), (1, 1, 1)]
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)

        plt.imshow(pattern,
                   cmap=custom_cmap,
                   extent=[-1, 1, -1, 1],
                   interpolation='bilinear')

        plt.title(f'Pattern at {frequency:.1f} Hz')
        plt.axis('off')
        plt.show()


def main():
    # Create plate with physical properties
    props = PlateProperties()
    plate = PhysicalChladni(properties=props)

    # Test frequencies including higher ones
    test_freqs = [36.8, 110, 220, 440, 880, 1356]  # Hz

    for freq in test_freqs:
        plate.analyze_frequency(freq)
        plate.plot_pattern(freq)


if __name__ == "__main__":
    main()