"""2D initial conditions: hot (noise), log-normal hot, perfect crystal,
crystal with vacancies, dislocation ring.

Provenance: log/gpu-spectral-base-implementation/_sim_initial_conditions.py,
byte-identical in all three source forks.

The two generators that draw randomness now take their stream from
`np.random.default_rng(seed)` rather than seeding global NumPy state. The
`seed=None` default is unchanged from the source - the guard it replaced was
`if seed is not None: np.random.seed(seed)`. Note that `default_rng(n)` and
`np.random.seed(n)` produce different streams, so a run seeded under the
source forks does not reproduce here.
"""
import numpy as np
from pfc_models.backends.numpy_fftw import npFloatType


def GenerateHotInitialConditions(geometry, meanDensity, stdDeviation, kMax, seed=None):
    rng = np.random.default_rng(seed)

    hotPhi = rng.normal(loc=0, scale=1.0, size=geometry.X.shape)
    hotPhi_hat = np.fft.rfft2(hotPhi)

    # filter out high frequency modes
    hotPhi_hat[np.sqrt(geometry.KX**2 + geometry.KY**2) > kMax**2] = 0

    # return to real space and ensure positivity
    hotPhi = np.fft.irfft2(hotPhi_hat, s=geometry.X.shape)

    # set std to scale
    hotPhi *= stdDeviation / hotPhi.std()

    # ensure positivity
    if hotPhi.min() <= 0:
        hotPhi -= hotPhi.min()

    # ensure meanDensity
    hotPhi *= meanDensity / hotPhi.mean()
    return hotPhi.astype(npFloatType)


def GenerateLogNormalHotInitialConditions(
    geometry, meanDensity, stdDeviation, kMax, seed=None
):
    rng = np.random.default_rng(seed)

    # 1. log-normal parameters
    tau2 = np.log(1 + stdDeviation**2)
    tau = np.sqrt(tau2)
    mu = np.log(meanDensity) - tau2 / 2

    # 2. white noise
    Z = rng.normal(0.0, 1.0, size=geometry.X.shape).astype(npFloatType)

    # 3. FFT
    Zhat = np.fft.rfft2(Z)
    K = np.sqrt(geometry.KX**2 + geometry.KY**2)

    # 4. filter
    H = np.exp(-((K / kMax) ** 2))  # smooth low-pass

    Yhat = Zhat * H

    # 4. inverse FFT -> smooth Gaussian field
    Y_raw = np.fft.irfft2(Yhat).real

    # 5. rescale to variance tau^2 and mean 0
    Y_mean = Y_raw.mean()
    Y_var = Y_raw.var()
    Y = (Y_raw - Y_mean) * (tau / np.sqrt(Y_var))

    # 6. exponentiate to get log-normal field
    X = np.exp(mu + Y)
    X_mean = X.mean()
    X *= meanDensity / X_mean  # rescale to desired mean
    return X.astype(npFloatType)


def GeneratePerfectCrystalInitialConditions(geometry, meanDensity, A0, q0 = 1.0):
    # Generate a perfect crystal initial condition based on the given parameters.
    # The crystal is defined by its lattice spacing (a, h), mean density (meanDensity),
    # and the parameters psi0 and A0 which define the amplitude and phase of the crystal.

    # 1-mode vectors
    q1 = q0 * np.array([-(3**0.5) / 2, -0.5])
    q2 = q0 * np.array([3**0.5 / 2, -0.5])
    q3 = q0 * np.array([0, 1])

    # Create a meshgrid for the spatial domain
    X, Y = geometry.X, geometry.Y

    # Calculate the crystal structure using a simple sinusoidal function
    phi = (
        meanDensity
        + A0 * np.exp(1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(-1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(-1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(1.0j * (q3[0] * X + q3[1] * Y))
        + A0 * np.exp(-1.0j * (q3[0] * X + q3[1] * Y))
    ).real

    return phi.astype(npFloatType)


def GeneratePerfectCrystalInitialWithCentralVacancy(geometry, meanDensity_S, meanDensity_V, A0, q0 = 1.0):
    # Generate a perfect crystal initial condition based on the given parameters.
    # The crystal is defined by its lattice spacing (a, h), mean density (meanDensity),
    # and the parameters psi0 and A0 which define the amplitude and phase of the crystal.

    # 1-mode vectors
    q1 = q0 * np.array([-(3**0.5) / 2, -0.5])
    q2 = q0 * np.array([3**0.5 / 2, -0.5])
    q3 = q0 * np.array([0, 1])

    # Create a meshgrid for the spatial domain
    X, Y = geometry.X, geometry.Y

    a = 4 * np.pi / 3**0.5 / q0
    h = 4 * np.pi / q0
    mx = geometry.Lx / a
    my = geometry.Ly / h
    mx_center = mx // 2
    my_center = my // 2
    x_center = mx_center * a
    y_center = my_center * h

    # Calculate the crystal structure using a simple sinusoidal function
    phi = (
        meanDensity_S
        + A0 * np.exp(1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(-1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(-1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(1.0j * (q3[0] * X + q3[1] * Y))
        + A0 * np.exp(-1.0j * (q3[0] * X + q3[1] * Y))
    ).real

    phi[((X - x_center) ** 2 + (Y - y_center) ** 2) < (a**2/4)] = meanDensity_V

    return phi.astype(npFloatType)


def GeneratePerfectCrystalInitialWithVancies(geometry, meanDensity_S, meanDensity_V, vacancyCoordsSet, A0):
    # Generate a perfect crystal initial condition based on the given parameters.
    # The crystal is defined by its lattice spacing (a, h), mean density (meanDensity),
    # and the parameters psi0 and A0 which define the amplitude and phase of the crystal.

    # 1-mode vectors
    q1 = np.array([-(3**0.5) / 2, -0.5])
    q2 = np.array([3**0.5 / 2, -0.5])
    q3 = np.array([0, 1])

    # Create a meshgrid for the spatial domain
    X, Y = geometry.X, geometry.Y

    a = 4*np.pi/3**0.5
    h = 4*np.pi

    # Calculate the crystal structure using a simple sinusoidal function
    phi = (
        meanDensity_S
        + A0 * np.exp(1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(-1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(-1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(1.0j * (q3[0] * X + q3[1] * Y))
        + A0 * np.exp(-1.0j * (q3[0] * X + q3[1] * Y))
    ).real

    for vacancyCoords in vacancyCoordsSet:
      mx_center, my_center = vacancyCoords
      x_center = mx_center * a
      y_center = my_center * h

      phi[((X - x_center) ** 2 + (Y - y_center) ** 2) < (a**2/4)] = meanDensity_V

    return phi.astype(npFloatType)


def GenerateDislocationRingInitialConditions(geometry, meanDensity, A0, radius, theta):
    # Generate a perfect crystal initial condition based on the given parameters.
    # The crystal is defined by its lattice spacing (a, h), mean density (meanDensity),
    # and the parameters psi0 and A0 which define the amplitude and phase of the crystal.
    rCenter = np.array([geometry.Lx / 2, geometry.Ly / 2])  # center of the disk
    Rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )  # rotation matrix

    # 1-mode vectors
    q1 = np.array([-(3**0.5) / 2, -0.5])
    q2 = np.array([3**0.5 / 2, -0.5])
    q3 = np.array([0, 1])

    # Create a meshgrid for the spatial domain
    X, Y = geometry.X, geometry.Y

    # Calculate the crystal structure using a simple sinusoidal function
    phi = (
        meanDensity
        + A0 * np.exp(1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(-1.0j * (q1[0] * X + q1[1] * Y))
        + A0 * np.exp(1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(-1.0j * (q2[0] * X + q2[1] * Y))
        + A0 * np.exp(1.0j * (q3[0] * X + q3[1] * Y))
        + A0 * np.exp(-1.0j * (q3[0] * X + q3[1] * Y))
    ).real

    # Rotated 1-mode vectors
    q1_ring = Rot @ q1
    q2_ring = Rot @ q2
    q3_ring = Rot @ q3
    phi_rotation = (
        meanDensity
        + A0 * np.exp(1.0j * (q1_ring[0] * X + q1_ring[1] * Y))
        + A0 * np.exp(-1.0j * (q1_ring[0] * X + q1_ring[1] * Y))
        + A0 * np.exp(1.0j * (q2_ring[0] * X + q2_ring[1] * Y))
        + A0 * np.exp(-1.0j * (q2_ring[0] * X + q2_ring[1] * Y))
        + A0 * np.exp(1.0j * (q3_ring[0] * X + q3_ring[1] * Y))
        + A0 * np.exp(-1.0j * (q3_ring[0] * X + q3_ring[1] * Y))
    ).real

    # Apply the rotation to the initial condition
    phi[((X - rCenter[0]) ** 2 + (Y - rCenter[1]) ** 2) < radius**2] = phi_rotation[
        ((X - rCenter[0]) ** 2 + (Y - rCenter[1]) ** 2) < radius**2
    ]

    return phi.astype(npFloatType)
