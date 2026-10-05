"""Response of the observables to internal variability, and the covariance it induces.

In the weak-constraint model `pco2geowc_reg_noic` internal variability is a set of
control variables. Here it is not estimated; instead its effect on the observables
is folded into the observation-error covariance. That is possible because, for
fixed parameters theta, the observables are *exactly* affine in the noise:

    y(theta, xi) = m(theta) + L(theta) xi

with m the noiseless trajectory and xi = [global block, q_R1, q_R2], each block of
length M = N - 1 (no noise at t = 0, as in noic). The induced covariance is then

    S(theta) = R_o + L(theta) Q_xi L(theta)^T.

Two designs differ only in how the global noise enters T1:

    flux  T1 <- T1 + dt/C1 (F - L T1 + G EPS (T2 - T1) + q)     heat-flux kick
    temp  T1 <- T1 + dt/C1 (F - L T1 + G EPS (T2 - T1)) + eps_T  temperature, K

T2 receives no noise term of its own in either design; it feels T1's noise only
through the physical coupling G (T1 - T2). The regional noise q_R enters the
regional temperatures directly and is identical in both designs.

With A the two-layer step matrix and b the injection vector, the response of
[T1, T2] to a unit kick at step k, n steps later, is h_n = A^n b. L is built
from h (and its parameter derivatives from dh), so it is exact rather than a
linearization, and it depends only on L, G, EPS, C1, C2, ALPHA_R1, ALPHA_R2.

Adam Michael Bauer
UChicago
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, toeplitz

DESIGNS = ("flux", "temp")

# control-vector indices, in the order dynamics.get_nonlin_path unpacks them
I_L, I_G, I_EPS, I_C1, I_C2 = 5, 6, 7, 8, 9
I_ALPHA = (11, 12)

# the controls L(theta) depends on. every other control (ICs, F1_CO2, the betas)
# enters the cost only through the noiseless trajectory
DYN_PARAMS = (I_L, I_G, I_EPS, I_C1, I_C2)
L_PARAMS = DYN_PARAMS + I_ALPHA

# observable blocks in the stacked, observable-major observation vector
OBS_VARS = ("T1", "Q", "T_R1", "T_R2")
N_OBS = len(OBS_VARS)


def check_design(design):
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")


def step_matrix(theta, DT):
    """Two-layer step matrix A: [T1, T2]_t = A [T1, T2]_{t-1} + (forcing)."""

    L, G, EPS, C1, C2 = theta[I_L], theta[I_G], theta[I_EPS], theta[I_C1], theta[I_C2]
    return np.array(
        [
            [1.0 - DT * (L + G * EPS) / C1, DT * G * EPS / C1],
            [DT * G / C2, 1.0 - DT * G / C2],
        ]
    )


def injection(theta, design, DT):
    """How one unit of global noise enters [T1, T2] in a single step.

    The second component is zero in both designs: T2 has no noise term.
    """

    check_design(design)
    if design == "flux":
        return np.array([DT / theta[I_C1], 0.0])
    return np.array([1.0, 0.0])


def step_matrix_derivs(theta, design, DT):
    """dA/dtheta_k and db/dtheta_k for the five dynamical parameters.

    Returns
    -------
    derivs: dict
        k -> (dA, db)
    """

    L, G, EPS, C1, C2 = theta[I_L], theta[I_G], theta[I_EPS], theta[I_C1], theta[I_C2]
    zero_b = np.zeros(2)

    dA = {
        I_L: np.array([[-DT / C1, 0.0], [0.0, 0.0]]),
        I_G: np.array([[-DT * EPS / C1, DT * EPS / C1], [DT / C2, -DT / C2]]),
        I_EPS: np.array([[-DT * G / C1, DT * G / C1], [0.0, 0.0]]),
        I_C1: np.array(
            [[DT * (L + G * EPS) / C1**2, -DT * G * EPS / C1**2], [0.0, 0.0]]
        ),
        I_C2: np.array([[0.0, 0.0], [-DT * G / C2**2, DT * G / C2**2]]),
    }

    # the injection depends on theta only for the heat-flux kick, through 1/C1
    db = {k: zero_b for k in DYN_PARAMS}
    if design == "flux":
        db[I_C1] = np.array([-DT / C1**2, 0.0])

    return {k: (dA[k], db[k]) for k in DYN_PARAMS}


def impulse_response(A, b, n):
    """h[j] = A^j b for j = 0..n-1, shape (n, 2)."""

    h = np.empty((n, 2))
    h[0] = b
    for j in range(1, n):
        h[j] = A @ h[j - 1]
    return h


def impulse_response_deriv(A, h, dA, db):
    """dh[j] = d(A^j b)/dtheta, from dh_0 = db, dh_j = A dh_{j-1} + dA h_{j-1}."""

    dh = np.empty_like(h)
    dh[0] = db
    for j in range(1, len(h)):
        dh[j] = A @ dh[j - 1] + dA @ h[j - 1]
    return dh


def _toe(c):
    """Lower-triangular Toeplitz matrix with first column c."""

    r = np.zeros_like(c)
    r[0] = c[0]
    return toeplitz(c, r)


def _stack_blocks(N, T1, Q, TR1, TR2):
    """Stack four (M, M) response blocks into the (4N, M) observable-major layout.

    Row t = 0 of every observable stays zero: the initial condition is the control
    vector verbatim, so no noise reaches it.
    """

    M = N - 1
    out = np.zeros((N_OBS * N, M))
    for b, blk in enumerate((T1, Q, TR1, TR2)):
        out[b * N + 1 : (b + 1) * N] = blk
    return out


def global_response(theta, design, N, DT):
    """L_g: response of the four observables to the global noise block, (4N, M).

    Entry (t, k) is the response at time t to the noise applied in the step
    k -> k + 1, i.e. it is nonzero only for t >= k + 1, with n = t - 1 - k.
    """

    A = step_matrix(theta, DT)
    h = impulse_response(A, injection(theta, design, DT), N - 1)
    C1, C2 = theta[I_C1], theta[I_C2]
    a1, a2 = theta[I_ALPHA[0]], theta[I_ALPHA[1]]

    H1, H2 = _toe(h[:, 0]), _toe(h[:, 1])
    return _stack_blocks(N, H1, C1 * H1 + C2 * H2, a1 * H1, a2 * H1)


def global_response_derivs(theta, design, N, DT):
    """dL_g/dtheta_k for every k in L_PARAMS; all other controls give exactly 0.

    Returns
    -------
    derivs: dict
        k -> (4N, M) array
    """

    A = step_matrix(theta, DT)
    h = impulse_response(A, injection(theta, design, DT), N - 1)
    C1, C2 = theta[I_C1], theta[I_C2]
    a1, a2 = theta[I_ALPHA[0]], theta[I_ALPHA[1]]
    H1, H2 = _toe(h[:, 0]), _toe(h[:, 1])
    Z = np.zeros_like(H1)

    out = {}
    for k, (dA, db) in step_matrix_derivs(theta, design, DT).items():
        dh = impulse_response_deriv(A, h, dA, db)
        dH1, dH2 = _toe(dh[:, 0]), _toe(dh[:, 1])

        # Q = C1 T1 + C2 T2: the product rule adds the response itself when the
        # parameter is one of the heat capacities
        dQ = C1 * dH1 + C2 * dH2
        if k == I_C1:
            dQ = dQ + H1
        elif k == I_C2:
            dQ = dQ + H2

        out[k] = _stack_blocks(N, dH1, dQ, a1 * dH1, a2 * dH1)

    # the pattern scalings enter only their own regional rows, linearly
    out[I_ALPHA[0]] = _stack_blocks(N, Z, Z, H1, Z)
    out[I_ALPHA[1]] = _stack_blocks(N, Z, Z, Z, H1)

    return out


class NoiseModel:
    """The theta-independent pieces of the induced observation-error covariance.

    S(theta) = D + L_g(theta) Q_g L_g(theta)^T, where D is diagonal: the
    measurement-error variances plus, on the regional rows for t >= 1, the
    regional noise variance. The regional noise columns of L are a shifted
    identity, so they never depend on theta and fold into D exactly. Only the
    global block has to be carried as a low-rank term.

    Parameters
    ----------
    design: str
        "flux" or "temp"

    sig_obs: (4,) array
        measurement-error std of T1, Q, T_R1, T_R2

    sig_reg: (2,) array
        std of the white regional noise q_R1, q_R2

    Q_g: (M, M) array
        prior covariance of the global noise block, in the design's units

    N: int
        number of timesteps in the window (M = N - 1 noise steps)
    """

    def __init__(self, design, sig_obs, sig_reg, Q_g, N, DT=1.0):
        check_design(design)
        self.design = design
        self.N = int(N)
        self.M = self.N - 1
        self.DT = float(DT)
        self.sig_obs = np.asarray(sig_obs, dtype=float)
        self.sig_reg = np.asarray(sig_reg, dtype=float)
        self.Q_g = np.asarray(Q_g, dtype=float)

        if self.Q_g.shape != (self.M, self.M):
            raise ValueError(
                f"Q_g must be ({self.M}, {self.M}) for N = {self.N}, got {self.Q_g.shape}"
            )

        D = np.repeat(self.sig_obs**2, self.N)
        for r in range(2):
            b = 2 + r
            D[b * self.N + 1 : (b + 1) * self.N] += self.sig_reg[r] ** 2
        self.D = D
        self.sqrtD = np.sqrt(D)
        self.logdet_D = float(np.sum(np.log(D)))

        # Q_g = Cq Cq^T, factored once per window
        self.Cq = cholesky(self.Q_g, lower=True)

    # -- L applied to and from vectors ---------------------------------------

    def L_g(self, theta):
        return global_response(theta, self.design, self.N, self.DT)

    def dL_g(self, theta):
        return global_response_derivs(theta, self.design, self.N, self.DT)

    def apply_L(self, L_g, xi):
        """L xi, for the full noise vector xi = [global, q_R1, q_R2]."""

        M, N = self.M, self.N
        out = L_g @ xi[:M]
        out[2 * N + 1 : 3 * N] += xi[M : 2 * M]
        out[3 * N + 1 : 4 * N] += xi[2 * M : 3 * M]
        return out

    def apply_QLt(self, L_g, v):
        """Q_xi L^T v, which is what the envelope identity needs for xi*."""

        M, N = self.M, self.N
        return np.hstack(
            [
                self.Q_g @ (L_g.T @ v),
                self.sig_reg[0] ** 2 * v[2 * N + 1 : 3 * N],
                self.sig_reg[1] ** 2 * v[3 * N + 1 : 4 * N],
            ]
        )

    def Q_blocks(self):
        """Full prior covariance of xi, block-diagonal (global, q_R1, q_R2)."""

        M = self.M
        Q = np.zeros((3 * M, 3 * M))
        Q[:M, :M] = self.Q_g
        Q[M : 2 * M, M : 2 * M] = np.eye(M) * self.sig_reg[0] ** 2
        Q[2 * M :, 2 * M :] = np.eye(M) * self.sig_reg[1] ** 2
        return Q

    def factor(self, L_g):
        return Factor(self, L_g)


class Factor:
    """Woodbury factorization of S = D + L_g Q_g L_g^T.

    Working in the M-dimensional global-noise subspace keeps everything
    well-conditioned regardless of units: with G = D^(-1/2) L_g Cq,
    S = D^(1/2) (I + G G^T) D^(1/2), and K = I + G^T G is symmetric with every
    eigenvalue >= 1, even though sigma_Q is ~200 times sigma_T1.
    """

    def __init__(self, noise, L_g):
        if not np.all(np.isfinite(L_g)):
            raise np.linalg.LinAlgError("non-finite noise response")
        self.noise = noise
        self.L_g = L_g
        self.G = (L_g @ noise.Cq) / noise.sqrtD[:, None]
        K = np.eye(noise.M) + self.G.T @ self.G
        if not np.all(np.isfinite(K)):
            # the response overflowed: the dynamics are far outside the stable
            # region. one error type for every failure mode keeps callers simple
            raise np.linalg.LinAlgError("noise covariance overflowed")
        self.Kc = cho_factor(K, lower=True)
        self.logdet = noise.logdet_D + 2.0 * float(np.sum(np.log(np.diag(self.Kc[0]))))

    def solve_quad(self, d):
        """S^{-1} d and d^T S^{-1} d, with the quadratic form non-negative by construction.

        Computing d^T (S^{-1} d) through the Woodbury subtraction
        D^{-1} d - D^{-1/2} G K^{-1} G^T D^{-1/2} d cancels catastrophically when
        G is huge, which happens when the optimizer probes parameters with
        unstable dynamics. Rounding can then make the "cost" large and negative,
        and the optimizer is drawn toward it. The identical split

            d^T S^{-1} d = |r_w|^2 + |z|^2,   z = K^{-1} G^T y_w,  r_w = y_w - G z,

        with y_w = D^{-1/2} d, is a sum of squares, so it cannot go negative.
        S^{-1} d = D^{-1/2} r_w follows from the same pieces.
        """

        y_w = d / self.noise.sqrtD
        z = cho_solve(self.Kc, self.G.T @ y_w)
        r_w = y_w - self.G @ z
        return r_w / self.noise.sqrtD, float(r_w @ r_w + z @ z)

    def solve(self, x):
        """S^{-1} x for a vector (4N,) or a matrix (4N, m)."""

        sD = self.noise.sqrtD
        D = self.noise.D
        if x.ndim == 1:
            z = cho_solve(self.Kc, self.G.T @ (x / sD))
            return x / D - (self.G @ z) / sD
        z = cho_solve(self.Kc, self.G.T @ (x / sD[:, None]))
        return x / D[:, None] - (self.G @ z) / sD[:, None]

    def dense(self):
        """S itself, (4N, 4N). Only used for archiving and checks."""

        return np.diag(self.noise.D) + self.L_g @ self.noise.Q_g @ self.L_g.T
