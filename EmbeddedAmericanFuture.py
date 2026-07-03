from numba import njit
import numpy as np
import time
import matplotlib.pyplot as plt

"""
Under the risk-neutral measure, the spot price follows the
regime-switching mean-reverting stochastic differential equation

    dS = kappa * (theta * exp(mu * t) - S) * dt
         + sigma(X) * S * dW,

where X ∈ {L,H} denotes the volatility regime.

The mean reversion level is time dependent and given by
    theta * exp(mu * t).

The volatility depends on the current regime:
    sigma(L) = sigma_L
    sigma(H) = sigma_H.

The value functions U_L and U_H satisfy the coupled
variational inequalities
    max(
        dU_L/dt + L_L[U_L] - rU_L
        + q_LH(U_H - U_L),
        G(S,t) - U_L
    ) = 0,

    max(
        dU_H/dt + L_H[U_H] - rU_H
        + q_HL(U_L - U_H),
        G(S,t) - U_H
    ) = 0.

During the early-exercise window t ≤ T1, the holder may
exchange the current contract for the alternative payoff
    G(S,t)
      = K(1-exp(-r(T-t)))
        - S(1-exp(-(r+kappa)(T-t))).

For t > T1, early exercise is not permitted and the
contract behaves as a European regime-switching claim.

The PDE system is solved using an explicit Euler finite
difference method together with fixed-point iteration to
handle the regime-switching coupling terms.
"""


# Drift and Switching functions
@njit
def p(S, t, kappa, theta, mu):

    """
    Mean-reverting drift appearing in the spot-price process.
    The drift is
        p(S,t)
          = kappa * (theta * exp(mu*t) - S).

    Positive values push the process upwards towards the
    long-run level, while negative values pull it downwards.
    """

    return kappa * (theta * np.exp(mu*t) - S)


@njit
def normal_cdf(x):

    """
    Fast approximation of the standard normal cumulative
    distribution function.

    Numba does not support scipy.stats.norm.cdf, so a smooth
    approximation based on the hyperbolic tangent is used.

    This function is required when evaluating the
    state-dependent regime-switching intensities.
    """

    return 0.5 * (
        1.0
        + np.tanh(
            0.7978845608 * x
            * (1.0 + 0.044715 * x * x)
        )
    )


@njit
def q(S, t, choice, alpha, beta, theta, mu, z):

    """
    State-dependent regime-switching intensity.

    Over a small time interval dt,
        P(L → H) = q_LH(S,t) dt,
        P(H → L) = q_HL(S,t) dt.

    Define
        Phi= N((ln(S)-ln(theta)-mu*t)/z),
    where N denotes the standard normal cumulative
    distribution function.

    The switching intensities are

        q_LH = alpha + beta * (1-Phi),

        q_HL = alpha * Phi + beta.

    The probability of switching therefore depends on
    both the current spot level and time.
    """

    if S < 1e-12:

        Phi = 0.0

    else:

        Phi = normal_cdf(
            (np.log(S) - np.log(theta) - mu * t) / z
        )

    # Low-volatility regime switching to high volatility
    if choice == 0:

        return alpha + beta * (1.0 - Phi)

    # High-volatility regime switching to low volatility
    else:

        return alpha * Phi + beta

#Explicit Euler Coefficients

"""
The differential operator appearing in the pricing PDE is

    L_X[U]
      = p(S,t) * dU/dS
        + 0.5 * sigma(X)^2 * S^2 * d^2U/dS^2,
where X ∈ {L,H}.

Using central differences in space and explicit Euler
time-stepping gives update coefficients
    A_j : coefficient of U_{j-1}
    B_j : coefficient of U_j
    C_j : coefficient of U_{j+1}

The only difference between the two regimes is the
volatility parameter sigma_L or sigma_H.
"""


@njit
def A(
    i, dt, j, dS, choice,
    kappa, theta, mu,
    sigma_L, sigma_H
):

    """
    Coefficient multiplying the left neighbour U_{j-1}.
    This coefficient contains contributions from both the
    drift and diffusion terms.
    """

    S = j * dS
    t = i * dt

    if choice == 0:
        return (
            0.5 * dt
            * (
                -p(S, t, kappa, theta, mu)/dS
                + sigma_L**2 * j**2
            )
        )

    else:
        return (
            0.5 * dt
            * (
                -p(S, t, kappa, theta, mu)/dS
                + sigma_H**2 * j**2
            )
        )


@njit
def B(
    i, dt, j, dS,
    choice,
    sigma_L, sigma_H
):

    """
    Central coefficient multiplying U_j.

    This term represents the contribution of the identity
    operator in the explicit Euler update.
    """

    if choice == 0:
        return 1.0 - dt * sigma_L**2 * j**2

    else:
        return 1.0 - dt * sigma_H**2 * j**2


@njit
def C(
    i, dt, j, dS, choice,
    kappa, theta, mu,
    sigma_L, sigma_H
):

    """
    Coefficient multiplying the right neighbour U_{j+1}.

    This coefficient contains contributions from both the
    drift and diffusion terms.
    """

    S = j * dS
    t = i * dt

    if choice == 0:
        return (
            0.5 * dt
            * (
                p(S, t, kappa, theta, mu)/dS
                + sigma_L**2 * j**2
            )
        )

    else:
        return (
            0.5 * dt
            * (
                p(S, t, kappa, theta, mu)/dS
                + sigma_H**2 * j**2
            )
        )

# Early Exercise Payoff
@njit
def G(S, t, K, r, T, kappa):

    """
    Early-exercise payoff appearing in the variational
    inequality.

    During the exercise window the holder may replace the
    current contract value by

        G(S,t) = K * (1-exp(-r(T-t)))- S * (1-exp(-(r+kappa) * (T-t))).

    The American constraint requires
        U_L ≥ G,
        U_H ≥ G.

    After each Euler step the numerical solution is projected
    onto this obstacle by enforcing
        U = max(U,G).
    """

    return (
        K * (1 - np.exp(-r * (T - t)))
        - S * (1 - np.exp(-(r + kappa) * (T - t)))
    )

@njit
def EulerSwitching(
    Smax, iMax, jMax, tol, maxiter,
    T, T1, r, kappa, theta, mu,
    sigma_L, sigma_H, alpha, beta,
    gamma, z, K
):

    """
    Prices the American-style regime-switching contract using an
    explicit Euler finite difference method.

    The value satisfies the coupled variational inequalities

        max(dU_L/dt + L_L[U_L] - rU_L + q_LH(U_H - U_L), G - U_L) = 0,

        max(dU_H/dt + L_H[U_H] - rU_H + q_HL(U_L - U_H), G - U_H) = 0,
    where
        G(S,t) = K * (1-exp(-r * (T-t))) - S * (1-exp(-(r+kappa) * (T-t)))

    is the early-exercise payoff.

    For t > T1 the contract behaves as a European contract,
    while for t <= T1 the numerical solution is projected onto
    the payoff to enforce the early-exercise constraint.

    Since the switching terms couple the two PDEs together,
    a fixed-point iteration is performed at every time level.
    """

    # Uniform discretisation of the computational domain.
    dt = T / iMax
    dS = Smax / jMax

    # Spatial grid for the underlying price.
    S = np.zeros(jMax + 1)
    for j in range(jMax + 1):
        S[j] = j * dS

    # Temporal grid.
    t = np.zeros(iMax + 1)
    for i in range(iMax + 1):
        t[i] = i * dt

    """
    Determine the time index corresponding to the end of the
    early-exercise window.

    The interval t > T1 is solved as a European contract, while
    t <= T1 requires the variational inequality to be enforced.
    """
    expiry1 = iMax - int((T - T1) / dt)

    """
    Storage for the numerical solution in each volatility regime.

    vOld_* contains the solution from the previous time level,
    vNew_* stores the current fixed-point iterate, whilst
    v*_temp contains the updated explicit Euler approximation.
    """

    vNew_L = np.zeros(jMax + 1)
    vOld_L = np.zeros(jMax + 1)

    vNew_H = np.zeros(jMax + 1)
    vOld_H = np.zeros(jMax + 1)

    vL_temp = np.zeros(jMax + 1)
    vH_temp = np.zeros(jMax + 1)

    # Storage for the early-exercise payoff.
    G_vals = np.zeros(jMax + 1)

    """
    Terminal condition.

    At maturity the contract has the same payoff as the
    European regime-switching contract,

        V(S,T) = theta if S exp(gamma T) > theta,
                 0 otherwise.

    Since the payoff is independent of the volatility regime,
    both PDEs share the same terminal condition.
    """

    for j in range(jMax + 1):

        if S[j] * np.exp(gamma * T) > theta:

            vNew_L[j] = theta
            vOld_L[j] = theta

            vNew_H[j] = theta
            vOld_H[j] = theta

    """
    First solve the European region of the problem.

    For times t > T1,
    early exercise is not permitted, so the coupled PDEs are
    solved exactly as a European regime-switching contract.
    """

    for i in range(iMax - 1, expiry1 - 1, -1):

        """
        Initialise the fixed-point iteration using the solution
        from the previous time level.

        Since the option value varies smoothly in time, this
        provides an excellent initial guess.
        """

        vNew_L[:] = vOld_L[:]
        vNew_H[:] = vOld_H[:]

        """
        Fixed-point iteration.

        The switching terms
            q_LH(U_H-U_L),
            q_HL(U_L-U_H),
        couple the two PDEs together, so they cannot be solved
        independently.

        At each iteration:
            1. Construct updated explicit Euler values using
               the current estimate of the opposite regime.
            2. Measure the maximum change between successive
               iterates.
            3. Repeat until the prescribed tolerance is reached.
        """

        for iter in range(maxiter):

            # Reset the temporary solution vectors.
            vL_temp[:] = 0.0
            vH_temp[:] = 0.0

            """
            Boundary conditions.

            At S = 0 the degenerate boundary condition is
            obtained directly from the governing PDE.

            At S = Smax the asymptotic behaviour is
                U = theta * exp(-r * (T-t)),
            corresponding to the discounted limiting payoff.
            """

            vL_temp[0] = (
                1 / (1 + dt * (r + q(S[0], t[i], 0, alpha, beta, theta, mu, z)))
            ) * (
                (B(i, dt, 0, dS, 0, sigma_L, sigma_H)
                 + 2 * A(i, dt, 0, dS, 0,
                         kappa, theta, mu,
                         sigma_L, sigma_H))
                * vOld_L[0]
                + 2 * C(i, dt, 0, dS, 0,
                        kappa, theta, mu,
                        sigma_L, sigma_H)
                * vOld_L[1]
                + q(S[0], t[i], 0,
                    alpha, beta, theta, mu, z)
                * dt * vNew_H[0]
            )

            vH_temp[0] = (
                1 / (1 + dt * (r + q(S[0], t[i], 1, alpha, beta, theta, mu, z)))
            ) * (
                (B(i, dt, 0, dS, 1, sigma_L, sigma_H)
                 + 2 * A(i, dt, 0, dS, 1,
                         kappa, theta, mu,
                         sigma_L, sigma_H))
                * vOld_H[0]
                + 2 * C(i, dt, 0, dS, 1,
                        kappa, theta, mu,
                        sigma_L, sigma_H)
                * vOld_H[1]
                + q(S[0], t[i], 1,
                    alpha, beta, theta, mu, z)
                * dt * vNew_L[0]
            )

            vL_temp[jMax] = theta * np.exp(-r * (T - t[i]))
            vH_temp[jMax] = theta * np.exp(-r * (T - t[i]))

            """
            Assemble the explicit Euler update at the interior
            grid points.

            The final term in each equation represents the
            coupling between the low- and high-volatility
            regimes through the state-dependent switching
            intensities.
            """

            for j in range(1, jMax):

                qL = q(S[j], t[i], 0,
                       alpha, beta, theta, mu, z)

                qH = q(S[j], t[i], 1,
                       alpha, beta, theta, mu, z)

                denomL = 1.0 / (1.0 + dt * (r + qL))
                denomH = 1.0 / (1.0 + dt * (r + qH))

                vL_temp[j] = denomL * (
                    A(i, dt, j, dS, 0,
                      kappa, theta, mu,
                      sigma_L, sigma_H) * vOld_L[j - 1]
                    + B(i, dt, j, dS, 0,
                        sigma_L, sigma_H) * vOld_L[j]
                    + C(i, dt, j, dS, 0,
                        kappa, theta, mu,
                        sigma_L, sigma_H) * vOld_L[j + 1]
                    + qL * dt * vNew_H[j]
                )

                vH_temp[j] = denomH * (
                    A(i, dt, j, dS, 1,
                      kappa, theta, mu,
                      sigma_L, sigma_H) * vOld_H[j - 1]
                    + B(i, dt, j, dS, 1,
                        sigma_L, sigma_H) * vOld_H[j]
                    + C(i, dt, j, dS, 1,
                        kappa, theta, mu,
                        sigma_L, sigma_H) * vOld_H[j + 1]
                    + qH * dt * vNew_L[j]
                )

            """
            Measure the convergence of the fixed-point iteration.

            The maximum absolute change over both volatility
            regimes is used as the stopping criterion.
            """

            error = 0.0

            for j in range(jMax + 1):

                err = abs(vL_temp[j] - vNew_L[j])
                if err > error:
                    error = err

                err = abs(vH_temp[j] - vNew_H[j])
                if err > error:
                    error = err

            # Update the current fixed-point iterate.
            vNew_L[:] = vL_temp[:]
            vNew_H[:] = vH_temp[:]

            # Stop once the prescribed tolerance has been reached.
            if error < tol:
                break

        """
        Accept the converged solution and move to the
        previous time level.
        """

        vOld_L[:] = vNew_L[:]
        vOld_H[:] = vNew_H[:]

    """
    Solve the American region of the problem.

    For t <= T1,
    early exercise is permitted, so after solving the coupled
    PDEs the numerical solution is projected onto the
    early-exercise payoff
        U = max(U,G),
    thereby enforcing the variational inequalities.
    """

    for i in range(expiry1, -1, -1):

        """
        Evaluate the early-exercise payoff across the
        spatial grid.

        This payoff is identical for both volatility regimes.
        """

        for j in range(jMax + 1):
            G_vals[j] = G(S[j], t[i], K, r, T, kappa)

        # Initialise the fixed-point iteration.
        vNew_L[:] = vOld_L[:]
        vNew_H[:] = vOld_H[:]

        """
        As in the European region, the regime-switching terms
        couple the two PDEs together, requiring fixed-point
        iteration at every time level.
        """

        for iter in range(maxiter):

            vL_temp[:] = 0.0
            vH_temp[:] = 0.0

            """
            Boundary conditions.

            At S = 0 the option value equals the immediate
            exercise payoff,
                U(0,t) = G(0,t) = K(1-exp(-r(T-t))),
            which agrees with the boundary condition derived
            from the variational inequalities.

            The upper boundary continues to satisfy the
            asymptotic European condition.
            """

            vL_temp[0] = G(S[0], t[i], K, r, T, kappa)
            vH_temp[0] = G(S[0], t[i], K, r, T, kappa)

            vL_temp[jMax] = theta * np.exp(-r * (T - t[i]))
            vH_temp[jMax] = theta * np.exp(-r * (T - t[i]))

            """
            Compute the explicit Euler approximation at each
            interior grid point.

            The only difference from the European region is
            that the resulting solution will subsequently be
            compared with the early-exercise payoff.
            """

            for j in range(1, jMax):

                qL = q(S[j], t[i], 0,
                       alpha, beta, theta, mu, z)

                qH = q(S[j], t[i], 1,
                       alpha, beta, theta, mu, z)

                denomL = 1.0 / (1.0 + dt * (r + qL))
                denomH = 1.0 / (1.0 + dt * (r + qH))

                vL_temp[j] = denomL * (
                    A(i, dt, j, dS, 0,
                      kappa, theta, mu,
                      sigma_L, sigma_H) * vOld_L[j - 1]
                    + B(i, dt, j, dS, 0,
                        sigma_L, sigma_H) * vOld_L[j]
                    + C(i, dt, j, dS, 0,
                        kappa, theta, mu,
                        sigma_L, sigma_H) * vOld_L[j + 1]
                    + qL * dt * vNew_H[j]
                )

                vH_temp[j] = denomH * (
                    A(i, dt, j, dS, 1,
                      kappa, theta, mu,
                      sigma_L, sigma_H) * vOld_H[j - 1]
                    + B(i, dt, j, dS, 1,
                        sigma_L, sigma_H) * vOld_H[j]
                    + C(i, dt, j, dS, 1,
                        kappa, theta, mu,
                        sigma_L, sigma_H) * vOld_H[j + 1]
                    + qH * dt * vNew_L[j]
                )

            """
            Measure convergence of the fixed-point iteration.
            """

            error = 0.0

            for j in range(jMax + 1):

                err = abs(vL_temp[j] - vNew_L[j])
                if err > error:
                    error = err

                err = abs(vH_temp[j] - vNew_H[j])
                if err > error:
                    error = err

            # Update the current iterate.
            vNew_L[:] = vL_temp[:]
            vNew_H[:] = vH_temp[:]

            if error < tol:
                break

        """
        Enforce the early-exercise constraint.
        The variational inequalities require

            U_X(S,t) >= G(S,t),

        for both volatility regimes.

        Numerically this is achieved by projecting the PDE
        solution onto the payoff function,

            U = max(U,G),

        at every spatial grid point.
        """

        for j in range(jMax + 1):

            if G_vals[j] > vNew_L[j]:
                vNew_L[j] = G_vals[j]

            if G_vals[j] > vNew_H[j]:
                vNew_H[j] = G_vals[j]

        # Accept the projected solution and continue
        # marching backwards in time.
        vOld_L[:] = vNew_L[:]
        vOld_H[:] = vNew_H[:]

    """
    After marching backwards through every time level,
    vNew_L and vNew_H contain the numerical approximations
    to the American contract value at the initial time t = 0.
    """

    return S, t, vNew_L, vNew_H

S0 = 4388.6
T = 3
T1 = 1.75
K = 34000
r= 0.0352
kappa = 0.14
theta= 6150
mu= 0.0379
sigma_L = 0.188
sigma_H = 0.455
alpha = 0.253
beta = 0.437
gamma = 0.016
z = 0.605
tol = 1e-6
maxiter = 50

# Testing functions (similar to the European Regime Switching case)

def ESGraph(Smax, iMax, jMax, tol, maxiter):
    
    start = time.perf_counter()

    S, t, vL, vH = EulerSwitching(
        Smax, iMax, jMax, tol, maxiter,
        T, T1, r, kappa, theta, mu,
        sigma_L, sigma_H, alpha, beta, gamma, z, K
    )

    UHS0 = np.interp(S0, S, vH)
    print(f"Option value at S0: {UHS0:.5f}")

    end = time.perf_counter()

    print(f"Execution time: {end - start:.4f} seconds")


    plt.figure()
    plt.plot(S, vL, label='Low Volatility Value')
    plt.plot(S, vH, label='High Volatility Value')
    plt.xlabel('Underlying Price S')
    plt.ylabel('Option Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def SmaxGraph(iMax, jMax, tol, maxiter):

    Smax = np.arange(5000, 20000 + 100, 100)

    UHS = []

    for smax in Smax:
        S, t, vL, vH = EulerSwitching(
            smax, iMax, jMax, tol, maxiter,
            T, T1, r, kappa, theta, mu,
            sigma_L, sigma_H, alpha, beta, gamma, z, K
        )
        print (f"Smax: {smax} has been completed")

        UHS.append(np.interp(S0, S, vH))

    plt.figure()
    plt.plot(Smax, UHS)
    plt.xlabel("Smax")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()

def jMaxData(Smax, iMax, tol, maxiter):

    jMax = [25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000]
    UHS = []

    for k in jMax:
        S, t, vL, vH = EulerSwitching(
            Smax, iMax, k, tol, maxiter,
            T, T1, r, kappa, theta, mu,
            sigma_L, sigma_H, alpha, beta, gamma, z, K
        )
        print (f"jMax: {k} has been completed")
        print(f"vH sample: {vH[:5]}")

        UHS.append(np.interp(S0, S, vH))

    plt.figure()
    plt.plot(jMax, UHS)
    plt.xlabel("jMax")
    plt.ylabel("Option Price")
    plt.grid(True)
    plt.show()

def iMaxData(Smax, jMax, tol, maxiter):

    iMax = [2500, 5000, 10000, 12500, 15000, 20000, 30000, 50000]
    UHS = []

    for i in iMax:
        S, t, vL, vH = EulerSwitching(
            Smax, i, jMax, tol, maxiter,
            T, T1, r, kappa, theta, mu,
            sigma_L, sigma_H, alpha, beta, gamma, z, K
        )
        print (f"iMax: {i} has been completed")
        print(f"vH sample: {vH[:5]}")

        UHS.append(np.interp(S0, S, vH))

    with open("Q2IMax.tex", "w") as f:
        f.write("\\begin{tabular}{|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("iMax & Option Price at S0 & |V_{k } - V_{k-1}| \\\\\n")
        f.write("\\hline\n")
        f.write(f"{iMax[0]} & {UHS[0]} & 0 \\\\\n")
        for j in range(1, len(iMax)):
            line = f"{iMax[j]} & {UHS[j]} & {abs(UHS[j] - UHS[j-1])} \\\\\n"
            f.write(line)
        f.write("\\end{tabular}\n")
        
"""
This function is why njit is used, so we can produce a best estimate within a second,
allowing us to quickly and accurately estimate the option price at t = 0.
"""
def FastestEstimate(Smax, tol, maxiter, C):

    jMax = np.arange(2, 2000 + 1, 1)

    UHS = []
    actual_time = 0
    BestI = 0
    BestJ = 0

    for j in range(len(jMax)):

        iMax = int(C * (jMax[j])**2)

        start = time.perf_counter()

        S, t, vL, vH = EulerSwitching(
            Smax, iMax, jMax[j], tol, maxiter,
            T, T1, r, kappa, theta, mu,
            sigma_L, sigma_H, alpha, beta, gamma, z, K
        )

        UHS0 = np.interp(S0, S, vH)

        end = time.perf_counter()

        t_elapsed = end - start

        if t_elapsed > 1 and j >0:
            break

        actual_time = t_elapsed
        UHS.append(UHS0)
        BestI = iMax
        BestJ = jMax[j]
    print(f"Best iMax: {BestI}, Best jMax: {BestJ}, Option Price at S0: {UHS[-1]:.5f}, Time taken: {actual_time:.4f} seconds")
    plt.figure()
    plt.plot(S, vH, label='High Volatility Value')
    plt.xlabel('Underlying Price S')
    plt.ylabel('Option Value')
    plt.legend()
    plt.grid(True)
    plt.show()


    

        
