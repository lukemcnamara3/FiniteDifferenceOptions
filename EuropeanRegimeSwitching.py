import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_banded

"""""
Under the risk-neutral measure, the spot price follows the
regime-switching mean-reverting stochastic differential equation
   dS = kappa * (theta * exp(mu * t) - S) * dt + sigma(X) * S * dW,
where X ∈ {L,H} denotes the volatility regime.

The mean reversion level is time dependent and given by
   theta * exp(mu * t)

The volatility depends on the current regime:
   sigma(L) = sigma_L
   sigma(H) = sigma_H

The option is priced by solving two coupled PDEs, one for each
volatility regime, using a Crank-Nicolson finite difference
method with fixed-point iteration.

"""
S0 = 4388.6          # Current spot price
T = 10               # Time to maturity
r = 0.25             # Risk-free interest rate

# Mean-reverting process parameters
kappa = 0.14         # Speed of mean reversion
theta = 6150         # Long-run mean level
mu = 0.0379          # Growth rate of the mean-reversion level

# Volatility in each regime
sigma_L = 0.188
sigma_H = 0.455

# Switching intensity parameters
z = 0.605
alpha = 0.253
beta = 0.437

# Growth parameter appearing in the payoff
gamma = 0.016

# Fixed-point iteration parameters
tol = 1e-6
maxiter = 50


def q(S, t, choice):

    """
    State-dependent regime-switching intensity.
    Over a small time interval dt,
        P(L → H) = q_LH(S,t) * dt
        P(H → L) = q_HL(S,t) * dt
    where
        q_LH = alpha + beta * (1 - phi)
        q_HL = alpha * phi  + beta 
    and
        phi = phi((ln(S)-ln(θ)-μt)/z)
    is the standard normal cumulative distribution function.

    The switching intensity therefore depends on both the
    current spot price and time.
    """

    if S == 0:
        Phi = 0.0
    else:
        Phi = norm.cdf((np.log(S) - np.log(theta) - mu * t) / z)

    # Low-volatility regime switching to high volatility
    if choice == 0:

        return alpha + beta * (1 - Phi)

    # High-volatility regime switching to low volatility
    elif choice == 1:

        return alpha * Phi + beta


""""
# Finite Difference Coefficients

The differential operator appearing in the pricing PDE is
     L_X[V]= kappa * (theta * exp(mu * t) - S) dV/dS 
                    + 0.5 * sigma(X)^2 * S^2 d^2V/dS^2
where X ∈ {L,H}.

Crank-Nicolson discretisation of this operator produces a
tridiagonal linear system whose coefficients are
     a_j : lower diagonal
     b_j : main diagonal
     c_j : upper diagonal

The only difference between the two regimes is the volatility
sigma_L or sigma_H.
"""

def aj(j, dS, i, dt, choice):

    """
    Lower diagonal coefficient corresponding to V_{j-1}.
    """

    S = j * dS
    t = i * dt

    if choice == 0:

        return (
            -0.25 * (kappa / dS) * (theta * np.exp(mu * t) - S)
            + 0.25 * sigma_L**2 * j**2
        )

    elif choice == 1:

        return (
            -0.25 * (kappa / dS) * (theta * np.exp(mu * t) - S)
            + 0.25 * sigma_H**2 * j**2
        )


def cj(j, dS, i, dt, choice):

    """
    Upper diagonal coefficient corresponding to V_{j+1}.
    """

    S = j * dS
    t = i * dt

    if choice == 0:

        return (
            0.25 * (kappa / dS) * (theta * np.exp(mu * t) - S)
            + 0.25 * sigma_L**2 * j**2
        )

    elif choice == 1:

        return (
            0.25 * (kappa / dS) * (theta * np.exp(mu * t) - S)
            + 0.25 * sigma_H**2 * j**2
        )


def bj(j, dS, i, dt, choice):

    """
    Main diagonal coefficient.
    This coefficient contains contributions from
      • the time derivative,
      • the diffusion term,
      • discounting at the risk-free rate,
      • the loss due to switching out of the current regime.
    """

    S = j * dS
    t = i * dt

    if choice == 0:

        return (
            -1 / dt
            - 0.5 * sigma_L**2 * j**2
            - 0.5 * r
            - 0.5 * q(S, t, choice)
        )

    elif choice == 1:

        return (
            -1 / dt
            - 0.5 * sigma_H**2 * j**2
            - 0.5 * r
            - 0.5 * q(S, t, choice)
        )
    
def FPCrankNichSwitching(Smax, iMax, jMax, tol, maxiter):

    """
    Prices the European contract by solving the coupled regime-
    switching PDEs using a Crank-Nicolson finite difference method.

    The governing PDEs are

        dV_L/dt + L_L[V_L] - rV_L + q_LH(V_H - V_L) = 0

        dV_H/dt + L_H[V_H] - rV_H + q_HL(V_L - V_H) = 0

    Since the switching terms couple the two PDEs together, the
    solution at each time level is obtained using fixed-point
    iteration.
    """

   
    # Time is discretised into iMax intervals 
    # Stock price is discretised into jMax intervals on [0,Smax].
    dt = T / iMax
    dS = Smax / jMax

    S = np.linspace(0.0, Smax, jMax + 1)
    t = np.linspace(0.0, T, iMax + 1)

    """""
    Storage for the option values in each volatility regime.
    vOld_* contains the solution at the previous time level,
    whilst vNew_* stores the current fixed-point iterate.
    """

    vNew_L = np.zeros(jMax + 1)
    vOld_L = np.zeros(jMax + 1)

    vNew_H = np.zeros(jMax + 1)
    vOld_H = np.zeros(jMax + 1)

    """
    Terminal condition
    The European payoff is
        V(S,T) = theta  if S * exp(gamma * T) > θ
                 0      otherwise.

    Since the payoff is independent of the volatility regime,
    both PDEs have the same terminal condition.
    """

    for j in range(jMax + 1):

        if S[j] * np.exp(gamma * T) > theta:

            vNew_L[j] = theta
            vOld_L[j] = theta

            vNew_H[j] = theta
            vOld_H[j] = theta

    """
    Crank-Nicolson produces a tridiagonal linear system at each time step.
    
    solve_banded() requires the matrix to be stored in banded form, 
    so separate matrices are created for the low and high volatility regimes.
    """

    VH_banded = np.zeros(shape=(3, jMax + 1))
    VL_banded = np.zeros(shape=(3, jMax + 1))

    # Right-hand side vectors for each regime
    dH = np.zeros(jMax + 1)
    dL = np.zeros(jMax + 1)

    # Number of lower and upper diagonals
    l = 1
    u = 1
    l_and_u = (l, u)

    """
    March backwards through time from maturity to the present, as
    required for backward parabolic PDEs arising in option pricing.
    
    """

    for i in range(iMax - 1, -1, -1):

        """
        Boundary conditions
        
        At S = 0 the PDE becomes degenerate since the diffusion
        term vanishes. The finite difference stencil therefore
        incorporates the degenerate boundary conditions
        obtained directly from the PDE.
        
        At S = Smax the asymptotic behaviour is
            V_L = V_H = B(t),
        where
             B(t) = theta * exp(-r * (T-t)).
        
        This corresponds to discounting the limiting payoff
        back to the current time.
        """

        # High-volatility regime
        VH_banded[1][0] = bj(0, dS, i, dt, 1)
        VH_banded[0][1] = 2 * cj(0, dS, i, dt, 1)

        VH_banded[2][jMax - 1] = 0.0
        VH_banded[1][jMax] = 1.0

        # Low-volatility regime
        VL_banded[1][0] = bj(0, dS, i, dt, 0)
        VL_banded[0][1] = 2 * cj(0, dS, i, dt, 0)

        VL_banded[2][jMax - 1] = 0.0
        VL_banded[1][jMax] = 1.0

        """
        Assemble the interior tridiagonal coefficients for both PDEs.
        
        The only difference between the two matrices is the
        volatility parameter appearing in the diffusion term.
        """
        for j in range(1, jMax):

            # High-volatility coefficients
            VH_banded[2][j - 1] = aj(j, dS, i, dt, 1)
            VH_banded[1][j] = bj(j, dS, i, dt, 1)
            VH_banded[0][j + 1] = cj(j, dS, i, dt, 1)

            # Low-volatility coefficients
            VL_banded[2][j - 1] = aj(j, dS, i, dt, 0)
            VL_banded[1][j] = bj(j, dS, i, dt, 0)
            VL_banded[0][j + 1] = cj(j, dS, i, dt, 0)

        """
        Initialise the fixed-point iteration using the solution
        from the previous time level.
        
        This generally provides an excellent initial guess
        because the solution changes smoothly in time.
        """

        vNew_L[:] = vOld_L[:]
        vNew_H[:] = vOld_H[:]

        """
        Fixed-point iteration
        
        The regime-switching terms:
             q_LH(V_H - V_L)
             q_HL(V_L - V_H)
        couple the two PDEs together, so they cannot be solved
        independently. Instead, at each time level:
        
           1. Construct the right-hand side using the current
              estimates of the opposite regime.
           2. Solve the two tridiagonal systems.
           3. Repeat until successive iterates differ by less
              than the prescribed tolerance.
        """

        for iter in range(maxiter):

            """"
            Boundary values of the Crank-Nicolson right-hand side.
            
            The left boundary incorporates the degenerate PDE
            condition at S = 0, while the right boundary uses
            the asymptotic condition:
                 V = θ exp(-r(T-t)).
            """
            dH[0] = (
                -(bj(0, dS, i, dt, 1) + 2 / dt) * vOld_H[0]
                - 2 * cj(0, dS, i, dt, 1) * vOld_H[1]
                - 0.5 * q(0, t[i], 1) * (vOld_L[0] + vNew_L[0])
            )

            dL[0] = (
                -(bj(0, dS, i, dt, 0) + 2 / dt) * vOld_L[0]
                - 2 * cj(0, dS, i, dt, 0) * vOld_L[1]
                - 0.5 * q(0, t[i], 0) * (vOld_H[0] + vNew_H[0])
            )

            dH[jMax] = theta * np.exp(-r * (T - t[i]))
            dL[jMax] = theta * np.exp(-r * (T - t[i]))

            """
            Assemble the interior right-hand side.
            The Crank-Nicolson method averages the spatial
            operator between time levels, giving second-order
            accuracy in time.
            
            The final term in each equation represents the
            coupling between the two volatility regimes.
            """

            for j in range(1, jMax):

                dH[j] = (
                    -aj(j, dS, i, dt, 1) * vOld_H[j - 1]
                    + (-bj(j, dS, i, dt, 1) - 2 / dt) * vOld_H[j]
                    - cj(j, dS, i, dt, 1) * vOld_H[j + 1]
                    - 0.5 * q(S[j], t[i], 1)
                    * (vOld_L[j] + vNew_L[j])
                )

                dL[j] = (
                    -aj(j, dS, i, dt, 0) * vOld_L[j - 1]
                    + (-bj(j, dS, i, dt, 0) - 2 / dt) * vOld_L[j]
                    - cj(j, dS, i, dt, 0) * vOld_L[j + 1]
                    - 0.5 * q(S[j], t[i], 0)
                    * (vOld_H[j] + vNew_H[j])
                )

            
            # Solve the two tridiagonal linear systems.
            vL_temp = solve_banded(l_and_u, VL_banded, dL)
            vH_temp = solve_banded(l_and_u, VH_banded, dH)

        
            # Measure convergence of the fixed-point iteration
            Lerror = np.max(np.abs(vL_temp - vNew_L))
            Herror = np.max(np.abs(vH_temp - vNew_H))
            error = max(Lerror, Herror)

            # Update the current iterate
            vNew_L[:] = vL_temp[:]
            vNew_H[:] = vH_temp[:]

            # Stop once the desired tolerance is achieved
            if error < tol:
                break

            # Report if the fixed-point iteration fails to
            # converge within the prescribed number of iterations
            if iter == maxiter - 1:
                print(
                    f"Warning: Maximum iterations reached at "
                    f"time step {i} with error {error}"
                )

        
        # Accept the converged solution and move to the previous time level.
        vOld_H = np.copy(vNew_H)
        vOld_L = np.copy(vNew_L)

    """
    After marching backwards through every time level,
    vNew_L and vNew_H contain the approximations to the
    option value at the initial time t = 0.
    """
    return S, vNew_L, vNew_H


# Plotting and Convergence Analysis

def CNGraph(Smax, iMax, jMax, tol, maxiter):

    """
    Solves the coupled PDE system and plots the option value in
    both volatility regimes as a function of the spot price.
    """

    S, vL, vH = FPCrankNichSwitching(Smax, iMax, jMax, tol, maxiter)

    # Interpolate the numerical solution to obtain the option
    # value at the observed spot price S0.
    VH_S0 = np.interp(S0, S, vH)

    print(f"Option Value at S0 for High Volatility: {VH_S0}")

    plt.plot(S, vL, label="Low Volatility")
    plt.plot(S, vH, label="High Volatility")

    plt.xlabel("S")
    plt.ylabel("Option Value")
    plt.grid(True)
    plt.legend()
    plt.show()


def jMaxData(Smax, choice, tol, maxiter):

    """
    Investigates spatial convergence by computing the option
    price for increasing numbers of spatial grid points (jMax)
    and several different time discretisations.
    """

    jMax = [250, 500, 750, 1000, 1250, 1500, 2000]
    iMax = [50, 100, 250]
    iMax_labels = ["50", "100", "250"]

    values = []

    for i in range(len(iMax)):

        VHS = []

        for k in jMax:

            k = int(k)

            S, VL, VH = FPCrankNichSwitching(
                Smax,
                iMax[i],
                k,
                tol,
                maxiter,
            )

            # Interpolate to the current market spot price.
            VH_S0 = np.interp(S0, S, VH)
            VHS.append(VH_S0)

        values.append(VHS)

        print(f"iMax: {iMax[i]} has been completed")

    if choice == 0:

        plt.figure()

        for i in range(len(iMax)):

            plt.plot(
                jMax,
                values[i],
                label=f"iMax: {iMax_labels[i]}",
            )

        plt.xlabel("jMax")
        plt.ylabel("Option Price")
        plt.legend()
        plt.grid(True)
        plt.show()


def jMaxTable(Smax, iMax, tol, maxiter, choice):

    """
    Produces a convergence table showing how the option price
    changes as the spatial mesh is refined.

    The final column gives the absolute difference between
    successive approximations, providing an estimate of
    convergence.
    """

    jMax = np.arange(100, 3000 + 100, 100)

    VHS = []

    for k in jMax:

        k = int(k)

        S, VL, VH = FPCrankNichSwitching(
            Smax,
            iMax,
            k,
            tol,
            maxiter,
        )

        print(f"jMax: {k} has been computed")

        VH_S0 = np.interp(S0, S, VH)
        VHS.append(VH_S0)

    with open("testing_jMax.tex", "w") as f:

        f.write("\\begin{tabular}{|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("jMax & Option Price at S0 & |V_{k} - V_{k-1}| \\\\\n")
        f.write("\\hline\n")

        f.write(f"{jMax[0]} & {VHS[0]} & 0 \\\\\n")

        for j in range(1, len(jMax)):

            line = (
                f"{jMax[j]} & {VHS[j]} & "
                f"{abs(VHS[j] - VHS[j-1])} \\\\\n"
            )

            f.write(line)

        f.write("\\end{tabular}\n")

    plt.figure()

    plt.plot(jMax, VHS)

    plt.xlabel("jMax")
    plt.ylabel("Option Price")
    plt.grid(True)

    plt.show()


def iMaxTable(Smax, jMax, tol, maxiter):

    """
    Produces a convergence table for the temporal
    discretisation by increasing the number of time steps.

    The difference between successive approximations is used
    to assess temporal convergence.
    """

    iMax = [
        10,
        25,
        50,
        100,
        250,
        500,
        750,
        1000,
        1500,
        2500,
        5000,
        7500,
        10000,
    ]

    VHS = []

    for i in iMax:

        i = int(i)

        S, VL, VH = FPCrankNichSwitching(
            Smax,
            i,
            jMax,
            tol,
            maxiter,
        )

        print(f"iMax: {i} has been computed")

        VH_S0 = np.interp(S0, S, VH)

        VHS.append(VH_S0)

    with open("testing_iMax.tex", "w") as f:

        f.write("\\begin{tabular}{|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("iMax & Option Price at S0 & |V_{k} - V_{k-1}| \\\\\n")
        f.write("\\hline\n")

        f.write(f"{iMax[0]} & {VHS[0]} & 0 \\\\\n")

        for j in range(1, len(iMax)):

            line = (
                f"{iMax[j]} & {VHS[j]} & "
                f"{abs(VHS[j] - VHS[j-1])} \\\\\n"
            )

            f.write(line)

        f.write("\\end{tabular}\n")


def SmaxGraph(iMax, jMax, tol, maxiter):

    """
    Investigates the sensitivity of the numerical solution to
    the choice of the truncated computational domain.

    As Smax increases, the finite difference solution should
    stabilise, indicating that the artificial upper boundary
    is sufficiently far from the region of interest.
    """

    Smax = np.arange(5000, 14000 + 100, 100)

    VHS = []

    for S in Smax:

        S = int(S)

        stock, VL, VH = FPCrankNichSwitching(
            S,
            iMax,
            jMax,
            tol,
            maxiter,
        )

        print(f"Smax: {S} has been computed")

        VHS0 = np.interp(S0, stock, VH)

        VHS.append(VHS0)

    plt.figure()

    plt.plot(Smax, VHS)

    plt.xlabel("Smax")
    plt.ylabel("Option Price")
    plt.grid(True)

    plt.show()

