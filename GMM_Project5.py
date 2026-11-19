### PART (b): JUST-IDENTIFIED TWO-STEP GMM ###

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load the dataset
df = pd.read_csv("PS5v3.csv")

# Extract variables
y = df["y"].to_numpy()
x = df["x"].to_numpy()
z1 = df["z1"].to_numpy()

# Constant E(v) = exp(1/2)
Ev = np.exp(0.5)

# -------------------------------------------------
# Moment conditions
# -------------------------------------------------

def moments(theta):
    """
    theta[0] = beta0
    theta[1] = beta1

    Returns the individual moment conditions:
    g1_i = y_i * exp(-beta0 - beta1*x_i) - exp(0.5)
    g2_i = g1_i * z1_i
    """

    beta0, beta1 = theta

    residual = y * np.exp(-beta0 - beta1 * x) - Ev

    g1 = residual
    g2 = residual * z1

    return np.column_stack((g1, g2))


# -------------------------------------------------
# GMM objective function
# -------------------------------------------------

def gmm_objective(theta, W):
    g = moments(theta)

    # Sample average of the moment conditions
    g_bar = g.mean(axis=0)

    # GMM criterion
    return g_bar.T @ W @ g_bar


# -------------------------------------------------
# STEP 1: Initial GMM using identity weighting matrix
# -------------------------------------------------

W1 = np.eye(2)

initial_values = np.array([0.1, 1.0])

step1 = minimize(
    gmm_objective,
    initial_values,
    args=(W1,),
    method="BFGS"
)

theta_step1 = step1.x

print("\n========================================")
print("PART (b): FIRST-STEP GMM")
print("========================================")
print(f"beta0 = {theta_step1[0]:.6f}")
print(f"beta1 = {theta_step1[1]:.6f}")


# -------------------------------------------------
# STEP 2: Construct optimal weighting matrix
# -------------------------------------------------

g_step1 = moments(theta_step1)

# Estimate covariance matrix of the moment conditions
S = (g_step1.T @ g_step1) / len(df)

# Optimal GMM weighting matrix
W2 = np.linalg.inv(S)


# -------------------------------------------------
# STEP 2: Efficient two-step GMM
# -------------------------------------------------

step2 = minimize(
    gmm_objective,
    theta_step1,
    args=(W2,),
    method="BFGS"
)

theta_hat = step2.x

beta0_hat = theta_hat[0]
beta1_hat = theta_hat[1]

print("\n========================================")
print("PART (b): TWO-STEP GMM ESTIMATES")
print("========================================")
print(f"beta0 = {beta0_hat:.6f}")
print(f"beta1 = {beta1_hat:.6f}")


# -------------------------------------------------
# Check sample moments
# -------------------------------------------------

g_final = moments(theta_hat)
g_bar_final = g_final.mean(axis=0)

print("\nSample moments at estimated parameters:")
print(f"Moment 1 = {g_bar_final[0]:.10f}")
print(f"Moment 2 = {g_bar_final[1]:.10f}")


# -------------------------------------------------
# J statistic
# -------------------------------------------------

n = len(df)

J_stat = n * (g_bar_final.T @ W2 @ g_bar_final)

print("\n========================================")
print("J-TEST")
print("========================================")
print(f"J statistic = {J_stat:.10f}")
print("Degrees of freedom = 0")
print(
    "Because the model is just identified, "
    "there are no overidentifying restrictions to test."
)
# -------------------------------------------------
# Standard errors for just-identified GMM
# -------------------------------------------------

def numerical_jacobian_just(theta, h=1e-6):

    k = len(theta)
    q = moments(theta).shape[1]

    D = np.zeros((q, k))

    for j in range(k):

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[j] += h
        theta_minus[j] -= h

        g_plus = moments(theta_plus).mean(axis=0)
        g_minus = moments(theta_minus).mean(axis=0)

        D[:, j] = (g_plus - g_minus) / (2 * h)

    return D


D_just = numerical_jacobian_just(theta_hat)

V_just = np.linalg.inv(D_just.T @ W2 @ D_just) / n

se_just = np.sqrt(np.diag(V_just))

print("\n========================================")
print("PART (b): GMM STANDARD ERRORS")
print("========================================")

print(f"SE(beta0) = {se_just[0]:.6f}")
print(f"SE(beta1) = {se_just[1]:.6f}")
###Part d) ###
### PART (d): OVERIDENTIFIED TWO-STEP GMM ###

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

# Load dataset
df = pd.read_csv("PS5v3.csv")

# Extract variables
y = df["y"].to_numpy()
x = df["x"].to_numpy()
z1 = df["z1"].to_numpy()
z2 = df["z2"].to_numpy()

# Number of observations
n = len(df)

# E(v) = exp(1/2)
Ev = np.exp(0.5)


# -------------------------------------------------
# Three moment conditions
# -------------------------------------------------

def moments_overid(theta):

    beta0, beta1 = theta

    # Structural residual implied by the model
    residual = y * np.exp(-beta0 - beta1 * x) - Ev

    # Moment 1
    g1 = residual

    # Moment 2
    g2 = residual * z1

    # Moment 3
    g3 = residual * z1 * z2

    return np.column_stack((g1, g2, g3))


# -------------------------------------------------
# GMM objective function
# -------------------------------------------------

def gmm_objective_overid(theta, W):

    g = moments_overid(theta)

    # Sample mean of moment conditions
    g_bar = g.mean(axis=0)

    # GMM criterion
    return g_bar.T @ W @ g_bar


# -------------------------------------------------
# STEP 1: GMM using identity weighting matrix
# -------------------------------------------------

W1 = np.eye(3)

initial_values = np.array([0.1, 1.0])

step1_overid = minimize(
    gmm_objective_overid,
    initial_values,
    args=(W1,),
    method="BFGS"
)

theta_step1_overid = step1_overid.x

print("\n========================================")
print("PART (d): FIRST-STEP OVERIDENTIFIED GMM")
print("========================================")

print(f"beta0 = {theta_step1_overid[0]:.6f}")
print(f"beta1 = {theta_step1_overid[1]:.6f}")


# -------------------------------------------------
# Construct second-step weighting matrix
# -------------------------------------------------

g_step1 = moments_overid(theta_step1_overid)

# Estimated covariance matrix of moment conditions
S = (g_step1.T @ g_step1) / n

# Optimal weighting matrix
W2 = np.linalg.inv(S)


# -------------------------------------------------
# STEP 2: Efficient two-step GMM
# -------------------------------------------------

step2_overid = minimize(
    gmm_objective_overid,
    theta_step1_overid,
    args=(W2,),
    method="BFGS"
)

theta_hat_overid = step2_overid.x

beta0_overid = theta_hat_overid[0]
beta1_overid = theta_hat_overid[1]

print("\n========================================")
print("PART (d): TWO-STEP OVERIDENTIFIED GMM")
print("========================================")

print(f"beta0 = {beta0_overid:.6f}")
print(f"beta1 = {beta1_overid:.6f}")


# -------------------------------------------------
# Sample moments at final estimates
# -------------------------------------------------

g_final = moments_overid(theta_hat_overid)

g_bar_final = g_final.mean(axis=0)

print("\nSample moments at estimated parameters:")

print(f"Moment 1 = {g_bar_final[0]:.10f}")
print(f"Moment 2 = {g_bar_final[1]:.10f}")
print(f"Moment 3 = {g_bar_final[2]:.10f}")


# -------------------------------------------------
# Numerical Jacobian for GMM standard errors
# -------------------------------------------------

def numerical_jacobian(theta, h=1e-6):

    k = len(theta)
    q = moments_overid(theta).shape[1]

    D = np.zeros((q, k))

    for j in range(k):

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[j] += h
        theta_minus[j] -= h

        g_plus = moments_overid(theta_plus).mean(axis=0)
        g_minus = moments_overid(theta_minus).mean(axis=0)

        D[:, j] = (g_plus - g_minus) / (2 * h)

    return D


D = numerical_jacobian(theta_hat_overid)

# Efficient two-step GMM covariance matrix
V = np.linalg.inv(D.T @ W2 @ D) / n

standard_errors = np.sqrt(np.diag(V))

print("\n========================================")
print("GMM STANDARD ERRORS")
print("========================================")

print(f"SE(beta0) = {standard_errors[0]:.6f}")
print(f"SE(beta1) = {standard_errors[1]:.6f}")


# -------------------------------------------------
# J-test of overidentifying restrictions
# -------------------------------------------------

J_stat = n * (
    g_bar_final.T @ W2 @ g_bar_final
)

# Number of moments - number of parameters
J_df = 3 - 2

J_pvalue = chi2.sf(J_stat, J_df)

print("\n========================================")
print("J-TEST OF OVERIDENTIFYING RESTRICTIONS")
print("========================================")

print(f"J statistic = {J_stat:.6f}")
print(f"Degrees of freedom = {J_df}")
print(f"P-value = {J_pvalue:.6f}")

if J_pvalue < 0.05:
    print(
        "Reject the null hypothesis: "
        "the overidentifying restrictions are not supported."
    )
else:
    print(
        "Fail to reject the null hypothesis: "
        "the overidentifying restrictions are supported by the data."
    )