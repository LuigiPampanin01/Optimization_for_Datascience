import numpy as np
from numpy.linalg import slogdet, inv, norm, LinAlgError
import matplotlib.pyplot as plt


def project_dual(U, gamma):
    """Project matrix U onto the feasible set of the dual problem"""
    U_proj = U.copy()
    n = U.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                U_proj[i, j] = 0
            else:
                U_proj[i, j] = np.clip(U[i, j], -gamma, gamma)
    return U_proj

def dual_objective(U, S):
    """Dual objective: logdet(S + U) - n"""
    A = S + U
    sign, logdet = slogdet(A)
    if sign != 1:
        return -np.inf
    return logdet - S.shape[0]

def dual_gradient(U, S):
    """Gradient of dual objective: grad = (S + U)^(-1)"""
    try:
        return inv(S + U)
    except LinAlgError:
        return np.zeros_like(S)

def graphical_lasso_dual(S, gamma, max_iter=1000, epsilon=1e-3, alpha=0.25, beta=0.5, t0=1.0):
    """
    Solves the dual problem of graphical LASSO using projected gradient ascent with backtracking
    """
    n = S.shape[0]
    U = np.zeros((n, n))  # Initial guess (symmetric zero matrix)

    for k in range(max_iter):
        grad = dual_gradient(U, S)

        # Backtracking line search
        t = t0
        while True:
            U_new = project_dual(U + t * grad, gamma)
            lhs = dual_objective(U_new, S)
            rhs = dual_objective(U, S) + alpha * t * np.sum(grad * (U_new - U))
            if lhs >= rhs:
                break
            t *= beta

        if norm(U_new - U, ord='fro') < epsilon:
            break

        U = U_new

    X_opt = inv(S + U)  # Recover primal precision matrix
    return X_opt, U


np.random.seed(0)
n = 492
A = np.random.randn(n, n)
S = A @ A.T  # sample covariance matrix (PD)
gamma = 0.1

X_opt, U_opt = graphical_lasso_dual(S, gamma)
print("Primal X (precision matrix):\n", X_opt)
print("Dual U:\n", U_opt)

# Plot the result_pgb with diagonal set to -inf
result_pgb_plot = X_opt
np.fill_diagonal(result_pgb_plot, -np.inf)

plt.imshow(result_pgb_plot, cmap='viridis', interpolation='none')
plt.colorbar()
plt.title("Heatmap of result_pgb (diagonal set to -inf)")
plt.show()