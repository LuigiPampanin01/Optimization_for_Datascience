import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys
import tqdm

def grad_x(X, S):
    X_inv = np.linalg.inv(X)
    return S - X_inv

def h_func_cp(X, gamma):
    offdiag_mask = ~np.eye(X.shape[0], dtype=bool)
    return gamma * cp.norm1(cp.multiply(offdiag_mask, X))

def h_func(X, gamma):
    offdiag_mask = ~np.eye(X.shape[0], dtype=bool)
    return gamma * np.sum(np.abs(np.multiply(offdiag_mask, X)))

def prox_h(X, grad, h_func_cp, t, gamma):
    """
    Compute prox_{h}(x) = argmin_y h(y) + 0.5 * ||y - (X - t*grad)||_2^2
    """
    x = X - t * grad

    y = cp.Variable(x.shape)
    objective = h_func_cp(y, gamma) + 0.5 * cp.sum_squares(y - x)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=cp.SCS, verbose=False)  # You can pick ECOS too
    return y.value

def g_func(X, S):
    sign, logdet = np.linalg.slogdet(X)
    if sign <= 0:
        raise ValueError("Matrix not positive definite")
    return np.trace(S @ X) - logdet

def compute_stopping_criterion(X, S, g_func, h_func, gamma, compute_U):
    n = X.shape[0]
    sign, logdet = np.linalg.slogdet(S + compute_U(X, S, gamma))
    if sign <= 0:
        raise ValueError("Matrix not positive definite in stopping criterion")
    delta = g_func(X, S) + h_func(X, gamma) - logdet - n
    return delta

def compute_U(X, S, gamma):
    """
    Compute U where:
    U_ij = max(-gamma, min(gamma, [X_inv - S]_ij)) for i ≠ j
           0 for i == j
    """
    X_inv = np.linalg.inv(X)
    diff = X_inv - S

    U = np.clip(diff, -gamma, gamma)
    np.fill_diagonal(U, 0)
    return U

def proximal_gradient_descend(X, h_func_cp, h_func, t, S, gamma, g_func, compute_U, epsilon=1e-2):
    while True:
        grad = grad_x(X, S)
        X_new = prox_h(X, grad, h_func_cp, t, gamma)
        delta = compute_stopping_criterion(X_new, S, g_func, h_func, gamma, compute_U)
        X = X_new
        if delta <= epsilon:
            break
    return X

def proximal_gradient_descent_backtracking(
    X, h_func_cp, h_func, t_init, S, gamma,
    g_func, compute_U, epsilon=1e-2, beta=0.5, max_iter=1000
):
    """
    Proximal gradient descent with backtracking line search for Graphical Lasso.
    """

    def g_only(X_val, S):
        return g_func(X_val, S)

    for it in tqdm.tqdm(range(max_iter)):
        t = t_init
        grad = grad_x(X, S)

        while True:
            X_new = prox_h(X, grad, h_func_cp, t, gamma)

            # Check if X_new is positive definite
            try:
                np.linalg.cholesky(X_new)
                is_pos_def = True
            except np.linalg.LinAlgError:
                is_pos_def = False

            lhs = g_only(X_new, S)
            diff = X_new - X
            rhs = g_only(X, S) + np.sum(grad * diff) + (1 / (2 * t)) * np.linalg.norm(diff, "fro")**2

            if is_pos_def and lhs <= rhs:
                break

            t *= beta  # shrink step size

        delta = compute_stopping_criterion(X_new, S, g_func, h_func, gamma, compute_U)
        X = X_new

        if delta <= epsilon:
            break

    return X

if __name__ == "__main__":
    subset_size = int(sys.argv[1]) if len(sys.argv) > 1 else 492
    print(f"Subset size: {subset_size}")

    # Parameters
    t_init = 1  # Initial step size
    beta = 0.5  # Backtracking parameter
    epsilon = 1e-2  # Convergence threshold
    gamma = 0.1  # Regularization parameter

    # Load data
    matrix = np.loadtxt('data/sp500.txt')
    matrix = matrix[:subset_size, :subset_size]  # Use a subset for faster testing
    n = matrix.shape[0]

    # Initialization
    X_init = np.eye(n)

    # Run Proximal Gradient Descent with Backtracking
    result_pgb = proximal_gradient_descent_backtracking(
        X_init, h_func_cp, h_func, t_init, matrix, gamma,
        g_func, compute_U, epsilon, beta
    )

    print("Result of Proximal Gradient Descent with Backtracking:")
    print(result_pgb)

    # Plot the result_pgb with diagonal set to -inf
    result_pgb_plot = result_pgb.copy()
    np.fill_diagonal(result_pgb_plot, -np.inf)

    plt.imshow(result_pgb_plot, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title("Heatmap of result_pgb (diagonal set to -inf)")
    plt.show()
