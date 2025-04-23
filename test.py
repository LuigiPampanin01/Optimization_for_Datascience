import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys

def grad_x(X, S):

    X_inv = np.linalg.inv(X)
    return S - X_inv

def arg_prox(X, t, grad_x, S):

    return X - t * grad_x(X, S)

def h_func_cp(X, gamma):
    offdiag_mask = ~np.eye(X.shape[0], dtype=bool)
    return gamma * cp.norm1(cp.multiply(offdiag_mask, X))

def h_func(X, gamma):
    offdiag_mask = ~np.eye(X.shape[0], dtype=bool)
    return gamma * np.sum(np.abs(np.multiply(offdiag_mask, X)))

def prox_h(X, h_func_cp, t, grad_x, arg_prox, S, gamma):
    """
    Compute prox_{h}(x) = argmin_y h(y) + 0.5 * ||y - x||_2^2

    Parameters:
    - x (np.ndarray): The point at which to evaluate the proximal operator
    - h_func_cp (callable): A function that accepts a cvxpy Variable y and returns h(y)

    Returns:
    - np.ndarray: Result of the proximal operator
    """
    x = arg_prox(X, t, grad_x, S)

    y = cp.Variable(x.shape)
    objective = h_func_cp(y, gamma) + 0.5 * cp.sum_squares(y - x)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve()
    return y.value

def g_func(X, S):
    return np.trace(S @ X) - np.log(np.linalg.det(X))

def compute_stopping_criterion(X, S, g_func, h_func, gamma, compute_U):

    n = X.shape[0]

    delta = g_func(X, S) + h_func(X, gamma) - np.log(np.linalg.det(S + compute_U(X, S, gamma))) - n
    
    return delta

def proximal_gradient_descend(X, h_func_cp, h_func, t, grad_x, arg_prox, S, gamma, g_func, compute_U, epsilon=1e-2): # TODO : i'm not fucking sure this work, can you guys check please 

    while True:
        X_new = prox_h(X, h_func_cp, t, grad_x, arg_prox, S, gamma)
        delta = compute_stopping_criterion(X_new, S, g_func, h_func, gamma, compute_U)

        X = X_new

        if delta <= epsilon:
            break 
    
    return X

def compute_U(X, S, gamma):
    """
    Compute U where:
    U_ij = max(-gamma, min(gamma, [X_inv - S]_ij)) for i ≠ j
           0 for i == j
    
    Parameters:
    - X (np.ndarray): Square positive definite matrix
    - S (np.ndarray): Symmetric matrix of same shape as X
    - gamma (float): Threshold parameter
    
    Returns:
    - np.ndarray: Matrix U
    """
    X_inv = np.linalg.inv(X)
    diff = X_inv - S

    # Apply soft thresholding only to off-diagonal elements
    U = np.zeros_like(diff)
    # Copy and clip everything
    U[:] = np.clip(diff, -gamma, gamma)

    # Set diagonal to 0
    np.fill_diagonal(U, 0)
    # for i in range(diff.shape[0]):
    #     for j in range(diff.shape[1]):
    #         if i != j:
    #             U[i, j] = np.clip(diff[i, j], -gamma, gamma)
    #         else:
    #             U[i, j] = 0  # optional, since we initialized with zeros

    return U

def backtracking_line_search(phi, phi_derivative_at_0, t_init, alpha1=0.1, beta=0.7):
    """
    Perform backtracking line search to find step size t.

    Parameters:
        phi (function): A continuously differentiable function φ: R → R.
        phi_derivative_at_0 (float): The derivative φ'(0).
        t_init (float): Initial step size (t >= 0).
        alpha1 (float): Parameter in (0, 0.5], default 0.1.
        beta (float): Parameter in (0, 1), default 0.7.

    Returns:
        float: Step size t such that φ(t) ≤ φ(0) + α1 * t * φ'(0)
    """
    t = t_init
    phi_0 = phi(0)

    while phi(t) > phi_0 + alpha1 * t * phi_derivative_at_0:
        t *= beta

    return t

def proximal_gradient_descent_backtracking(
    X, h_func_cp, h_func, t_init, grad_x, arg_prox, S, gamma,
    g_func, compute_U, epsilon=1e-2, alpha=0.1, beta=0.5, max_iter=1
):
    """
    Proximal gradient descent with backtracking line search for Graphical Lasso.
    """

    def objective(X_val, S, gamma):
        return g_func(X_val, S) + h_func(X_val, gamma)

    while True:
        t = t_init
        for _ in range(max_iter):
            X_new = prox_h(X, h_func_cp, t, grad_x, arg_prox, S, gamma)

            # Armijo condition
            lhs = objective(X_new, S, gamma)

            rhs = objective(X, S, gamma) + alpha * np.sum(grad_x(X, S) * (X_new - X))  # <∇g(X), X_new - X>
            if lhs <= rhs:
                break
            t *= beta  # backtrack if Armijo not satisfied

        delta = compute_stopping_criterion(X_new, S, g_func, h_func, gamma, compute_U)
        X = X_new

        if delta <= epsilon:
            break

    return X


if __name__ == "__main__":

    subset_size = int(sys.argv[1]) if len(sys.argv) > 1 else 492
    print(f"Subset size: {subset_size}")
    # Test proximal_gradient_descent_backtracking
    t_init = 0.1  # Initial step size
    alpha = 0.1  # Armijo condition parameter
    beta = 0.5  # Backtracking parameter
    epsilon = 1e-2  # Convergence threshold
    gamma = 0.1  # Regularization parameter

    # Load data
    matrix = np.loadtxt('data/sp500.txt')
    matrix = matrix[:subset_size, :subset_size]  # Use a smaller subset for testing
    n = matrix.shape[0]

    # Example usage
    X_init = prox_h(np.eye(n), h_func_cp, 0.1, grad_x, arg_prox, matrix, 0.1)

    result_pgb = proximal_gradient_descent_backtracking(
        X_init, h_func_cp, h_func, t_init, grad_x, arg_prox, matrix, gamma,
        g_func, compute_U, epsilon, alpha, beta
    )

    print("Result of Proximal Gradient Descent with Backtracking:")
    print(result_pgb)
