import numpy as np
import matplotlib.pyplot as plt
import sys
import tqdm

# --- Functions ---

def grad_x(X, S):
    X_inv = np.linalg.inv(X)
    return S - X_inv

def h_func(X, gamma):
    offdiag_mask = ~np.eye(X.shape[0], dtype=bool)
    return gamma * np.sum(np.abs(np.multiply(offdiag_mask, X)))

def prox_h(X, grad, t, gamma):
    """
    Fast proximal operator for off-diagonal soft-thresholding.
    """
    Z = X - t * grad
    offdiag_mask = ~np.eye(Z.shape[0], dtype=bool)

    prox = Z.copy()
    prox[offdiag_mask] = np.sign(Z[offdiag_mask]) * np.maximum(np.abs(Z[offdiag_mask]) - t * gamma, 0)
    
    return prox

def g_func(X, S):
    sign, logdet = np.linalg.slogdet(X)
    if sign <= 0:
        raise ValueError("Matrix not positive definite")
    return np.trace(S @ X) - logdet

def compute_stopping_criterion(X, S, g_func, h_func, gamma, compute_U):
    n = X.shape[0]
    U = compute_U(X, S, gamma)
    sign, logdet = np.linalg.slogdet(S + U)
    if sign <= 0:
        return np.inf  # If not positive definite, we cannot compute a valid duality gap
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

    U = np.clip(diff.copy(), -gamma, gamma)
    np.fill_diagonal(U, 0)
    return U

def proximal_gradient_descent_backtracking(
    X, h_func, t_init, S, gamma,
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
            X_new = prox_h(X, grad, t, gamma)

            # Check if X_new is positive definite
            try:
                np.linalg.cholesky(X_new)
                is_pos_def = True
            except np.linalg.LinAlgError:
                is_pos_def = False

            diff = X_new - X
            rhs = g_only(X, S) + np.sum(grad * diff) + (1 / (2 * t)) * np.linalg.norm(diff, "fro")**2

            if is_pos_def:
                lhs = g_only(X_new, S)
                if lhs <= rhs:
                    break

            t *= beta  # shrink step size


        delta = compute_stopping_criterion(X_new, S, g_func, h_func, gamma, compute_U)
        X = X_new

        if delta <= epsilon:
            break

    return X

# --- Main ---

if __name__=='__main__':
    # np.random.seed(42)
    subset_size = int(sys.argv[1]) if len(sys.argv) > 1 else 492
    matrix = np.loadtxt('data/sp500.txt')
    matrix = matrix[:subset_size, :subset_size]

    n = matrix.shape[0]
    X_init = np.eye(n)

    x_points = []
    y_points = []
    n_no_zeros = []

    epsilon = 1e-2
    interval_zero = [-epsilon, epsilon]
    gammas = np.logspace(-2, -1, 10)

    for gamma in gammas:
        X_opt = proximal_gradient_descent_backtracking(
            X_init, h_func, 1.0, matrix, gamma,
            g_func, compute_U, epsilon, beta=0.05
        )
        print("Primal X (precision matrix):\n", X_opt)

        # Plot
        result_pgb_plot = X_opt.copy()
        np.fill_diagonal(result_pgb_plot, -np.inf)

        plt.imshow(result_pgb_plot, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title(f"Heatmap (gamma={gamma:.2e})")
        plt.savefig(f'outputs/heatmap_gamma_{gamma:.2e}.png')
        plt.close()

        # Save trade-off points
        sign, logdet = np.linalg.slogdet(X_opt)
        x_points.append(np.trace(matrix @ X_opt) - logdet)

        off_diag_mask = ~np.eye(X_opt.shape[0], dtype=bool)
        off_diag_l1 = np.sum(np.abs(X_opt[off_diag_mask]))
        y_points.append(off_diag_l1)

        n_no_zeros.append(np.sum((X_opt[off_diag_mask] < interval_zero[0]) | (X_opt[off_diag_mask] > interval_zero[1])))
    

    print(x_points, y_points)
    plt.figure(figsize=(8, 6))
    plt.plot(x_points, y_points, marker='o', label='Trade-off Curve')
    plt.xlabel('Trace - LogDet')
    plt.ylabel('L1 Norm of Off-Diagonal Elements')
    plt.title('Trade-off Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/l1_norm_off_diag.png')

    plt.figure(figsize=(8, 6))
    plt.plot(gammas, n_no_zeros, marker='o', label='Number of Non Zeros')
    plt.xlabel('Gamma')
    plt.ylabel('Number of Non Zeros in X')
    plt.title('Number of Non Zeros vs Gamma')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/non_zeros_X.png')

