import numpy as np
from matplotlib import pyplot as plt
import sys
from tqdm import tqdm

def graphical_lasso_proximal_gradient(S, gamma, epsilon=1e-2, max_iter=1000, verbose=False):
    n = S.shape[0]

    # Helper functions
    def g(X):
        return np.trace(S @ X) - np.linalg.slogdet(X)[1]

    def grad_g(X):
        return S - np.linalg.inv(X)

    def prox_h(X, t):
        X_new = np.copy(X)
        off_diag_mask = ~np.eye(n, dtype=bool)
        X_new[off_diag_mask] = np.sign(X[off_diag_mask]) * np.maximum(np.abs(X[off_diag_mask]) - t * gamma, 0)
        return X_new

    def duality_gap(X, S, gamma):
        W = np.linalg.inv(X) - S
        U = np.clip(W, -gamma, gamma)
        np.fill_diagonal(U, 0)
        if np.all(np.linalg.eigvals(S + U) > 0):
            gap = g(X) + gamma * np.sum(np.abs(X[~np.eye(n, dtype=bool)])) \
                  - np.linalg.slogdet(S + U)[1] - n
            return gap
        else:
            return np.inf

    # Initialization
    X = np.eye(n)
    beta = 0.5

    for iteration in tqdm(range(max_iter)):
        grad = grad_g(X)
        g_X = g(X)
        # Backtracking line search
        t = 1.0
        while True:
            X_next = prox_h(X - t * grad, t)
            try:
                np.linalg.cholesky(X_next)
                pd = True
            except np.linalg.LinAlgError:
                pd = False

            if pd:
                lhs = g(X_next)
                rhs = g_X + np.sum(grad * (X_next - X)) + (1 / (2 * t)) * np.linalg.norm(X_next - X, 'fro') ** 2
                if lhs <= rhs:
                    break
            t *= beta

        X = X_next

        # Check duality gap
        gap = duality_gap(X, S, gamma)
        if verbose:
            print(f"Iteration {iteration}, duality gap: {gap:.4e}")

        if gap <= epsilon:
            break

    return X

if __name__=='__main__':
    subset_size = int(sys.argv[1]) if len(sys.argv) > 1 else 492
    matrix = np.loadtxt('data/sp500.txt')
    matrix = matrix[:subset_size, :subset_size]

    x_points = []
    y_points = []
    n_no_zeros = []

    epsilon = 1e-2
    interval_zero = [-epsilon, epsilon]
    gammas = np.logspace(-2, -1, 10)

    for gamma in gammas:
        X_opt = graphical_lasso_proximal_gradient(matrix, gamma)
        print("Primal X (precision matrix):\n", X_opt)

        # Plot the result_pgb with diagonal set to -inf
        result_pgb_plot = np.copy(X_opt)
        np.fill_diagonal(result_pgb_plot, -np.inf)

        plt.imshow(result_pgb_plot, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title(f"Heatmap of result_pgb (gamma={gamma:.2e})")
        plt.savefig(f'outputs/heatmap_gamma_{gamma:.2e}.png') 
        plt.close()

        x_points.append(np.trace(matrix @ X_opt) - np.log(np.linalg.det(X_opt)))

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