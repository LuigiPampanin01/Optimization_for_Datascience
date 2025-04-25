import numpy as np
from numpy.linalg import inv, slogdet, norm, eigh
import scipy.linalg
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

def soft_threshold_off_diagonal(Z, lam):
    Z_new = Z - np.sign(Z) * lam
    np.fill_diagonal(Z_new, np.diag(Z))  # Preserve diagonal elements
    return np.where(np.abs(Z_new) > lam, Z_new, 0)

def is_pos_def(X):
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False

def proximal_gradient_glasso(S, gamma, tol=1e-2, max_iter=500):
    n = S.shape[0]
    X = np.eye(n)  # initialization
    t = 1.0
    beta = 0.5

    for k in range(max_iter):
        grad_g = S - inv(X)
        # Backtracking line search
        for it in tqdm(range(max_iter)):
            Z = X - t * grad_g
            X_new = soft_threshold_off_diagonal(Z, t * gamma)
            if is_pos_def(X_new):
                g_X_new = np.trace(S @ X_new) - slogdet(X_new)[1]
                g_X = np.trace(S @ X) - slogdet(X)[1]
                rhs = g_X + np.sum(grad_g * (X_new - X)) + (1/(2*t)) * norm(X_new - X, 'fro')**2
                if g_X_new <= rhs:
                    break
            t *= beta  # shrink step size

        # Compute duality gap
        X_inv = inv(X_new)
        U = np.clip(X_inv - S, -gamma, gamma)
        np.fill_diagonal(U, 0)
        try:
            dual_obj = slogdet(S + U)[1] + n
            primal_obj = np.trace(S @ X_new) - slogdet(X_new)[1] + gamma * np.sum(np.abs(X_new) - np.abs(np.diag(X_new)))
            delta = primal_obj - dual_obj
        except np.linalg.LinAlgError:
            delta = np.inf

        if delta <= tol:
            break

        X = X_new

    return X

if __name__=='__main__':
    subset_size = int(sys.argv[1]) if len(sys.argv) > 1 else 492
    matrix = np.loadtxt('data/sp500.txt')
    matrix = matrix[:subset_size, :subset_size]

    x_points = []
    y_points = []
    n_no_zeros = []

    epsilon = 1e-1
    interval_zero = [-epsilon, epsilon]
    gammas = np.logspace(-2, -1, 10)

    for gamma in gammas:
        X_opt = proximal_gradient_glasso(matrix, gamma)

        # Plot the result_pgb with diagonal set to -inf
        result_pgb_plot = np.copy(X_opt)
        np.fill_diagonal(result_pgb_plot, -np.inf)

        plt.imshow(result_pgb_plot, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title(f"Heatmap of result_pgb (gamma={gamma:.2e})")
        plt.savefig(f'outputs/heatmap_gamma_{gamma:.2e}.png') 
        plt.close()

        x_points.append(np.trace(matrix @ X_opt) + np.log(np.linalg.det(X_opt)))

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