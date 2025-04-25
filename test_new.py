import numpy as np
from numpy.linalg import slogdet, inv, norm, LinAlgError
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm 

def project_dual(U, gamma):
    """Project matrix U onto the feasible set of the dual problem using vectorized operations"""
    U_proj = np.clip(U, -gamma, gamma)
    np.fill_diagonal(U_proj, 0)
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

def graphical_lasso_dual(S, gamma, max_iter=20000, epsilon=1e-3, alpha=0.25, beta=0.5, t0=0.1):
    """
    Solves the dual problem of graphical LASSO using projected gradient ascent with backtracking
    """
    n = S.shape[0]
    U = np.eye(n)  # Initial guess (symmetric zero matrix)

    for k in tqdm(range(max_iter)):
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

if __name__=='__main__':
    # np.random.seed(42)
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

        X_opt, U_opt = graphical_lasso_dual(matrix, gamma)
        print("Primal X (precision matrix):\n", X_opt)
        print("Dual U:\n", U_opt)

        # Plot the result_pgb with diagonal set to -inf
        result_pgb_plot = X_opt
        np.fill_diagonal(result_pgb_plot, -np.inf)

        plt.imshow(result_pgb_plot, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title("Heatmap of result_pgb (diagonal set to -inf)")
        plt.savefig(f'outputs/heatmap_gamma_{gamma}.png') 
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

