import numpy as np
from numpy.linalg import inv, norm, slogdet
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def proximal_h(X_minus_S, gamma):
    """Proximal operator for h(X) with respect to U"""
    U = np.zeros_like(X_minus_S)
    n = X_minus_S.shape[0]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                U[i,j] = max(-gamma, min(gamma, X_minus_S[i,j]))
    
    return U

def is_pos_def(X):
    """Check if matrix is positive definite"""
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False

def proximal_operator_h(Z, gamma):
    """Proximal operator for h(X) = γ∑|X_ij| for i≠j"""
    X_new = Z.copy()
    n = Z.shape[0]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                if Z[i,j] > gamma:
                    X_new[i,j] -= gamma
                elif Z[i,j] < -gamma:
                    X_new[i,j] += gamma
                else:
                    X_new[i,j] = 0
    
    return X_new

def compute_g(X, S):
    """Compute g(X) = tr(SX) - ln(det(X))"""
    return np.trace(S @ X) - slogdet(X)[1]

def compute_h(X, gamma):
    """Compute h(X) = γ∑|X_ij| for i≠j"""
    n = X.shape[0]
    h_sum = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                h_sum += abs(X[i,j])
    return gamma * h_sum

def compute_gradient_g(X, S):
    """Compute gradient of g(X) = tr(SX) - ln(det(X))"""
    return S - inv(X)

def proximal_gradient_glasso(S, gamma, tol=1e-2, max_iter=1000, X_init=None, verbose=False):
    """
    Solve the graphical LASSO problem using proximal gradient method
    
    Parameters:
    -----------
    S : array-like
        Empirical covariance matrix
    gamma : float
        Regularization parameter
    tol : float
        Tolerance for stopping criterion (duality gap)
    max_iter : int
        Maximum number of iterations
    X_init : array-like or None
        Initial guess for precision matrix. If None, identity matrix is used
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    X : array-like
        Estimated precision matrix
    duality_gaps : list
        History of duality gaps during optimization
    """
    n = S.shape[0]
    X = np.eye(n) if X_init is None else X_init.copy()  # Initialization
    t = 1.0        # Initial step size
    beta = 0.5     # Backtracking line search parameter
    
    # For tracking convergence
    duality_gaps = []
    
    for k in tqdm(range(max_iter), disable=not verbose):
        grad_g_X = compute_gradient_g(X, S)
        
        # Backtracking line search
        while True:
            # Compute Z_k = X_k - t_k ∇g(X_k)
            Z = X - t * grad_g_X
            
            # Apply proximal operator to get X_{k+1}
            X_next = proximal_operator_h(Z, t * gamma)
            
            # Check if X_next is positive definite
            if not is_pos_def(X_next):
                t *= beta
                continue
            
            # Check the sufficient decrease condition
            g_X_next = compute_g(X_next, S)
            g_X = compute_g(X, S)
            diff = X_next - X
            rhs = g_X + np.sum(grad_g_X * diff) + (1/(2*t)) * np.square(norm(diff, 'fro'))
            
            if g_X_next <= rhs:
                break
            
            t *= beta
        
        # Compute the dual variable U based on optimality condition
        X_inv = inv(X_next)
        U = proximal_h(X_inv - S, gamma)
        
        # Compute duality gap
        if is_pos_def(S + U):
            primal_obj = compute_g(X_next, S) + compute_h(X_next, gamma)
            dual_obj = -slogdet(S + U)[1] - n
            delta = primal_obj - dual_obj
        else:
            delta = np.inf
        
        duality_gaps.append(delta)
        
        # Check stopping criterion
        if delta <= tol:
            if verbose:
                print(f"Converged after {k+1} iterations with duality gap: {delta}")
            break
        
        X = X_next
    
    return X, duality_gaps

# Ensure output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    # Ensure output directory exists
    ensure_dir('outputs')
    
    # Load S&P 500 data
    try:
        S = np.loadtxt('data/sp500.txt')
        print(f"Loaded covariance matrix of shape {S.shape}")
    except FileNotFoundError:
        print("Error: Could not find data/sp500.txt")
        print("Generating a synthetic dataset instead...")
        # Generate a synthetic dataset
        np.random.seed(42)
        n = 100
        X = np.random.randn(2*n, n)
        S = (X.T @ X) / (2*n)
    
    # Ensure matrix is positive definite
    if not is_pos_def(S):
        print("Making the input matrix positive definite...")
        min_eig = np.min(np.linalg.eigvalsh(S))
        if min_eig <= 0:
            S = S + (-min_eig + 1e-6) * np.eye(S.shape[0])
    
    # Define gamma values logarithmically spaced (from large to small)
    gammas = np.logspace(-2, -1, 10)[::-1]  # 10 values from 10^-1 to 10^-2, reversed

    # Track metrics for plotting
    g_values = []
    h_values = []
    nonzeros = []
    epsilon = 1e-4  # Threshold for considering an element as non-zero

    # Previous solution to use as initial guess
    X_prev = None

    # Lists to store results in reverse order (for plotting from small to large gamma)
    g_values_ordered = []
    h_values_ordered = []
    nonzeros_ordered = []
    gammas_ordered = gammas[::-1]  # Store the original order (small to large)

    for gamma in gammas:
        print(f"\nSolving with gamma = {gamma:.6f}")
        
        # Solve graphical LASSO using previous solution as initial guess if available
        X_opt, duality_gaps = proximal_gradient_glasso(
            S, gamma, tol=1e-2, max_iter=1000, X_init=X_prev, verbose=True
        )
        
        # Use current solution as initial guess for next iteration
        X_prev = X_opt.copy()
        
        # Calculate metrics
        g_value = compute_g(X_opt, S)
        h_value = compute_h(X_opt, 1.0)  # Using gamma=1 to get raw L1 norm
        
        # Count non-zero off-diagonal elements
        off_diag_mask = ~np.eye(X_opt.shape[0], dtype=bool)
        n_nonzero = np.sum(np.abs(X_opt[off_diag_mask]) > epsilon)
        
        # Store values for later reordering
        g_values.append(g_value)
        h_values.append(h_value)
        nonzeros.append(n_nonzero)
                
    # Reorder the results for plotting (small to large gamma)
    g_values_ordered = g_values[::-1]
    h_values_ordered = h_values[::-1]
    nonzeros_ordered = nonzeros[::-1]

    # Plot trade-off curve with reordered data
    plt.figure(figsize=(8, 6))
    plt.plot(g_values_ordered, h_values_ordered, marker='o', linestyle='-')
    plt.xlabel('g(X) = tr(SX) - ln(det(X))')
    plt.ylabel('L1 Norm of Off-Diagonal Elements')
    plt.title('Trade-off Curve for Graphical LASSO')
    plt.grid(True)
    plt.savefig('outputs/tradeoff_curve.png')
    plt.close()

    # Plot the number of non-zeros as a function of gamma with reordered data
    plt.figure(figsize=(8, 6))
    plt.semilogx(gammas_ordered, nonzeros_ordered, marker='o', linestyle='-')
    plt.xlabel('Regularization Parameter (gamma)')
    plt.ylabel('Number of Non-Zero Off-Diagonal Elements')
    plt.title('Sparsity vs Regularization')
    plt.grid(True)
    plt.savefig('outputs/sparsity_vs_gamma.png')
    plt.close()