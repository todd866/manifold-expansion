"""
Operational validation: Eigenvalue emergence in coupled OU processes.

This script computes the ACTUAL pullback Fisher information I_B = J^T g J
on the physical parameter space B = (gamma, sigma), not a schematic.

The computation:
1. For each coupling kappa, compute stationary covariance Sigma(kappa, beta)
2. Compute Jacobian J = d(Sigma_11, Sigma_12) / d(gamma, sigma)
3. Compute Fisher metric g on Gaussian manifold (known analytically)
4. Compute pullback I_B = J^T g J
5. Plot eigenvalues of I_B vs kappa

This validates Theorem 1 Part II: rank increases from 1 to 2 at kappa > 0.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ou_covariance(gamma, sigma, kappa):
    """
    Stationary covariance for symmetric coupled OU.

    Returns (Sigma_11, Sigma_12) where Sigma_22 = Sigma_11 by symmetry.

    Formulas:
        Sigma_11 = sigma^2 (gamma + kappa) / (2 gamma (gamma + 2 kappa))
        Sigma_12 = kappa sigma^2 / (2 gamma (gamma + 2 kappa))
    """
    denom = 2 * gamma * (gamma + 2 * kappa)
    Sigma_11 = sigma**2 * (gamma + kappa) / denom
    Sigma_12 = kappa * sigma**2 / denom
    return Sigma_11, Sigma_12

def jacobian_numerical(gamma, sigma, kappa, eps=1e-7):
    """
    Compute Jacobian J = d(Sigma_11, Sigma_12) / d(gamma, sigma) numerically.

    Returns 2x2 matrix: J[i,j] = d(Sigma_i) / d(beta_j)
    """
    J = np.zeros((2, 2))

    # Partial w.r.t. gamma
    S11_plus, S12_plus = ou_covariance(gamma + eps, sigma, kappa)
    S11_minus, S12_minus = ou_covariance(gamma - eps, sigma, kappa)
    J[0, 0] = (S11_plus - S11_minus) / (2 * eps)
    J[1, 0] = (S12_plus - S12_minus) / (2 * eps)

    # Partial w.r.t. sigma
    S11_plus, S12_plus = ou_covariance(gamma, sigma + eps, kappa)
    S11_minus, S12_minus = ou_covariance(gamma, sigma - eps, kappa)
    J[0, 1] = (S11_plus - S11_minus) / (2 * eps)
    J[1, 1] = (S12_plus - S12_minus) / (2 * eps)

    return J

def fisher_metric_gaussian(Sigma_11, Sigma_12):
    """
    Fisher information metric on centered bivariate Gaussians.

    For covariance parameterization (Sigma_11, Sigma_22, Sigma_12) with Sigma_22 = Sigma_11,
    the Fisher metric is derived from:
        g_ij = (1/2) Tr(Sigma^{-1} dSigma/dtheta_i Sigma^{-1} dSigma/dtheta_j)

    We use coordinates (Sigma_11, Sigma_12) with constraint Sigma_22 = Sigma_11.
    """
    # Build covariance matrix
    Sigma = np.array([[Sigma_11, Sigma_12],
                      [Sigma_12, Sigma_11]])

    det_Sigma = Sigma_11**2 - Sigma_12**2
    if det_Sigma <= 0:
        return np.eye(2) * 1e10  # Degenerate case

    Sigma_inv = np.array([[Sigma_11, -Sigma_12],
                          [-Sigma_12, Sigma_11]]) / det_Sigma

    # Derivative matrices w.r.t. our coordinates
    # d Sigma / d Sigma_11 (remembering Sigma_22 = Sigma_11)
    dS_dS11 = np.array([[1, 0],
                        [0, 1]])

    # d Sigma / d Sigma_12
    dS_dS12 = np.array([[0, 1],
                        [1, 0]])

    # Compute Fisher metric components: g_ij = (1/2) Tr(Sigma^{-1} dS_i Sigma^{-1} dS_j)
    g = np.zeros((2, 2))
    dS = [dS_dS11, dS_dS12]

    for i in range(2):
        for j in range(2):
            g[i, j] = 0.5 * np.trace(Sigma_inv @ dS[i] @ Sigma_inv @ dS[j])

    return g

def pullback_fisher(gamma, sigma, kappa):
    """
    Compute pullback Fisher information I_B = J^T g J on parameter space B.

    Returns 2x2 matrix and its eigenvalues.
    """
    # Get covariance
    Sigma_11, Sigma_12 = ou_covariance(gamma, sigma, kappa)

    # Jacobian: d(Sigma_11, Sigma_12) / d(gamma, sigma)
    J = jacobian_numerical(gamma, sigma, kappa)

    # Fisher metric on Gaussian manifold
    g = fisher_metric_gaussian(Sigma_11, Sigma_12)

    # Pullback: I_B = J^T g J
    I_B = J.T @ g @ J

    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(I_B)

    return I_B, np.sort(eigenvalues)[::-1]

def main():
    # Parameters
    gamma = 1.0
    sigma = 1.0
    kappa_values = np.linspace(0, 2, 100)

    # Compute eigenvalues for each kappa
    eigenvalues_list = []
    ranks = []

    for kappa in kappa_values:
        I_B, eigs = pullback_fisher(gamma, sigma, kappa)
        eigenvalues_list.append(eigs)
        # Numerical rank (eigenvalues > threshold)
        ranks.append(np.sum(eigs > 1e-10))

    eigenvalues_array = np.array(eigenvalues_list)

    # Create figure with more space
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Eigenvalues vs kappa
    ax = axes[0]
    ax.plot(kappa_values, eigenvalues_array[:, 0], 'b-', linewidth=2, label=r'$\lambda_1$')
    ax.plot(kappa_values, eigenvalues_array[:, 1], 'r-', linewidth=2, label=r'$\lambda_2$')
    ax.set_xlabel(r'Coupling strength $\kappa$', fontsize=11)
    ax.set_ylabel(r'Fisher eigenvalue', fontsize=11)
    ax.set_title(r'(A) Eigenvalues of $I_\mathcal{B} = F_\kappa^* g$', fontsize=11)
    ax.legend(loc='center right', fontsize=10)
    ax.set_xlim(0, 2)
    ax.set_ylim(bottom=0, top=6)
    ax.text(0.05, 0.35, r'$\lambda_2 = 0$ at $\kappa = 0$' + '\n' + r'$\lambda_2 > 0$ for $\kappa > 0$',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Panel B: Rank vs kappa (zoom on transition)
    ax = axes[1]
    kappa_fine = np.linspace(0, 0.5, 200)
    lambda2_fine = []
    for k in kappa_fine:
        _, eigs = pullback_fisher(gamma, sigma, k)
        lambda2_fine.append(eigs[1])

    ax.semilogy(kappa_fine[1:], lambda2_fine[1:], 'r-', linewidth=2)
    ax.axhline(y=1e-10, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'Coupling strength $\kappa$', fontsize=11)
    ax.set_ylabel(r'$\lambda_2$ (log scale)', fontsize=11)
    ax.set_title(r'(B) Rank transition at $\kappa_c = 0$', fontsize=11)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(1e-6, 1)
    ax.text(0.95, 0.05, r'Rank $1 \to 2$' + '\n' + r'at any $\kappa > 0$',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout(pad=1.5)

    # Save figure
    output_dir = Path(__file__).resolve().parent.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / 'fig4_operational_validation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'fig4_operational_validation.png', bbox_inches='tight', dpi=150)
    print(f"Saved to {output_dir / 'fig4_operational_validation.pdf'}")

    # Print verification
    print("\nVerification:")
    print(f"At kappa=0: eigenvalues = {pullback_fisher(gamma, sigma, 0)[1]}")
    print(f"At kappa=0.01: eigenvalues = {pullback_fisher(gamma, sigma, 0.01)[1]}")
    print(f"At kappa=0.1: eigenvalues = {pullback_fisher(gamma, sigma, 0.1)[1]}")
    print(f"At kappa=1.0: eigenvalues = {pullback_fisher(gamma, sigma, 1.0)[1]}")

if __name__ == '__main__':
    main()
