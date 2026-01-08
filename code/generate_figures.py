"""
Generate figures for "Communication Beyond Information: Manifold Expansion via High-Dimensional Coupling"

Figures:
1. Eigenvalue emergence under coupling (OU and Kuramoto)
2. Geometric schematic of manifold expansion
3. OU covariance visualization

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, Arc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from scipy.linalg import solve_continuous_lyapunov

# Style settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150


def compute_ou_fisher_eigenvalues(kappa_range, gamma=1.0, sigma=1.0):
    """
    Compute Fisher information eigenvalues for coupled OU process.

    Parameters β = (γ, σ) for symmetric case.
    Stationary distribution is bivariate Gaussian with:
        Σ_11 = Σ_22 = σ²/(2γ)
        Σ_12 = κσ²/(2γ(γ + 2κ))

    Fisher information for Gaussian:
        I_ij = (1/2) tr(Σ⁻¹ ∂Σ/∂θ_i Σ⁻¹ ∂Σ/∂θ_j) + (∂μ/∂θ_i)ᵀ Σ⁻¹ (∂μ/∂θ_j)

    For zero mean, only first term matters.
    """
    eigenvalues = []

    for kappa in kappa_range:
        # Compute covariance matrix
        v = sigma**2 / (2 * gamma)  # variance
        if kappa > 1e-10:
            rho = kappa / (gamma + 2*kappa)  # correlation
        else:
            rho = 0.0

        Sigma = np.array([[v, rho*v], [rho*v, v]])
        Sigma_inv = np.linalg.inv(Sigma)

        # Derivatives of Sigma w.r.t. parameters
        # ∂Σ/∂γ
        dv_dgamma = -sigma**2 / (2 * gamma**2)
        if kappa > 1e-10:
            drho_dgamma = -kappa / (gamma + 2*kappa)**2
        else:
            drho_dgamma = 0.0

        dSigma_dgamma = np.array([
            [dv_dgamma, drho_dgamma*v + rho*dv_dgamma],
            [drho_dgamma*v + rho*dv_dgamma, dv_dgamma]
        ])

        # ∂Σ/∂σ
        dv_dsigma = sigma / gamma
        dSigma_dsigma = np.array([
            [dv_dsigma, rho*dv_dsigma],
            [rho*dv_dsigma, dv_dsigma]
        ])

        # ∂Σ/∂ρ (treating ρ as the "interaction coordinate")
        dSigma_drho = np.array([[0, v], [v, 0]])

        # Fisher information matrix (3x3 for γ, σ, ρ)
        params = [dSigma_dgamma, dSigma_dsigma, dSigma_drho]
        n_params = 3
        Fisher = np.zeros((n_params, n_params))

        for i in range(n_params):
            for j in range(n_params):
                Fisher[i, j] = 0.5 * np.trace(
                    Sigma_inv @ params[i] @ Sigma_inv @ params[j]
                )

        # Eigenvalues
        eigs = np.linalg.eigvalsh(Fisher)
        eigenvalues.append(sorted(eigs, reverse=True))

    return np.array(eigenvalues)


def compute_kuramoto_fisher_eigenvalues(K_range, K_c=1.0, D=0.5):
    """
    Compute Fisher information eigenvalues for Kuramoto model.

    Simplified model: order parameter r emerges at K > K_c.
    Below K_c: r = 0, Ψ unidentifiable (Fisher eigenvalue = 0)
    Above K_c: r > 0, Ψ identifiable (Fisher eigenvalue > 0)

    We model this as:
        r(K) = 0 for K < K_c
        r(K) = sqrt(1 - K_c/K) for K >= K_c (standard mean-field result)

    Fisher information for Ψ scales with r² (phase sensitivity requires coherence)
    """
    eigenvalues = []

    for K in K_range:
        if K < K_c:
            r = 0.0
        else:
            r = np.sqrt(1 - K_c / K) if K > K_c else 0.0

        # Fisher eigenvalues:
        # λ_1: always identifiable (related to D or frequency distribution)
        # λ_2: phase sensitivity, scales with r²
        lambda_1 = 1.0 / D  # noise parameter always identifiable
        lambda_2 = r**2 * K / D  # phase sensitivity emerges with r

        eigenvalues.append([lambda_1, lambda_2])

    return np.array(eigenvalues)


def fig1_eigenvalue_emergence(save_path='../figures/fig1_eigenvalue_emergence.pdf'):
    """
    Figure 1: Eigenvalue emergence under coupling.
    Side-by-side: OU (κ_c = 0) and Kuramoto (K_c > 0)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Coupled OU
    ax = axes[0]
    kappa_range = np.linspace(0, 2, 100)
    eigs = compute_ou_fisher_eigenvalues(kappa_range)

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = [r'$\lambda_1$ (variance)', r'$\lambda_2$ (variance)', r'$\lambda_3$ (correlation)']

    for i in range(3):
        ax.plot(kappa_range, eigs[:, i], color=colors[i], linewidth=2, label=labels[i])

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate(r'$\kappa_c = 0$', xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, color='gray')

    ax.set_xlabel(r'Coupling strength $\kappa$')
    ax.set_ylabel(r'Fisher eigenvalue $\lambda_k$')
    ax.set_title('(A) Coupled Ornstein-Uhlenbeck\n(Transversality criterion)')
    ax.legend(loc='right')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Highlight emergence region
    ax.fill_between([0, 0.3], 0, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.text(0.15, ax.get_ylim()[1]*0.5, 'Emergence\nregion', ha='center',
            fontsize=8, color='green', alpha=0.7)

    # Panel B: Kuramoto
    ax = axes[1]
    K_c = 1.0
    K_range = np.linspace(0, 3, 100)
    eigs = compute_kuramoto_fisher_eigenvalues(K_range, K_c=K_c)

    colors = ['#2E86AB', '#F18F01']
    labels = [r'$\lambda_1$ (noise)', r'$\lambda_2$ (phase)']

    for i in range(2):
        ax.plot(K_range, eigs[:, i], color=colors[i], linewidth=2, label=labels[i])

    ax.axvline(x=K_c, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate(r'$K_c$', xy=(K_c + 0.1, 0.95), xycoords=('data', 'axes fraction'),
                fontsize=10, color='gray')

    ax.set_xlabel(r'Coupling strength $K$')
    ax.set_ylabel(r'Fisher eigenvalue $\lambda_k$')
    ax.set_title('(B) Kuramoto oscillators\n(Symmetry-breaking criterion)')
    ax.legend(loc='right')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Highlight emergence region
    ax.fill_between([K_c, K_c + 0.5], 0, ax.get_ylim()[1], alpha=0.1, color='green')
    ax.text(K_c + 0.25, ax.get_ylim()[1]*0.5, 'Emergence\nregion', ha='center',
            fontsize=8, color='green', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def fig2_geometric_schematic(save_path='../figures/fig2_geometric_schematic.pdf'):
    """
    Figure 2: Geometric schematic of manifold expansion.
    Shows M_0 (constraint submanifold) and how F_κ escapes it.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw the ambient space P (as a box outline)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)

    # Constraint submanifold M_0 (horizontal band)
    M0_y = 1.0
    ax.fill_between([0, 4], M0_y - 0.15, M0_y + 0.15, alpha=0.3, color='#2E86AB')
    ax.plot([0, 4], [M0_y, M0_y], 'b-', linewidth=2, label=r'$\mathcal{M}_0$ (constraint submanifold)')
    ax.text(4.1, M0_y, r'$\mathcal{M}_0$', fontsize=12, color='#2E86AB', va='center')

    # Image of F_0 (on M_0)
    x_F0 = np.linspace(0.5, 2.5, 50)
    y_F0 = M0_y * np.ones_like(x_F0)
    ax.plot(x_F0, y_F0, 'r-', linewidth=3, label=r'Image$(F_0)$')
    ax.scatter([0.5, 2.5], [M0_y, M0_y], color='red', s=50, zorder=5)
    ax.text(1.5, M0_y - 0.35, r'Image$(F_0) \subseteq \mathcal{M}_0$',
            fontsize=10, ha='center', color='red')

    # Image of F_κ (escaping M_0)
    x_Fk = np.linspace(0.5, 3.0, 50)
    y_Fk = M0_y + 0.8 * (x_Fk - 0.5)**0.7
    ax.plot(x_Fk, y_Fk, color='#F18F01', linewidth=3, label=r'Image$(F_\kappa)$')
    ax.scatter([0.5], [M0_y], color='#F18F01', s=50, zorder=5)
    ax.scatter([3.0], [y_Fk[-1]], color='#F18F01', s=50, zorder=5)
    ax.text(2.8, y_Fk[-1] + 0.2, r'Image$(F_\kappa)$', fontsize=10, color='#F18F01')

    # Arrow showing "new identifiable direction"
    ax.annotate('', xy=(2.0, 2.3), xytext=(2.0, M0_y + 0.2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(2.15, 1.7, 'New identifiable\ndirection', fontsize=9, color='green')

    # Transverse direction η
    ax.annotate('', xy=(0.2, 2.8), xytext=(0.2, 0.5),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(0.35, 1.7, r'$\eta$ (transverse)', fontsize=10, color='gray', rotation=90, va='center')

    # Tangent direction f
    ax.annotate('', xy=(3.8, 0.3), xytext=(0.5, 0.3),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
    ax.text(2.0, 0.1, r'$f$ (tangent to $\mathcal{M}_0$)', fontsize=10, color='gray', ha='center')

    # Ambient space label
    ax.text(4.2, 3.2, r'$\mathcal{P}$', fontsize=14, style='italic')

    # Add annotation for rank increase
    ax.text(0.5, 3.0, r'$r(\kappa) = r(0) + 1$', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Manifold Expansion: Constraint Release', fontsize=12, pad=20)

    # Legend
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


def fig3_ou_covariance(save_path='../figures/fig3_ou_covariance.pdf'):
    """
    Figure 3: OU covariance visualization.
    Shows correlation ρ vs κ and the 3D Gaussian manifold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Correlation vs coupling
    ax = axes[0]
    gamma = 1.0
    kappa_range = np.linspace(0, 3, 100)
    rho = kappa_range / (gamma + 2*kappa_range)

    ax.plot(kappa_range, rho, 'b-', linewidth=2.5)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(3.05, 0.5, r'$\rho \to 0.5$', fontsize=9, color='gray', va='center')

    ax.fill_between(kappa_range, 0, rho, alpha=0.2, color='blue')

    ax.set_xlabel(r'Coupling strength $\kappa$')
    ax.set_ylabel(r'Correlation $\rho = \Sigma_{12}/\sqrt{\Sigma_{11}\Sigma_{22}}$')
    ax.set_title('(A) Correlation emerges under coupling')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add formula
    ax.text(1.5, 0.15, r'$\rho(\kappa) = \frac{\kappa}{\gamma + 2\kappa}$',
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # Panel B: Schematic of Gaussian manifold
    ax = axes[1]

    # Draw axes
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3)

    # Independence submanifold (ρ = 0 plane)
    ax.fill_between([0, 3], 0.3, 0.7, alpha=0.3, color='blue')
    ax.plot([0, 3], [0.5, 0.5], 'b-', linewidth=2)
    ax.text(3.1, 0.5, r'$\rho = 0$', fontsize=10, color='blue', va='center')

    # Trajectory under coupling
    kappa_vis = np.linspace(0, 2, 50)
    rho_vis = kappa_vis / (1 + 2*kappa_vis)
    x_traj = 0.5 + kappa_vis
    y_traj = 0.5 + rho_vis * 2  # Scale for visualization

    ax.plot(x_traj, y_traj, 'r-', linewidth=2.5, label=r'Trajectory as $\kappa$ increases')
    ax.scatter([0.5], [0.5], color='red', s=80, zorder=5, marker='o')
    ax.scatter([x_traj[-1]], [y_traj[-1]], color='red', s=80, zorder=5, marker='s')

    ax.annotate(r'$\kappa = 0$', xy=(0.5, 0.5), xytext=(0.2, 0.9),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)
    ax.annotate(r'$\kappa > 0$', xy=(x_traj[-1], y_traj[-1]), xytext=(2.8, 1.8),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)

    # Axes labels
    ax.annotate('', xy=(3.2, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(3.3, 0, r'$\Sigma_{11}, \Sigma_{22}$', fontsize=10, va='center')

    ax.annotate('', xy=(0, 2.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.1, 2.9, r'$\rho$', fontsize=10)

    ax.set_title('(B) Escape from independence submanifold')
    ax.axis('off')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("Generating figures for Manifold Expansion paper...")
    print("=" * 50)

    fig1_eigenvalue_emergence()
    fig2_geometric_schematic()
    fig3_ou_covariance()

    print("=" * 50)
    print("All figures generated successfully!")
