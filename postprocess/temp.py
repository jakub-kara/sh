import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

np.random.seed(42)

def lowdin(V):
    """Orthonormalize the columns of V using Löwdin's method."""
    S = V.conj().T @ V
    S_inv_sqrt = fractional_matrix_power(S, -0.5)
    return V @ S_inv_sqrt

def permute(U_ref, U_target):
    ovl = np.abs(U_ref.conj().T @ U_target)**2
    row, col = linear_sum_assignment(-ovl)
    print(col)
    return U_target[:,col]

def flip(U_ref, U_target):
    ovl = U_ref.conj().T @ U_target
    for i in range(ovl.shape[0]):
        if ovl[i,i] < 0:
            U_target[:,i] *= -1
    return U_target

def procrustes(U_ref, U_target):
    """Align U_target to U_ref using Procrustes (unitary), assuming both are square."""
    M = U_ref.conj().T @ U_target
    U, _, Vh = np.linalg.svd(M)
    rot = U @ Vh
    rot[:,-1] /= np.linalg.det(rot)
    return U_target @ rot.T

def phase(U_ref, U_target):
    """Align phases of U_target to match U_ref."""
    aligned = np.zeros_like(U_target, dtype=U_target.dtype)
    for i in range(U_target.shape[1]):
        phase = np.vdot(U_ref[:, i], U_target[:, i])
        theta = np.angle(phase)
        aligned[:, i] = U_target[:, i] * np.exp(-1j * theta)
    return aligned

def expand(U, target_dim):
    """Pad square matrix U to target_dim with identity block."""
    curr_dim = U.shape[0]
    if curr_dim == target_dim:
        return U
    padded = np.eye(target_dim, dtype=U.dtype)
    padded[:curr_dim, :curr_dim] = U
    return padded

def pad(A, B):
    """Pad A and B to same square size."""
    m = max(A.shape[0], B.shape[0])
    return expand(A, m), expand(B, m)

# --- Generate synthetic evolving bases ---
T = 10          # number of time steps
m0 = 3          # initial basis size
V_list = []
m_t = m0

for t in range(T):
    if t == 5:
        m_t += 1  # Add one basis vector at t=5
    A = np.random.randn(m_t, m_t) + 1j * np.random.randn(m_t, m_t)
    V_list.append(A)

arr1 = np.array([[1,1], [-1,1]], dtype=float).T
arr2 = np.array([[-2,1], [1,2]], dtype=float).T
arr1 = lowdin(arr1)
arr2 = lowdin(arr2)
arr2 = permute(arr1, arr2)
breakpoint()

# --- Process: Orthonormalize, align, track ---
U_prev = lowdin(V_list[0])
U_list = [U_prev]

for t in range(1, T):
    V_t = V_list[t]
    U_raw = lowdin(V_t)

    # Pad both previous and current basis to max dimension
    U_prev_pad, U_raw_pad = pad(U_prev, U_raw)

    # Align and phase-align
    U_aligned = procrustes(U_prev_pad, U_raw_pad)
    U_aligned = phase(U_prev_pad, U_aligned)

    # Truncate to current basis size
    m_curr = U_raw.shape[0]
    U_curr = U_aligned[:m_curr, :m_curr]

    U_list.append(U_curr)
    U_prev = U_curr

# --- Compute and plot overlaps ---
overlaps = []
for t in range(1, T):
    U_prev, U_curr = U_list[t-1], U_list[t]
    U_prev_pad, U_curr_pad = pad(U_prev, U_curr)

    k = min(U_prev.shape[1], U_curr.shape[1])
    ov = np.abs(np.diag(U_prev_pad[:, :k].conj().T @ U_curr_pad[:, :k]))
    overlaps.append(ov)

breakpoint()

# Plot
plt.figure(figsize=(10, 4))
for i in range(overlaps.shape[1]):
    plt.plot(overlaps[:, i], label=f'Mode {i+1}')
plt.title("Mode Overlaps Between Time Steps (Abstract Basis, With Growth)")
plt.xlabel("Time step")
plt.ylabel("Overlap (|⟨uᵗ⁻¹|uᵗ⟩|)")
plt.ylim(0.9, 1.01)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
