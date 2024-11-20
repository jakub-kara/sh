import numpy as np

def get_energies_and_states(H_ss):
    energies_s, states_ss = np.linalg.eig(H_ss)
    idx = np.argsort(energies_s)
    energies_s = energies_s[idx]
    states_ss = states_ss[:,idx]
    return energies_s, states_ss

def get_deriv_coupling(state_ss, gradH_ssad, ham_ss):
    n_states, _, n_atoms, _ = gradH_ssad.shape

    deriv_coupling_ssad = np.zeros((n_states,n_states,n_atoms,3))

    for s1 in range(n_states):
        for s2 in range(n_states):
            for i in range(n_atoms):
                for d in range(3):
                    deriv_coupling_ssad[s1,s2,i,d] = state_ss[:,s1].T @ gradH_ssad[:,:,i,d] @ state_ss[:,s2]
            if s1 != s2:
                deriv_coupling_ssad[s1,s2] /= ham_ss[s2,s2] - ham_ss[s1,s1]
    
    return deriv_coupling_ssad

def quadratic(position_ad):
    # Quadratic potential with H2-like parameters
    # Intended for 1 atom only

    # 1/2 * 0.35 * x^2

    a = 0.35
    potential_energy = np.array([0.5*a*np.sum(position_ad**2)])
    gradient = a*position_ad

    return potential_energy, gradient

def spin_boson(position_ad):
    # Spin boson potential
    # Intended for 1 atom only

    # H = (a * x^2 + b * x      w              )
    #     (w*                   a * x^2 - b * x)

    a, b, w = 0.1, 0.3, 0.1

    r = np.linalg.norm(position_ad)
    r_hat = position_ad/r

    H_ss = np.zeros((2,2))
    H_ss[0,0] = a*r**2 + b*r
    H_ss[1,1] = a*r**2 - b*r
    H_ss[0,1] = w
    H_ss[1,0] = H_ss[0,1]

    gradH_ssad = np.zeros((2,2,1,3))
    gradH_ssad[0,0,0,:] = 2*a*position_ad + b*r_hat
    gradH_ssad[1,1,0,:] = 2*a*position_ad - b*r_hat

    return H_ss, gradH_ssad

def tully_1(position_ad):
    # First model potential by Tully
    # Intended for 1 atom only moving in 1D

    # V11(x) = A[1-exp(-Bx)], x > 0
    # V11(x) = -A[1-exp(Bx)], x <= 0
    # V22(x) = -V11(x)
    # V12(x) = V21(x) = C exp(-Dx**2)
    # A = 0.01, B = 1.6, C = 0.005, D = 1.0

    A, B, C, D = 0.01, 1.6, 0.005, 1.0

    x = position_ad[0,0]

    H_ss = np.zeros((2,2))
    if x > 0:
        H_ss[0,0] = A*(1 - np.exp(-B*x))
    else:
        H_ss[0,0] = -A*(1 - np.exp(B*x))
    H_ss[1,1] = -H_ss[0,0]
    H_ss[0,1] = C*np.exp(-D*x**2)
    H_ss[1,0] = H_ss[0,1]

    gradH_ssad = np.zeros((2,2,1,3))
    if x > 0:
        gradH_ssad[0,0,0,0] = A*B*np.exp(-B*x)
    else:
        gradH_ssad[0,0,0,0] = A*B*np.exp(B*x)
    gradH_ssad[1,1,0,0] = -gradH_ssad[0,0,0,0]
    gradH_ssad[0,1,0,0] = -2*C*D*x*np.exp(-D*x**2)
    gradH_ssad[1,0,0,0] = gradH_ssad[0,1,0,0]

    return H_ss, gradH_ssad

def tully_1m(position_ad):
    # First model potential by Tully
    # Intended for 1 atom only moving in 1D

    # V11(x) = A[1-exp(-Bx)], x > 0
    # V11(x) = -A[1-exp(Bx)], x <= 0
    # V22(x) = -V11(x)
    # V12(x) = V21(x) = C exp(-Dx**2)
    # A = 0.01, B = 1.6, C = 0.005, D = 1.0

    A, B, C, D = 2/200, 2.5, 0.8/200, 1.2

    x = position_ad[0,0]

    H_ss = np.zeros((2,2))
    H_ss[0,0] = A*(2/(1 + np.exp(-B*x)) - 1)
    H_ss[1,1] = -H_ss[0,0]
    H_ss[0,1] = C*np.exp(-D*x**2)
    H_ss[1,0] = H_ss[0,1]

    gradH_ssad = np.zeros((2,2,1,3))
    gradH_ssad[0,0,0,0] = A*B*np.exp(-B*x)/(1 + np.exp(-B*x))**2
    gradH_ssad[1,1,0,0] = -gradH_ssad[0,0,0,0]
    gradH_ssad[0,1,0,0] = -2*C*D*x*np.exp(-D*x**2)
    gradH_ssad[1,0,0,0] = gradH_ssad[0,1,0,0]

    return H_ss, gradH_ssad

def tully_2(position_ad):
    # Second model potential by Tully
    # Intended for 1 atom only moving in 1D

    # V11(x) = 0
    # V22(x) = -A exp(-Bx**2) + E0
    # V12(x) = V21(x) = C exp(-Dx**2)
    # A = 0.1, B = 0.28, C = 0.015, D = 0.06, E0 = 0.05

    A, B, C, D, E0 = 0.1, 0.28, 0.015, 0.06, 0.05

    x = position_ad[0,0]

    H_ss = np.zeros((2,2))
    H_ss[1,1] = -A*np.exp(-B*x**2) + E0
    H_ss[0,1] = C*np.exp(-D*x**2)
    H_ss[1,0] = H_ss[0,1]

    gradH_ssad = np.zeros((2,2,1,3))
    gradH_ssad[1,1,0,0] = 2*A*B*x*np.exp(-B*x**2)
    gradH_ssad[0,1,0,0] = -2*C*D*x*np.exp(-D*x**2)
    gradH_ssad[1,0,0,0] = gradH_ssad[0,1,0,0]

    return H_ss, gradH_ssad

def tully_3(position_ad):
    # Third model potential by Tully
    # Intended for 1 atom only moving in 1D

    # V11(x) = A
    # V22(x) = -A
    # V12(x) = V21(x) = B exp(Cx) if x < 0
    # v12(x) = V21(x) = B (2 - exp(-Cx)) if x >= 0
    # A = 6e-4, B = 0.1, C = 0.9

    A, B, C = 6.e-4, 0.1, 0.9

    x = position_ad[0,0]

    H_ss = np.zeros((2,2))
    H_ss[0,0] = A
    H_ss[1,1] = -A
    if x < 0:
        H_ss[0,1] = B*np.exp(C*x)
    else:
        H_ss[0,1] = B*(2 - np.exp(-C*x))
    H_ss[1,0] = H_ss[0,1]

    gradH_ssad = np.zeros((2,2,1,3))
    if x < 0:
        gradH_ssad[0,1,0,0] = B*C*np.exp(C*x)
    else:
        gradH_ssad[0,1,0,0] = B*C*np.exp(-C*x)
    gradH_ssad[1,0,0,0] = gradH_ssad[0,1,0,0]

    return H_ss, gradH_ssad
