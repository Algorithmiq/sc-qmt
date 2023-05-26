from functions import *

theta = 2 * np.pi / 3
phi = 4 * np.pi / 3 

v_0 = np.array([1,0])
v_1 = np.array([1, np.sqrt(2)]) / np.sqrt(3)
v_2 = np.array([1, np.exp(theta * 1j) * np.sqrt(2)]) / np.sqrt(3)
v_3 = np.array([1, np.exp(phi * 1j) * np.sqrt(2)]) / np.sqrt(3)

M = [] # list of effects of the SIC-POVM
M.append(np.outer(v_0.conj().T, v_0)/2)
M.append(np.outer(v_1.conj().T, v_1)/2)
M.append(np.outer(v_2.conj().T, v_2)/2)
M.append(np.outer(v_3.conj().T, v_3)/2)

rhoPaulis = [np.array([[1, 0],
       [0, 0]]), np.array([[0, 0],
       [0, 1]]), np.array([[0.5+0.j, 0.5+0.j],
       [0.5+0.j, 0.5+0.j]]), np.array([[ 0.5+0.j, -0.5+0.j],
       [-0.5-0.j,  0.5+0.j]]), np.array([[0.5+0.j , 0. +0.5j],
       [0. -0.5j, 0.5+0.j ]]), np.array([[0.5+0.j , 0. -0.5j],
       [0. +0.5j, 0.5+0.j ]])] # list of Pauli eigenstates

""" Here is an example of the code to simulate num_exps different single-delta SDP experiments for QMT of the SIC-POVM, with the Pauli eigenstates as input states, and with and without noise thereon
    """  
num_exps = 1 # number of different experiments 
num_shots = 100000 # number of shots per input state

vec_delta_ideal = np.zeros(num_exps) # we store here the SDP delta for each experiment (ideal case)
vec_delta_noisy001 = np.zeros(num_exps) # we store here the SDP delta for each experiment (incoherent noise)
    
for jj in range(num_exps):
    
    id_freq = generate_frequencies(num_shots,M,rhoPaulis) # only shot noise
    noisy_freq = generate_frequencies_noisy(num_shots,M,rhoPaulis,0.001,0.0) # incoherent noise with p = 0.001

    vec_delta_ideal[jj] =run_SDP_QMT(rhoPaulis,id_freq,SDPtype = 'singleDelta')[0]
    vec_delta_noisy001[jj] =run_SDP_QMT(rhoPaulis,noisy_freq,SDPtype = 'singleDelta')[0]