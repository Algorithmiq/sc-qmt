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

""" Here is an example of the code to simulate num_exps_per_set different experiments of seesaw for each num_sets
sets of 4 random input states of QMT of the SIC-POVM, with and without noise on the input states
    """  

num_exps_per_set = 1 # number of different experiments per set of input states
num_sets = 1 # number of different sets of input states

vec_delta_ideal4 = np.zeros((num_sets,num_exps_per_set)) # vector of final deltas (4 random input states, only shot noise)
vec_delta_noisy4st001 = np.zeros((num_sets,num_exps_per_set)) # vector of final deltas (4 random input states, coherent noise)

rhoRand4 =  np.zeros((4, 2, 2), dtype=complex)

num_shots = 150000 # number of shots per input state
nu_delta = 1e-6 # nu_delta as defined in the paper

for jj in range(num_sets):

    while np.linalg.matrix_rank(np.column_stack([np.ndarray.flatten(rhoRand4[ll,:,:]) for ll in range(4)]))<4:
        for i in range(0, 4):
            rhoRand4[i] = rand_dm_ginibre(2)
            
    for kk in range(num_exps_per_set):   
        
        id_freq4 = generate_frequencies(num_shots,M,rhoRand4) # only shot noise
        noisy_freq4001 = generate_frequencies_noisy(num_shots,M,rhoRand4,0.0,0.01) # coherent noise with epsilon = 0.01
        
        vec_delta_ideal4[jj,kk] = run_seesaw(rhoRand4,id_freq4,nu_delta)[2] # delta at the final seesaw step
        vec_delta_noisy4st001[jj,kk] = run_seesaw(rhoRand4,noisy_freq4001,nu_delta)[2] # delta at the final seesaw step