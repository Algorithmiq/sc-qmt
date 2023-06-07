from sc_qmt import *

theta = 2 * np.pi / 3
phi = 4 * np.pi / 3

v_0 = np.array([1, 0])
v_1 = np.array([1, np.sqrt(2)]) / np.sqrt(3)
v_2 = np.array([1, np.exp(theta * 1j) * np.sqrt(2)]) / np.sqrt(3)
v_3 = np.array([1, np.exp(phi * 1j) * np.sqrt(2)]) / np.sqrt(3)

M = []  # list of effects of the SIC-POVM
M.append(np.outer(v_0.conj().T, v_0) / 2)
M.append(np.outer(v_1.conj().T, v_1) / 2)
M.append(np.outer(v_2.conj().T, v_2) / 2)
M.append(np.outer(v_3.conj().T, v_3) / 2)

""" Here is an example of the code to simulate num_exps_per_set different experiments of seesaw with the cross-validation method to fight finite-shot overfitting for each num_sets sets of 4 random input states of QMT of the SIC-POVM, with and without noise on the input states
    """

num_exps_per_set = 1  # number of different experiments per set of input states
num_sets = 1  # number of different sets of input states

vec_delta_ideal4 = np.zeros(
    (num_sets, num_exps_per_set, 2)
)  # vector of final deltas and numbers of seesaw steps (4 random input states, only shot noise)
vec_delta_noisy4st001 = np.zeros(
    (num_sets, num_exps_per_set, 2)
)  # vector of final deltas and numbers of seesaw steps (4 random input states, coherent noise)

rhoRand4 = np.zeros((4, 2, 2), dtype=complex)

num_shots = 150000  # number of shots per input state
nu_delta = 1e-6  # nu_delta as defined in the paper

for jj in range(num_sets):

    while (
        np.linalg.matrix_rank(
            np.column_stack([np.ndarray.flatten(rhoRand4[ll, :, :]) for ll in range(4)])
        )
        < 4
    ):
        for i in range(0, 4):
            rhoRand4[i] = rand_dm_ginibre(2)

    for kk in range(num_exps_per_set):

        id_freq4 = gen_frequencies_list_noisy(15000, M, rhoRand4, 0.0, 0.00, num_k=10)
        noisy_freq4001 = gen_frequencies_list_noisy(
            15000, M, rhoRand4, 0.0, 0.01, num_k=10
        )  # coherent noise with epsilon = 0.01

        temp = run_seesaw_crossV(
            rhoRand4,
            id_freq4,
            1e-6,
            num_sampling=5,
            num_repetitions=10,
            SDPtype="singleDelta",
        )
        vec_delta_ideal4[jj, kk, 0] = temp[2]  # delta at the final seesaw step
        vec_delta_ideal4[jj, kk, 1] = temp[3]  # number of total seesaw steps

        temp = run_seesaw_crossV(
            rhoRand4,
            noisy_freq4001,
            1e-6,
            num_sampling=5,
            num_repetitions=10,
            SDPtype="singleDelta",
        )
        vec_delta_noisy4st001[jj, kk, 0] = temp[2]  # delta at the final seesaw step
        vec_delta_noisy4st001[jj, kk, 1] = temp[3]  # number of total seesaw steps
