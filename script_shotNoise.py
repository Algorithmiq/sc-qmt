from functions import *

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

""" Here is the code to simulate num_exps different experiments of QMT of the SIC-POVM with random input states and
with shot noise only (Fig.1 of the paper)
    """

num_exps = 1  # number of different experiments

vec_delta_1000 = np.zeros(num_exps)  # 1000 shots per input state
vec_delta_100000 = np.zeros(num_exps)  # 100000 shots per input state
vec_delta_5000000 = np.zeros(num_exps)  # 5000000 shots per input state

rhoRand = np.zeros((4, 2, 2), dtype=complex)

for jj in range(num_exps):

    while (
        np.linalg.matrix_rank(
            np.column_stack([np.ndarray.flatten(rhoRand[ll, :, :]) for ll in range(4)])
        )
        < 4
    ):
        # we check that the set of input states is a basis in the space of system operators
        for i in range(0, 4):
            rhoRand[i] = rand_dm_ginibre(2)
    # we generate the experimental frequencies
    id_freq_1000 = generate_frequencies(1000, M, rhoRand)
    id_freq_100000 = generate_frequencies(100000, M, rhoRand)
    id_freq_5000000 = generate_frequencies(5000000, M, rhoRand)
    # we fit the frequencies through the single-delta SDP
    vec_delta_1000[jj] = run_SDP_QMT(rhoRand, id_freq_1000, SDPtype="singleDelta")[0]
    vec_delta_100000[jj] = run_SDP_QMT(rhoRand, id_freq_100000, SDPtype="singleDelta")[
        0
    ]
    vec_delta_5000000[jj] = run_SDP_QMT(
        rhoRand, id_freq_5000000, SDPtype="singleDelta"
    )[0]

# Save the results as:
# saveDic = {}
# saveDic['1000'] = vec_delta_1000
# saveDic['100000'] = vec_delta_100000
# saveDic['5000000'] = vec_delta_5000000
# with open('shotNoise_histogram.msgpack', "wb") as outfile:
#    packed = msgpack.packb(saveDic)
#    outfile.write(packed)
