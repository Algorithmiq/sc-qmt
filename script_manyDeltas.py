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

rhoPaulis = [
    np.array([[1, 0], [0, 0]]),
    np.array([[0, 0], [0, 1]]),
    np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]]),
    np.array([[0.5 + 0.0j, -0.5 + 0.0j], [-0.5 - 0.0j, 0.5 + 0.0j]]),
    np.array([[0.5 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 + 0.0j]]),
    np.array([[0.5 + 0.0j, 0.0 - 0.5j], [0.0 + 0.5j, 0.5 + 0.0j]]),
]  # list of Pauli eigenstates

""" Here is the code to simulate num_exps different experiments of QMT via the many-deltas SDP of the SIC-POVM with the Pauli eigenstates and
with unbalanced coherent and incoherent noise on the input states (Fig.5 of the paper)
    """

num_shots = 100000  # number of shots per input state
num_exps = 1  # number of numerical experiments


vec_manydeltas_Pauli_incoh = np.zeros(
    (4, 6, num_exps)
)  # We store here the values of the deltas for incoherent noise
vec_manydeltas_Pauli_coh = np.zeros(
    (4, 6, num_exps)
)  # We store here the values of the deltas for coherent noise

zeroNoise = np.zeros(6)
cohNoiseVec = [
    0.0,
    0.01,
    0.0,
    0.0,
    0.0,
    0.0,
]  # vector for the coherent noise on the second input state
incohNoiseVec = [
    0.0,
    0.00,
    0.1,
    0.1,
    0.0,
    0.0,
]  # vector for incoherent noise on the third and fourth input state

for jj in range(num_exps):  # loop over num_exps experiments
    # We generate the experimental frequencies with noisy input states
    noisy_freqPaulis_incoh = generate_frequencies_noisy_states_vec(
        num_shots, M, rhoPaulis, incohNoiseVec, zeroNoise
    )
    noisy_freqPaulis_coh = generate_frequencies_noisy_states_vec(
        num_shots, M, rhoPaulis, zeroNoise, cohNoiseVec
    )

    # We process the data through the many-deltas SDP
    vec_manydeltas_Pauli_incoh[:, :, jj] = run_SDP_manyDeltas(
        rhoPaulis, noisy_freqPaulis_incoh
    )[0]
    vec_manydeltas_Pauli_coh[:, :, jj] = run_SDP_manyDeltas(
        rhoPaulis, noisy_freqPaulis_coh
    )[0]

# Save the results as:
# saveDic = {}
# saveDic['incoherent'] = vec_manydeltas_Pauli_incoh
# saveDic['coherent'] = vec_manydeltas_Pauli_coh
# with open('manyDeltas.msgpack', "wb") as outfile:
#    packed = msgpack.packb(saveDic)
#    outfile.write(packed)
