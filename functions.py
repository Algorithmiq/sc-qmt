import random
from operator import countOf
import timeit 
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
from qutip import rand_dm,rand_dm_ginibre
import pickle
import msgpack
import msgpack_numpy as m
m.patch()
from itertools import product, starmap
import math
import qutip.random_objects as qtrandom
import scipy.sparse as sp
from numpy import ndarray
from qutip.operators import identity

def trace_distance(rho, sigma) -> float:
    r"""Calculate the trace distance between two density matrices.
    The trace distance between $\rho$ and $\sigma$ is defined as
    $$ T(\rho,\sigma) = \frac 12 \| \rho-\sigma \|_1 = \frac 12 \sqrt{(\rho-\sigma)^\dag
        (\rho-\sigma)}
    Args:
        rho (np.ndarray): First input matrix.
        sigma (np.ndarray): Second input matrix.
    Returns:
        float: the trace distance between rho and sigma.
    Example:
        >>> A = np.array([[1, 0], [0, 0]])
        >>> B = np.array([[0, 0], [0, 1]])
        >>> trace_distance(A, B)
        1.0
    """
    diff = rho - sigma
    svals = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(svals)

class myarray(ndarray):    
    @property
    def H(self):
        return self.conj().T
    
def dep_channel(rho,p): # create a depolarizing channel with noise strength p
    return ((1-p)*rho+p/2*np.identity(2))

def amp_damping_channel(rho,gamma): # create an amplitude damping channel with noise strength p
    K0 = np.array([[1,0],[0,np.sqrt(1-gamma)]])
    K1 = np.array([[0,np.sqrt(gamma)],[0,0]])
    return (K0@rho@K0.transpose()+K1@rho@K1.transpose())

def phase_damping_channel(rho,gamma): # create a phase damping channel with noise strength p
    K0 = np.array([[1,0],[0,np.sqrt(1-gamma)]])
    K1 = np.array([[0,0],[0,np.sqrt(gamma)]])
    return (K0@rho@K0.transpose()+K1@rho@K1.transpose())

def random_noise(rho,pgamma): # create the incoherent noise map (Eq.(19))
    threeOutCoin=random.random()
    if threeOutCoin<0.33:
        return dep_channel(rho,pgamma)
    elif threeOutCoin>0.66:
        return amp_damping_channel(rho,pgamma)
    else:
        return phase_damping_channel(rho,pgamma)
    
def random_rotation(dim,phiMagn,psiMagn): # create a random rotation (Eq.(20))
    # phiMagn and psiMagn: magnitude of the random angles phi and psi in Eq.(20)
    randM = np.zeros((dim,dim),dtype=complex)
    
    phi = np.arcsin(np.sqrt(phiMagn * random.random()))
    psi = psiMagn * 2 * math.pi * random.random()
    varpsi = 2 * math.pi * random.random()
    
    randM[0,0] = np.exp(1j*psi)*np.cos(phi)
    randM[1,1] = np.exp(-1j*psi)*np.cos(phi)
    randM[0,1] = -np.exp(-1j*varpsi)*np.sin(phi)
    randM[1,0] = np.exp(1j*varpsi)*np.sin(phi)
    
    return randM

"""
dim: dimension of the system Hilbert space
"""

def generate_frequencies(num_shots,list_effects,list_states):
    """Function to generate the experimental frequencies for the different effects (in list_effects) of a POVM obtained
by measuring the states in list_states with num_shots shots

    Args:
        num_shots (int): number of shots per input state
        list_effects (list): list of dim x dim matrices representing the POVM effects
        list_states (list): list of dim x dim density matrices for the input states

    Returns:
        np.ndarray: matrix of experimental frequencies, whose element `[j,k]` is the frequency for the jth effect with the
kth input state
    """
    num_inputStates = len(list_states) # number of input states
    num_outcomes = len(list_effects) # number of possible outcomes (i.e., number of effects)

    # In prob_matrix we will store the ideal probabilities of getting an outcome given an input state 
    prob_matrix = np.empty((num_outcomes, num_inputStates)) 

    for kk in range(0, num_inputStates):
        for jj in range(0, num_outcomes):
            # We compute the ideal probabilities according to Born's rule
            prob_matrix[jj, kk] = np.real(np.trace(list_effects[jj] @ list_states[kk]))

    # Here we will store the experimental frequencies of the different outcomes    
    frequencies = np.zeros((num_outcomes, num_inputStates))

    for kk in range(0, num_inputStates):
        # We sample the different experimental outcomes (from module "random")
        sample = np.random.choice(range(num_outcomes), size=num_shots, p=prob_matrix[:, kk])
        outc, counts = np.unique(sample, return_counts=True)
        frequencies[outc, kk] = counts / num_shots
    return frequencies

def generate_frequencies_noisy(num_shots,list_effects,list_states,p,rotMagnitude):
    """Function to generate the experimental frequencies for the different effects (in list_effects) of a POVM obtained
by measuring the states in list_states with num_shots shots and with both incoherent and coherent noise applied to the input states (same noise magnitude over all the input states)

    Args:
        num_shots (int): number of shots per input state
        list_effects (list): list of dim x dim matrices representing the POVM effects
        list_states (list): list of dim x dim density matrices for the input states
        p (float): magnitude of incoherent noise
        rotMagnitude (float): magnitude of coherent

    Returns:
        np.ndarray: matrix of experimental frequencies, whose element `[j,k]` is the frequency for the jth effect with the
kth input state
    """    
    num_inputStates = len(list_states) # number of input states
    num_outcomes = len(list_effects) # number of possible outcomes (i.e., number of effects)
    dim = len(list_states[0])
    
    # In prob_matrix we will store the ideal probabilities of getting an outcome given an input state 
    prob_matrix = np.empty((num_outcomes, num_inputStates), dtype=object) 

    for kk in range(0, num_inputStates):
        randRot = random_rotation(dim,rotMagnitude,rotMagnitude)
        for jj in range(0, num_outcomes):
            # We compute the ideal probabilities according to Born's rule
            prob_matrix[jj, kk] = np.trace(list_effects[jj] @ randRot @ random_noise(list_states[kk],p) @ randRot.conj().transpose()) 
            
    # Here we will store the experimental results        
    sample = np.zeros((num_shots, num_inputStates))

    for kk in range(0, num_inputStates):
        # We sample the different experimental outcomes (from module "random")
        sample[:, kk] = np.array(random.choices(range(num_outcomes), weights = prob_matrix[:, kk], k = num_shots))

    # Here we will store the experimental frequencies of the different outcomes    
    frequencies = np.zeros((num_outcomes, num_inputStates))

    for kk in range(0, num_inputStates):
        for jj in range(0, num_outcomes):
            # We compute the frequency with an ad hoc function (from module "operator")
            frequencies[jj, kk] = countOf(sample[:, kk], jj)/num_shots

    return frequencies #matrix of frequencies. frequencies[j,k] gives the frequency for the jth effect and kth state

def generate_frequencies_noisy_states_vec(num_shots,list_effects,list_states,p_vec,rotMagnitude_vec):
    """Function to generate the experimental frequencies for the different effects (in list_effects) of a POVM obtained
by measuring the states in list_states with num_shots shots and with both incoherent and coherent noise applied to the input states (noise magnitude can vary over the different input states)

    Args:
        num_shots (int): number of shots per input state
        list_effects (list): list of dim x dim matrices representing the POVM effects
        list_states (list): list of dim x dim density matrices for the input states
        p_vec (list): list of len(list_states) floats whose j-th element is the magnitude of incoherent noise on the j-th input state
        rotMagnitude_vec (list): list of len(list_states) floats whose j-th element is the magnitude of coherent noise on the j-th input state

    Returns:
        np.ndarray: matrix of experimental frequencies, whose element `[j,k]` is the frequency for the jth effect with the
kth input state
    """        
    
    num_inputStates = len(list_states) # number of input states
    num_outcomes = len(list_effects) # number of possible outcomes (i.e., number of effects)
    dim = len(list_states[0])
    list_states_noisy = []
    
    # In prob_matrix we will store the ideal probabilities of getting an outcome given an input state 
    prob_matrix = np.empty((num_outcomes, num_inputStates), dtype=object) 

    for kk in range(0, num_inputStates):
        randRot = random_rotation(dim,rotMagnitude_vec[kk],rotMagnitude_vec[kk])
        list_states_noisy.append(randRot @ random_noise(list_states[kk],p_vec[kk]) @ randRot.conj().transpose())
        for jj in range(0, num_outcomes):
            # We compute the ideal probabilities according to Born's rule
            prob_matrix[jj, kk] = np.trace(list_effects[jj] @ list_states_noisy[kk]) 
            
    # Here we will store the experimental results        
    sample = np.zeros((num_shots, num_inputStates))

    for kk in range(0, num_inputStates):
        # We sample the different experimental outcomes (from module "random")
        sample[:, kk] = np.array(random.choices(range(num_outcomes), weights = prob_matrix[:, kk], k = num_shots))

    # Here we will store the experimental frequencies of the different outcomes    
    frequencies = np.zeros((num_outcomes, num_inputStates))

    for kk in range(0, num_inputStates):
        for jj in range(0, num_outcomes):
            # We compute the frequency with an ad hoc function (from module "operator")
            frequencies[jj, kk] = countOf(sample[:, kk], jj)/num_shots

    return frequencies #matrix of frequencies. frequencies[j,k] gives the frequency for the jth effect and kth state


def gen_noisy_states(rhoInput_list,p,rotMagnitude):
    """Function to generate a list of noise input states with both coherent and incoherent noise

    Args:
        rhoInput_list (list): list of dim x dim density matrices for the noiseless input states
        p (float): magnitude of incoherent noise
        rotMagnitude (float): magnitude of coherent

    Returns:
        list of dim x dim noisy input states
    """        
    
    list_states_noisy = []
    dim = len(rhoInput_list[0])
    for kk in range(len(rhoInput_list)):
        randRot = random_rotation(dim,rotMagnitude,rotMagnitude)
        list_states_noisy.append(randRot @ random_noise(rhoInput_list[kk],p) @ randRot.conj().transpose())
        
    return list_states_noisy


def run_SDP_QMT(list_states, frequencies, SDPtype = 'norm', **kwargs):
    """Run a semidefinite program for measurement tomography that computes the experimental effects given the experimental frequencies
of a tomographic experiment

    Args:
        list_states (list): list of dim x dim matrices representing the input states. list_states[j] gives the jth input state
        frequencies (np.ndarray): matrix of frequencies as given by the function generate_frequencies
        SDPtype (str, optional): type of SDP we want to run. The available types are the following:
            -- 'singleDelta': we minimize a single delta for all the effects
            -- 'manyDeltas': we minimize the sum of the differences between ideal and experimental frequencies (many-deltas SDP)
            -- 'norm': equivalent way of writing the many-deltas SDP as a norm minimization

    Returns:
        tuple: the value of the objective function and the list of reconstructed effects
    """

    num_inputStates = len(list_states) # number of input states
    num_outcomes = len(frequencies) # number of possible outcomes
    dim = len(list_states[0]) # dimension of the system Hilbert space
 
    Pi_list = {} # list of effects that are variables in cvxpy
    for jj in range(num_outcomes):
        Pi_list[jj] = cp.Variable((dim, dim), hermitian = True)        
    
    # the sum of the effects gives the identity
    constraints_list = [np.sum([Pi_list[jj] for jj in range(num_outcomes)]) == np.eye(dim)]

    if SDPtype == 'singleDelta':    
        
        delta = cp.Variable(nonneg = True) # variable we minimize
        obj = cp.Minimize(delta)

        for jj in range(0, num_outcomes):
            constraints_list.append(Pi_list[jj]>>0)  # the effects are positive
            for kk in range(0, num_inputStates):
                constraints = [     
                    # inequalities for the SDP
                    cp.real(cp.trace(Pi_list[jj] @ list_states[kk])) <= (frequencies[jj, kk] + delta),
                    cp.real(cp.trace(Pi_list[jj] @ list_states[kk])) >= (frequencies[jj, kk] - delta)
                ]
                constraints_list += constraints
                
    elif SDPtype == 'manyDeltas':

        deltamatrix = cp.Variable((num_outcomes, num_inputStates)) # matrix of deltas we minimized

        obj = cp.Minimize(cp.sum(deltamatrix))

        for jj in range(0, num_outcomes):
            constraints_list.append(Pi_list[jj]>>0)
            for kk in range(0, num_inputStates):
                constraints = [     
                    deltamatrix[jj,kk]>=0, # each delta must be positive
                    cp.real(cp.trace(Pi_list[jj] @ list_states[kk])) <= (frequencies[jj, kk] + deltamatrix[jj,kk]),
                    cp.real(cp.trace(Pi_list[jj] @ list_states[kk])) >= (frequencies[jj, kk] - deltamatrix[jj,kk])
                ]
                constraints_list += constraints

    elif SDPtype == 'norm':

        norm = 0 # norm we minimize
        for jj in range(0, num_outcomes):
            constraints_list.append(Pi_list[jj]>>0)
            for kk in range(0, num_inputStates):
                norm += cp.abs(cp.trace(Pi_list[jj] @ list_states[kk])-frequencies[jj, kk])
        obj = cp.Minimize(norm)

               
    problem = cp.Problem(obj, constraints_list)

    problem.solve(**kwargs)

    vector_effects = [] #vector of the effects as an output
    for jj in range(num_outcomes):
        vector_effects.append(Pi_list[jj].value)

    return problem.value, vector_effects


def run_SDP_QST(list_effects,frequencies,SDPtype = 'norm'):
    """Run a semidefinite program for state tomography that computes the experimental effects given the experimental frequencies
of a tomographic experiment

    Args:
        list_effects (list): list of dim x dim matrices representing the POVM effects. list_effects[j] gives the jth effect
        frequencies (np.ndarray): matrix of frequencies as given by the function generate_frequencies
        SDPtype (str, optional): type of SDP we want to run. The available types are the following:
            -- 'singleDelta': we minimize a single delta for all the effects
            -- 'manyDeltas': we minimize the sum of the differences between ideal and experimental frequencies (many-deltas SDP)
            -- 'norm': equivalent way of writing the many-deltas SDP as a norm minimization

    Returns:
        tuple: the value of the objective function and the list of reconstructed states
    """    
    num_outcomes = len(list_effects) # number of possible outcomes
    dim = len(list_effects[0]) # dimension of the system Hilbert space
    num_inputStates = len(frequencies.transpose())
 
    rho_list = {} # list of states that are variables in cvxpy
    for jj in range(num_inputStates):
        rho_list[jj] = cp.Variable((dim, dim), hermitian = True)        

    constraints_list = []
    
    if SDPtype == 'singleDelta':    
        
        delta = cp.Variable(nonneg = True) # variable we minimize
        obj = cp.Minimize(delta)

        
        for kk in range(0, num_inputStates):
            constraints_list +=  [     
                    rho_list[kk]>>0, # the states are positive
                    cp.trace(rho_list[kk])==1,
                ]
            for jj in range(0, num_outcomes):
                constraints = [    
                    # inequalities for the SDP
                    cp.real(cp.trace(rho_list[kk] @ list_effects[jj])) <= (frequencies[jj, kk] + delta),
                    cp.real(cp.trace(rho_list[kk] @ list_effects[jj])) >= (frequencies[jj, kk] - delta)
                ]
                constraints_list += constraints
                
    elif SDPtype == 'manyDeltas':
        
        deltamatrix = cp.Variable((num_outcomes, num_inputStates)) # matrix of deltas we minimized
        
        obj = cp.Minimize(cp.sum(deltamatrix))

        for kk in range(0, num_inputStates):
            constraints_list += [     
                    rho_list[kk]>>0, # the states are positive
                    cp.trace(rho_list[kk])==1,
                ]
            for jj in range(0, num_outcomes):
                constraints = [     
                    deltamatrix[jj,kk]>=0, # each delta must be positive
                    cp.real(cp.trace(rho_list[kk] @ list_effects[jj])) <= (frequencies[jj, kk] + deltamatrix[jj,kk]),
                    cp.real(cp.trace(rho_list[kk] @ list_effects[jj])) >= (frequencies[jj, kk] - deltamatrix[jj,kk])
                ]
                constraints_list += constraints
                
    elif SDPtype == 'norm':

        norm = 0 # norm we minimize
        for jj in range(0, num_outcomes):
            for kk in range(0, num_inputStates):
                norm += cp.abs(cp.trace(rho_list[kk] @ list_effects[jj])-frequencies[jj, kk])
        obj = cp.Minimize(norm)

        for kk in range(0, num_inputStates):
            constraints = [     
                rho_list[kk]>>0, # the states are positive
                cp.trace(rho_list[kk])==1
            ]
            constraints_list += constraints

    problem = cp.Problem(obj, constraints_list)

    problem.solve()

    vector_states = [] #vector of the effects as an output
    for jj in range(num_inputStates):
        vector_states.append(rho_list[jj].value)

    return [problem.value,vector_states]

def run_seesaw(list_states_input,frequencies,nu_delta, SDPtype = 'singleDelta'):
    """Run the see-saw method, without the cross-validation method for finite-shot overfitting, given a list of initial input states (initial guess) and a matrix of experimental frequencies to obtain a final set of reconstructed effects and reconstructed input states

    Args:
        list_states_input (list): list of dim x dim matrices representing the initial input states. list_states_input[j] gives the jth input state
        frequencies (np.ndarray): matrix of frequencies as given by the function generate_frequencies
        nu_delta (float): we stop seesaw if delta(final step)-delta(previous step)<nu_delta (see paper for further details)
        SDPtype (str, optional): type of SDP we want to run (see the function run_SDP_QMT for details)
            
    Returns:
        tuple: list of reconstructed effects, list of reconstructed input states, value of the SDP delta at the final seesaw step
    """
        
    list_states = list_states_input
    old_delta = 1
    new_delta = 0.5
        
    while np.abs(old_delta-new_delta)>nu_delta:
        old_delta = new_delta
        temp_var = run_SDP_QMT(list_states,frequencies,SDPtype=SDPtype)
        new_delta = temp_var[0]
        list_effects = temp_var[1]
        #print(new_delta)
        if np.abs(old_delta-new_delta)<nu_delta:
            return [list_effects,list_states,new_delta] 
        old_delta = new_delta
        temp_var = run_SDP_QST(list_effects,frequencies,SDPtype=SDPtype)
        new_delta = temp_var[0]
        #print(new_delta)
        list_states = temp_var[1]

    return [list_effects,list_states,new_delta]

def gen_frequencies_list_noisy(num_shots,list_effects,list_states,p,rotMagnitude,num_k=10):
    """Generate a list of sampled matrices of experimental frequencies for the different effects (in list_effects) of a POVM obtained by measuring the states in list_states with num_shots shots and with both incoherent and coherent noise applied to the input states (same noise magnitude over all the input states)

    Args:
        num_shots (int): number of shots per input state
        list_effects (list): list of dim x dim matrices representing the POVM effects
        list_states (list): list of dim x dim density matrices for the input states
        p (float): magnitude of incoherent noise
        rotMagnitude (float): magnitude of coherent
        num_k (int, optional): length of the list of sampled matrices
        
    Returns:
        list of matrices of experimental frequencies
    """
    
    return_list = []
    list_states_noisy = gen_noisy_states(rhoInput_list=list_states,p=p,rotMagnitude=rotMagnitude)
    for jj in range(num_k):
        return_list.append(generate_frequencies(num_shots=num_shots,list_effects=list_effects,list_states=list_states_noisy))
        
    return return_list

def run_seesaw_crossV(list_states_input,frequencies_list,nu_delta,num_sampling=5,num_repetitions = 10,SDPtype = 'singleDelta'):
    """Run the see-saw method, with the cross-validation method for finite-shot overfitting, given a list of initial input states (initial guess) and a list of matrices of experimental frequencies to obtain a final set of reconstructed effects and reconstructed input states. The see-saw procedure is computed using the matrix of experimental frequencies given by the average of all the matrices in the collection. The average distance between left and right subset is computed over different repetitions. In each repetitions, a given subset of matrices in the collection is taken and their average creates the left subset (and analogously for the right one)

    Args:
        list_states_input (list): list of dim x dim matrices representing the initial input states. list_states_input[j] gives the jth input state
        frequencies_list (list of np.ndarray): list of matrices of frequencies as given by the function gen_frequencies_list_noisy
        nu_delta (float): we stop seesaw if delta(final step)-delta(previous step)<nu_delta (see paper for further details)
        num_sampling (int, optional): at each repetition, we sample num_sampling matrices of frequencies in frequencies_list and these will constitute the left subset of the data
        num_repetitions (int, optional): number of repetitions to compute the average distance between left and right subsets
        SDPtype (str, optional): type of SDP we want to run (see the function run_SDP_QMT for details)
            
    Returns:
        tuple: list of reconstructed effects, list of reconstructed input states, value of the SDP delta at the final seesaw step, total number of see-saw steps
    """
        
    list_states = list_states_input
    old_delta = 1
    new_delta = 0.5
       
    dim = len(list_states[0]) # dimension of the system Hilbert space
    num_inputStates = len(list_states) # number of input states
    num_outcomes = len(frequencies_list[0]) # number of possible outcomes
    num_k = len(frequencies_list) # number of subsets into which we devide the total dataset

    av_exp_frequencies = np.mean(frequencies_list,axis=0) # experimental frequencies for the whole subset
    list_distance = [] # distances between A and B experimental frequencies for each partition
    
    for jj in range(num_repetitions): # loop over the different repetitions (different partitions in each repetition)
        
        random_list = random.sample(range(num_k),num_sampling) # we sample num_sampling subsets over the total number of subsets
        
        # we compute the average experimental frequencies for the sampled ("A") subset
        matrix_mean_training = np.mean([x for i,x in enumerate(frequencies_list) if i in random_list],axis=0)
        # we compute the average experimental frequencies for the remaining ("B") subset
        matrix_mean_test = np.mean([x for i,x in enumerate(frequencies_list) if i not in random_list],axis=0)

        temp = np.max(np.abs(matrix_mean_test-matrix_mean_training))
        # distance between the frequencies of A and B
        list_distance.append(temp)
    
    av_distance = np.mean(list_distance)
    
    #print(av_distance)
    
    step_count = 0
    
    # Then we run seesaw with this additional stopping condition
    while np.abs(old_delta-new_delta)>nu_delta:
        step_count +=1
        old_delta = new_delta
        temp_var = run_SDP_QMT(list_states,av_exp_frequencies,SDPtype=SDPtype)
        new_delta = temp_var[0]
        list_effects = temp_var[1]
        #print(new_delta)
        if new_delta<av_distance/2. or np.abs(old_delta-new_delta)<nu_delta:
            return [list_effects,list_states,new_delta,step_count]
        step_count += 1
        old_delta = new_delta
        temp_var = run_SDP_QST(list_effects,av_exp_frequencies,SDPtype=SDPtype)
        new_delta = temp_var[0]
        #print(new_delta)
        list_states = temp_var[1]
        if new_delta<av_distance/2.:
            return [list_effects,list_states,new_delta,step_count] 

    return [list_effects,list_states,new_delta,step_count]

def run_MLE_log(list_states,frequencies):
    """ Run the log maximum likelihood estimation that computes the experimental effects given the experimental frequencies of a  QMT experiment

    Args:    
        list_states (list): list of dim x dim matrices representing the input states. list_states[j] gives the jth input state
        frequencies (np.ndarray): matrix of frequencies as given by the function generate_frequencies

    Returns:
            list of reconstructed effects    
    """
    num_inputStates = len(list_states) # number of input states
    num_outcomes = len(frequencies) # number of possible outcomes
    dim = len(list_states[0]) # dimension of the system Hilbert space

    Pi_list = {} # list of effects that are variables in cvxpy
    for jj in range(num_outcomes):
        Pi_list[jj] = cp.Variable((dim, dim), hermitian = True)        

    # the sum of the effects gives the identity
    constraints_list = [np.sum([Pi_list[jj] for jj in range(num_outcomes)]) == np.eye(dim)]

    logF = 0 # variable we minimize
    for jj in range(0, num_outcomes):
        constraints_list.append(Pi_list[jj] >> 0)
        for kk in range(0, num_inputStates):
            logF += (-1)*frequencies[jj, kk] *cp.log(cp.real(cp.trace(Pi_list[jj] @ list_states[kk])))

    obj = cp.Minimize(logF)

    problem = cp.Problem(obj, constraints_list)

    problem.solve()

    vector_effects = [] #vector of the effects as an output
    for jj in range(num_outcomes):
        vector_effects.append(Pi_list[jj].value)

    return vector_effects

def run_SDP_manyDeltas(list_states, frequencies,  **kwargs):
    """Run a many-deltas SDP that computes the experimental effects given the experimental frequencies
    of a tomographic experiment and stores the final values of each delta[j,k]

    Args:
        list_states (list): list of dim x dim matrices representing the input states. list_states[j] gives the jth input state
        frequencies (np.ndarray): matrix of frequencies as given by the function generate_frequencies
  
    Returns:
        tuple: matrix of final delta[j,k] and the list of reconstructed effects
    """

    num_inputStates = len(list_states) # number of input states
    num_outcomes = len(frequencies) # number of possible outcomes
    dim = len(list_states[0]) # dimension of the system Hilbert space
 
    Pi_list = {} # list of effects that are variables in cvxpy
    for jj in range(num_outcomes):
        Pi_list[jj] = cp.Variable((dim, dim), hermitian = True)        
    
    # the sum of the effects gives the identity
    constraints_list = [np.sum([Pi_list[jj] for jj in range(num_outcomes)]) == np.eye(dim)]

    deltamatrix = cp.Variable((num_outcomes, num_inputStates)) # matrix of deltas we minimized

    obj = cp.Minimize(cp.sum(deltamatrix))

    for jj in range(0, num_outcomes):
        constraints_list.append(Pi_list[jj]>>0)
        for kk in range(0, num_inputStates):
            constraints = [     
                deltamatrix[jj,kk]>=0, # each delta must be positive
                cp.real(cp.trace(Pi_list[jj] @ list_states[kk])) <= (frequencies[jj, kk] + deltamatrix[jj,kk]),
                cp.real(cp.trace(Pi_list[jj] @ list_states[kk])) >= (frequencies[jj, kk] - deltamatrix[jj,kk])
            ]
            constraints_list += constraints


    problem = cp.Problem(obj, constraints_list)

    problem.solve(**kwargs)

    vector_effects = [] #vector of the effects as an output
    for jj in range(num_outcomes):
        vector_effects.append(Pi_list[jj].value)

    return deltamatrix.value, vector_effects


        
def anRes(frequencies_matrix,list_states,which_method='SDP',SDPtype = 'norm', **kwargs):
    """
     Postprocess a matrix of experimental frequencies through either SDP or log MLE to obtain a set of reconstruct effects and the time it takes for running the fitting method

    Args:
        list_states (list): list of dim x dim matrices representing the input states. list_states[j] gives the jth input state
        frequencies_matrix (np.ndarray): matrix of frequencies as given by the function generate_frequencies
        which_method (str, optional): fitting method. The available methods are:
            -- 'SDP': semidefinite programming
            -- 'logMLE': log maximum likelihood estimation
        SDPtype (str, optional): type of SDP we want to run (if the method is log MLE this argument does not matter). The available types are the following:
        -- 'singleDelta': we minimize a single delta for all the effects
        -- 'manyDeltas': we minimize the sum of the differences between ideal and experimental frequencies (many-deltas SDP)
        -- 'norm': equivalent way of writing the many-deltas SDP as a norm minimization
            
    Returns:
        tuple: the runtime value of the chosen fitting methods and the list of reconstructed effects
    """
    if which_method == 'SDP':
        starttime = timeit.default_timer()
        vector_effects = run_SDP_QMT(list_states=list_states,frequencies=frequencies_matrix, SDPtype = SDPtype, **kwargs)[1]
        stoptime = timeit.default_timer()
    elif which_method == 'logMLE':
        starttime = timeit.default_timer()
        vector_effects = run_MLE_log(list_states=list_states,frequencies=frequencies_matrix)
        stoptime = timeit.default_timer()

    return  stoptime-starttime, vector_effects


def random_experiment_methods(dim,num_shots,num_states,num_exps, **kwargs):
    """Run different quantum tomographic experiments with random input states and random POVM effects and compute the average runtime and the average trace distance between ideal and reconstructed effects over the different experiments, for different fitting methods

    Args:
        dim (int): dimension of the Hilbert space of the system
        num_shots (int): number of shots per input state
        num_states (int): number of random input states
        num_exps (int): number of different experiments over which we compute the mean values
            
    Returns:
        dictionary: it stores the average runtime values and the average trace distance between ideal and reconstructed effects
        for the different methods (log MLE, single-delta SDP and norm (many-delta) SDP)"""
    
    random_states = np.zeros((num_states, dim, dim), dtype=complex)
     
    resDict = {}
   
    tracedist_matrix_Dict = {}
    time_vector_Dict = {}
    
    for keyMethod in ['singleDelta','norm','logMLE']:
        tracedist_matrix_Dict[keyMethod] = np.zeros((num_exps,dim**2))  # we store here the trace distances
        time_vector_Dict[keyMethod] = np.zeros(num_exps) # we store here the runtimes

    for ll in range(num_exps):
    
        while np.linalg.matrix_rank(np.column_stack([np.ndarray.flatten(random_states[ll,:,:]) for ll in range(num_states)]))<dim**2: # we want a set of input states that forms a basis in the space of system operators
            for i in range(0, num_states):
                random_states[i] = rand_dm_ginibre(dim) # we generate random input states

        rand_map = qtrandom.rand_kraus_map(dim) # we generate random POVM effects 
        listKraus = []
        listEffects = []
        for jj in range(len(rand_map)):
            listKraus.append(rand_map[jj].full())
            listEffects.append(listKraus[jj].conj().T@listKraus[jj])

        frequencies_matrix = generate_frequencies(num_shots=num_shots,list_effects=listEffects,list_states=random_states)
        # we simulate a matrix of experimental outcome frequencies 
        
        # here we apply the different fitting methods
        [time_vector_Dict['logMLE'][ll],vector_effects_MLE] = anRes(frequencies_matrix=frequencies_matrix,list_states=random_states,which_method='logMLE',SDPtype = 'singleDelta')
        [time_vector_Dict['singleDelta'][ll],vector_effects_sD] = anRes(frequencies_matrix=frequencies_matrix,list_states=random_states,which_method='SDP',SDPtype = 'singleDelta')
        [time_vector_Dict['norm'][ll],vector_effects_norm] = anRes(frequencies_matrix=frequencies_matrix,list_states=random_states,which_method='SDP',SDPtype = 'norm')
     
        for jj in range(dim**2):
            # trace distance between the estimated vector_effects and the ideal list_effects
            tracedist_matrix_Dict['logMLE'][ll,jj] =  trace_distance(vector_effects_MLE[jj], listEffects[jj])
            tracedist_matrix_Dict['singleDelta'][ll,jj] =  trace_distance(vector_effects_sD[jj], listEffects[jj])
            tracedist_matrix_Dict['norm'][ll,jj] =  trace_distance(vector_effects_norm[jj], listEffects[jj])
            
        for keyMethod in ['singleDelta','norm','logMLE']:
            resDict[keyMethod] = {}
            resDict[keyMethod]['vecMean'] = np.mean(np.mean(tracedist_matrix_Dict[keyMethod],axis = 1)) # average trace distance
            resDict[keyMethod]['vecStd'] = np.std(np.mean(tracedist_matrix_Dict[keyMethod],axis = 1))
            resDict[keyMethod]['timeMean'] = np.mean(time_vector_Dict[keyMethod]) # average runtime
            resDict[keyMethod]['timeStd'] = np.std(time_vector_Dict[keyMethod])    
            
    return resDict


