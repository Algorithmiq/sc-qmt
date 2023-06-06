from functions import *

""" The following code produces the data for the comparison between the different SDP methods and log MLE for different dimensions, for 10k number of shots, and for a complete set of input states (Figs. 8 and 9 (upper left))
    """

dim_vector = range(2, 5)  # we explore the dimensions from dim=2 to dim=4

saveDict = []

num_shots = 10000  # number of shots per input state
num_exps = 1  # number of different experiments

for jj in range(len(dim_vector)):

    dim = dim_vector[jj]
    num_states = dim**2  # complete set of input states

    tempSDP = random_experiment_methods(
        dim, num_shots, num_states, num_exps, solver=cp.MOSEK
    )
    saveDict.append(tempSDP)

# Save the results as:
# with open('dim_10k_com.msgpack', "wb") as outfile:
#    packed = msgpack.packb(saveDict)
#    outfile.write(packed)
