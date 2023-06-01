# sc-qmt
A repository with the code for quantum measurement tomography (QMT), quantum state tomography (QST) and self-consistent tomography, following the paper *Self-consistent 
quantum measurement tomography based on semidefinite programming* (Cattaneo et al., preprint arXiv:2212.10262 (2022)).
All the results of the paper can be found and reproduced here.

## Installation
Clone the repository with
```
git clone https://github.com/Algorithmiq/sc-qmt.git
```

## Usage

### Reproducing the plots of the paper
The data for the plots of the paper are stored in the folder `data_folder`. The notebook `Figures.ipynb` can be used to produce and save the plots.

### Using semidefinite programming as a fitting method for QMT
First, let us simulate a QMT experiment by preparing a set of input states given by the Pauli eigenstates `rhoPaulis` for the SIC-POVM with effects given by the list `M`:

```
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
```

The experimental frequencies of the QMT experiment with 10000 shots can be generated through the following function:

```
num_shots = 10000
exp_freq = generate_frequencies(num_shots,M,rhoPaulis)
```

Then, we can process the data using a semidefinite program and we reconstruct the POVM effects:
```
process_results = run_SDP_QMT(rhoPaulis,exp_freq,SDPtype = 'singleDelta') # single-delta SDP
reconstructed_effects = process_results[1] # list of output effects
SDP_delta = process_results[0] # infinite distance between experimental and reconstructed probability distribution
```
### Examples
Some examples of how to use the code for the different tomographic tasks discussed in the paper can be found in the files `script_*.py`. The examples include measurement
tomography with and without noise on the input states and self-consistent tomography through the see-saw method.

## Authors and citation

If you find this code useful, please consider citing the paper *Cattaneo et al., preprint arXiv:2212.10262 (2022)*.

BibTeX record:

```
@article{Cattaneo2022,
   author = {Marco Cattaneo and Elsi-Mari Borrelli and Guillermo García-Pérez and Matteo A. C. Rossi and Zoltán Zimborás and Daniel Cavalcanti},
   month = {12},
   title = {Semidefinite programming for self-consistent quantum measurement tomography},   
   journal = {preprint arXiv:2212.10262},
   url = {http://arxiv.org/abs/2212.10262},
   year = {2022},
}
```




