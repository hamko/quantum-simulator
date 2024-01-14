# Import packages to build circuits

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

num_qubit = 30
depth = 3
theta = Parameter('θ')
phis = ParameterVector('φ', num_qubit)
qc = QuantumCircuit(num_qubit)

# Initialize a state
for q in range(0, num_qubit, 2):
    qc.x(q)

# Build dynamics
for d in range(depth):
    # Even layers
    for even_qubit in range(0, num_qubit, 2):
        qc.cz(even_qubit, even_qubit+1)
        qc.u(theta, 0, -np.pi, even_qubit)
        qc.u(theta, 0, -np.pi, even_qubit+1)
    # Local random phase gates
    for q in range(num_qubit):
        qc.p(phis[q], q)
    # Odd layers
    for odd_qubit in range(1, num_qubit-1, 2):
        qc.cz(odd_qubit, odd_qubit+1)
        qc.u(theta, 0, -np.pi, odd_qubit)
        qc.u(theta, 0, -np.pi, odd_qubit+1)
    # Local random phase gates
    for q in range(num_qubit):
        qc.p(phis[q], q)

# Define measurement observable
obs = SparsePauliOp('I' + 'Z' + 'I' * (num_qubit-2))

# Specify circuit parameter values
np.random.seed(0) # Specify the seed for debugging purpose such that the circuit is the same very time we run it
phi_max = 0.5 * np.pi
parameter_values = [np.random.uniform(-1 * phi_max, phi_max) for _ in range(num_qubit)] + [0.01 * np.pi]

print(qc.draw(output='text'))

