import numpy as np
import matplotlib.pyplot as plt
import qiskit as qk
import qutip as qt
from TSPviaGrover import *
qk.IBMQ.load_account()

# ### Setting the problem

# In[3]:

def ham_cicles_via_grover_ibmq(num_cities):
    pi = np.pi
    num_paths = np.math.factorial(num_cities - 1)
    num_qbits_per_city = int(np.ceil(np.log2(num_cities)))
    num_qbits_total = num_cities*num_qbits_per_city

    Q_alg = quantum_operators_hamiltonian_cicles_via_grover(num_cities)
    Oracle = Q_alg.hamiltonian_cicles_oracle_projector()
    Hadamard_op = qt.qip.operations.hadamard_transform(num_qbits_total)

    N = num_cities**num_cities
    M = num_paths
    num_aplic = int((pi/4)*np.sqrt(N/M))

    Grover_op = Q_alg.grover_operator(Oracle, num_aplic) 

    Hadamard_gate = qk.extensions.UnitaryGate(Hadamard_op,
                                              label = 'Hadamard'
                                             )
    Grover_gate = qk.extensions.UnitaryGate(Grover_op,
                                            label = 'Grover^'+str(num_aplic)
                                           )

    circuit = qk.QuantumCircuit()
    qargs = [i for i in range(num_qbits_total)]
    cargs = qargs

    for i in range(0,num_cities):
        circuit.add_register(qk.QuantumRegister( num_qbits_per_city,'q'+str(i) ),
                             qk.ClassicalRegister( num_qbits_per_city,'c'+str(i) )
                            )

    circuit.append(Hadamard_gate,
                   qargs
                  )

    circuit.append(Grover_gate,
                   qargs
                  )

    circuit.barrier(
                  )

    circuit.measure(qargs,
                    cargs
                    )

    # Run the quantum circuit on a statevector simulator backend
    simulator = qk.Aer.get_backend('qasm_simulator')
    job = qk.execute(circuit, simulator, shots = 1000)
    result = job.result()
    count = result.get_counts(circuit)
    plot = qk.visualization.plot_histogram(count)
    #count
    
    return circuit.draw(),plot.show()



