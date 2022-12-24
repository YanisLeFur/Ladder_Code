import csv
import pathlib
import time
from dataclasses import dataclass
import string
from attr import frozen
from regex import E
import stim
import numpy as np
import pymatching
import networkx as nx
from typing import Callable, List, Dict, Any, Set, FrozenSet, Iterable, Tuple
import math
import matplotlib.pyplot as plt
import sinter
import sys
from tqdm.auto import tqdm


#periodic Boundary conditions
def loop(c: complex, *, distance:int)->complex:
    r = c.real %(distance*4) 
    i =c.imag 
    return r+i*1j


@dataclass
class EdgeType:
    pauli: str
    size_check: complex 
    lad_to_qubit_delta: complex

edge_types = [
        EdgeType(pauli = "Z",size_check=2j,lad_to_qubit_delta= -1 - 1j ),
        EdgeType(pauli = "X",size_check=2,lad_to_qubit_delta= -1 - 1j),
        EdgeType(pauli = "Y",size_check=2,lad_to_qubit_delta= -1 - 1j)
    
    ]

edges_around_lad: list[tuple[complex,complex]] = [
        (-1-1j,-1+1j),
        (+1-1j,+1+1j),
        (-1-1j,+1-1j),
        (-1+1j,+1+1j)
    ]

def sorted_complex(xs)->list[complex]:
    return sorted(xs, key=lambda v: (v.real,v.imag))

#traduction of the number of round to the pauli matrix checked
def condition_round(r)->string:
    condition ="NONE"
    if r%2==0:
        condition ="Z"
    if r%4==1:
        condition ="X"
    if r%4==3:
        condition ="Y"
    return condition

def generate_circuit_cycle(*,
                           q2i: dict[complex,int],
                           before_parity_measure_2q_depolarization: float,
                           before_round_1q_depolarization: float,
                           before_cycle_1q_depolarization: float,
                           lad_centers: dict[complex,int],
                           distance: int,
                           detectors: bool) -> stim.Circuit:

    #initializing the stable circuit
    round_circuit=[]
    edge_groups = {"X": [], "Y": [], "Z": []} 
    edge_type_condition =[0,1,0,2]

    #Detector variables 
    measurement_time: dict[frozenset[int], int] ={} 
    current_time = 0
    measurement_per_round: int 
    r_value =  ['Z1','X','Z2','Y'] #have a different time between the two Z checks measurement
    for r in range(4):
        if r==3:
            relevant_lads =[h for h, category in lad_centers.items() if category == 1 ]
        else:
            relevant_lads =[h for h, category in lad_centers.items() if category == 0 ]
        for h in relevant_lads:
            for sign in [+1,-1]:
                    q1 = loop(h + edge_types[edge_type_condition[r]].lad_to_qubit_delta*sign, distance=distance)
                    q2 = loop(h + (edge_types[edge_type_condition[r]].lad_to_qubit_delta + edge_types[edge_type_condition[r]].size_check)*sign, distance = distance)
                    if r!=2:
                        edge_groups[condition_round(r)].append(frozenset([q1,q2]))

        circuit = stim.Circuit()
         #Turn parity observables into single qubit observable
        pair_target =[q2i[q] for pair in edge_groups[condition_round(r)] for q in sorted_complex(pair)]
        
        center_length = len(lad_centers.items())
        ancilla =  np.asarray(np.linspace(2*center_length+1,3*center_length,center_length),dtype = 'int')
        if before_round_1q_depolarization>0:
            circuit.append_operation("X_ERROR",ancilla,before_round_1q_depolarization) 

        #Make all the parity operation Z basis parities
        if r==1:
            circuit.append_operation("H", pair_target)
        if r ==3:
            circuit.append_operation("H_YZ", pair_target)

        if before_parity_measure_2q_depolarization>0:
           circuit.append_operation("DEPOLARIZE2",pair_target,before_parity_measure_2q_depolarization)

        circuit.append_operation("CNOT", pair_target)

        #detector search measurement time 
        for k in range(0,len(pair_target),2):
            edge_key = frozenset([pair_target[k],pair_target[k+1],r_value[r]]) 
            measurement_time[edge_key] = current_time 
            current_time+=1
        
        

        #Measure with ancilla qubit
        new_pair =  [None]*(len(pair_target[1::2])+len(ancilla))
        new_pair[::2]=pair_target[1::2]
        new_pair[1::2]=ancilla
        circuit.append_operation("CNOT",new_pair)
        circuit.append_operation("MR", ancilla)

        circuit.append_operation("CNOT", pair_target)
        if r ==3:
            circuit.append_operation("H_YZ", pair_target)
        if r==1:
            circuit.append_operation("H", pair_target)

        #multiply relevant measurement into the observable
        included_measurement =[]
        condition_observable = "NONE"
        for pair in edge_groups["Z"]:
            a,b =pair
            if r<2:
                condition_observable = "Z1"
            else:
                condition_observable = "Z2"
            edge_key = frozenset([q2i[a],q2i[b],condition_observable])
            included_measurement.append(stim.target_rec(measurement_time[edge_key]-current_time))
        circuit.append_operation("OBSERVABLE_INCLUDE", included_measurement,0)
       
        round_circuit.append(circuit)
    measurement_per_cycle = current_time
    measurement_per_round = measurement_per_cycle//4

    # Generate the detector circuit we will create  4 different detector circuit 
    if detectors:
        key_condition = [[1,0,0,1],
                        ['Z1','Z1','Z2','Z2'],
                        ['Z1','Z1','Z2','Z2'],
                        ['Y','X','X','Y'],
                        ['Y','X','X','Y']
                        ]
        for r in range(4):
            circuit = stim.Circuit()
            end_time= (r+1)*measurement_per_round
            relevant_lads =[h for h, category in lad_centers.items() if category == key_condition[0][r] ]
            for h in relevant_lads:   
                record_targets =[]
                count_edge =0
      
                for a,b in edges_around_lad:
                    q1 = loop(h+a,distance =distance)
                    q2 = loop(h+b,distance =distance)
                    key = frozenset([q2i[q1],q2i[q2],key_condition[count_edge+1][r]])
                    relative_index = (measurement_time[key]-end_time)%measurement_per_cycle - measurement_per_cycle
                    if count_edge<2:
                        old_key = frozenset([q2i[q1],q2i[q2],key_condition[count_edge+1][r-2]])
                        
                        old_relative_index = (measurement_time[old_key]-end_time)%measurement_per_cycle - measurement_per_cycle
                        if r%2==0:
                            old_relative_index-=2*measurement_per_round
                    else:   
                        old_relative_index = relative_index - measurement_per_cycle          
                    record_targets.append(stim.target_rec(relative_index))  
                    record_targets.append(stim.target_rec(old_relative_index))
                    count_edge+=1                                          
                circuit.append_operation("DETECTOR", record_targets, [h.real, h.imag, 0])
            circuit.append_operation("SHIFT_COORDS", [], [0, 0, 1])
            
            round_circuit[r] += circuit

    full_circuit = stim.Circuit()
    if before_cycle_1q_depolarization > 0:
       full_circuit.append_operation("DEPOLARIZE1", sorted(q2i.values()), before_cycle_1q_depolarization)
    full_circuit += round_circuit[0] + round_circuit[1] + round_circuit[2] + round_circuit[3]

    return full_circuit



#Generate a circuit with noise (Y/N) and detector (Y/N) also implement the qubits coordinatesand the logical operators
def generate_circuit(distance: int, cycles: int,
                     before_parity_measure_2q_depolarization: float,
                     before_round_1q_depolarization: float,
                     before_cycle_1q_depolarization: float,
                     start_of_all_noisy_cycles_1q_depolarization: float,
                     ) -> stim.Circuit:
    
    lad_centers : dict[complex,int] = {}
    for row in range(2*distance):
        center = 1+row*2 + 1j
        category = row%2
        lad_centers[loop(center,distance=distance)]=category
    qubit_coordinates = set()
    for h in lad_centers:
            for sign in [-1,+1]:
                q = h+edge_types[0].lad_to_qubit_delta*sign
                qubit_coordinates.add(loop(q,distance = distance))
    q2i: dict[complex, int] = {q: i for i, q in enumerate(sorted_complex(qubit_coordinates))}

    #generation of the different type of circuit
    round_circuit_no_noise_no_detectors = generate_circuit_cycle(
        q2i=q2i,
        before_parity_measure_2q_depolarization=0,
        before_round_1q_depolarization=0,
        before_cycle_1q_depolarization=0,
        lad_centers=lad_centers,
        distance=distance,
        detectors=False,
    )
    round_circuit_no_noise_yes_detectors = generate_circuit_cycle(
        q2i=q2i,
        before_parity_measure_2q_depolarization=0,
        before_round_1q_depolarization=0,
        before_cycle_1q_depolarization=0,
        lad_centers=lad_centers,
        distance=distance,
        detectors=True,
    )
    round_circuit_yes_noise_yes_detectors = generate_circuit_cycle(
        q2i=q2i,
        before_parity_measure_2q_depolarization=before_parity_measure_2q_depolarization,
        before_round_1q_depolarization=before_round_1q_depolarization,
        before_cycle_1q_depolarization=before_cycle_1q_depolarization,
        lad_centers=lad_centers,
        distance=distance,
        detectors=True,
    )
    

    full_circuit = stim.Circuit()
    for q,i in q2i.items():
        full_circuit.append_operation("QUBIT_COORDS", [i], [q.real,q.imag])
    
    #Initialize data qubits along logical observables leg into correct basis for observable to be deterministic
    qubits_top_leg = sorted([q for q in qubit_coordinates if q.imag ==2], key = lambda v: v.real)

    full_circuit += (
             round_circuit_no_noise_no_detectors * 2
             + round_circuit_no_noise_yes_detectors * 0
    )
    #full_circuit.append_operation("DEPOLARIZE2",
    #                            sorted(q2i.values()),
    #                            start_of_all_noisy_cycles_1q_depolarization)

    full_circuit += (
            round_circuit_yes_noise_yes_detectors * cycles
            + round_circuit_no_noise_yes_detectors * 0
            + round_circuit_no_noise_no_detectors * 0
    )

    #finish circuit with data measurement
    qubits_coords_to_measure = [q for q in qubits_top_leg]
    qubits_indices_to_measure =[q2i[q] for q in qubits_coords_to_measure]
    order = {q: i for i,q in enumerate(qubits_coords_to_measure)}
    full_circuit.append_operation("M",qubits_indices_to_measure)
    full_circuit.append_operation("OBSERVABLE_INCLUDE",
                               [stim.target_rec(i-len(qubits_indices_to_measure)) for i in order.values()],
                               0)
    return full_circuit












'''THRESHOLD COMPUTATION================================'''

# Generates surface code circuit tasks using Stim's circuit generation.
def generate_example_tasks():
    for p in [0.05, 0.08, 0.1]:
        for d in [1,2]:
            yield sinter.Task(
                circuit = generate_circuit(distance=d, cycles=2,
                     before_parity_measure_2q_depolarization=0,
                     before_round_1q_depolarization=p,
                     before_cycle_1q_depolarization=0,
                     start_of_all_noisy_cycles_1q_depolarization=0,
                     ),
                json_metadata={
                    'p': p,
                    'd': d,
                },
            )

def main():
    d = 2
    cycles = 2
    p = 0.01
    circuit = generate_circuit(distance=d, cycles=cycles,
                     before_parity_measure_2q_depolarization=0,
                     before_round_1q_depolarization=p,
                     before_cycle_1q_depolarization=0,
                     start_of_all_noisy_cycles_1q_depolarization=0,
                     )
    print(circuit)
    print(circuit.detector_error_model(flatten_loops=True))

    # Collect the samples (takes a few minutes).
    samples = sinter.collect(
        num_workers=4,
        max_shots=1_000_000,
        max_errors=1000,
        tasks=generate_example_tasks(),
        decoders=['pymatching'],
    )

    # Print samples as CSV data.
    print(sinter.CSV_HEADER)
    for sample in samples:
        print(sample.to_csv_line())

    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        x_func=lambda stats: stats.json_metadata['p'],
        group_func=lambda stats: stats.json_metadata['d'],
    )
    ax.set_ylim(1e-4, 1e-0)
    ax.set_xlim(5e-2, 5e-1)
    ax.loglog()
    ax.set_title("Repetition Code Error Rates (Phenomenological Noise)")
    ax.set_xlabel("Phyical Error Rate")
    ax.set_ylabel("Logical Error Rate per Shot")
    ax.grid(which='major')
    ax.grid(which='minor')
    ax.legend()
    fig.set_dpi(120)

    # Save to file and also open in a window.
    fig.savefig('plot.png')
    plt.show()


# NOTE: This is actually necessary! If the code inside 'main()' was at the
# module level, the multiprocessing children spawned by sinter.collect would
# also attempt to run that code.
if __name__ == '__main__':
    main()