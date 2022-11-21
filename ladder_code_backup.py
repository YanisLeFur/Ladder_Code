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

def print_2d(values: dict[complex, any]):
    assert all(v.real ==int(v.real) for v in values)
    assert all(v.imag ==int(v.imag) for v in values)
    assert all(v.real >=0 and v.imag>=0 for v in values)
    w = int(max((v.real for v in values), default = 0) + 1)
    h = int(max((v.imag for v in values), default = 0) + 1)
    s=""
    for y in range(h):
        for x in range(w):
            s += str(values.get(x+y*1j,"_"))
        s+= "\n"
    print(s)


#periodic BC
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
        x_qubits = [q2i[q] for pair in edge_groups["X"] for q in sorted_complex(pair)]
        y_qubits = [q2i[q] for pair in edge_groups["Y"] for q in sorted_complex(pair)]

        circuit = stim.Circuit()
        if before_round_1q_depolarization>0:
            circuit.append_operation("DEPOLARIZE1",sorted(q2i.values()),before_round_1q_depolarization)
        
        
        #Make all the parity operation Z basis parities
     
        if r==1:
            circuit.append_operation("H", x_qubits)
        if r ==3:
            circuit.append_operation("H_YZ", y_qubits)

        # Turn parity observables into single qubit observable
        pair_target =[q2i[q] for pair in edge_groups[condition_round(r)] for q in sorted_complex(pair)]
        if before_parity_measure_2q_depolarization>0:
            circuit.append_operation("DEPOLARIZE2",pair_target,before_parity_measure_2q_depolarization)
        circuit.append_operation("CNOT", pair_target)

    

        #detector search measurement time 
        for k in range(0,len(pair_target),2):
            edge_key = frozenset([pair_target[k],pair_target[k+1],r_value[r]]) 
            measurement_time[edge_key] = current_time 
            current_time+=1

        #Measure
        circuit.append_operation("M", pair_target[1::2])
    
        #restore qubit bases
        circuit.append_operation("CNOT", pair_target)
        if r ==3:
            circuit.append_operation("H_YZ", y_qubits)
        if r==1:
            circuit.append_operation("H", x_qubits)




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
        det_circuit = []
        key_condition = [[1,0,0,1],
                        ['Z1','Z1','Z2','Z2'],
                        ['Z1','Z1','Z2','Z2'],
                        ['Y','X','X','Y'],
                        ['Y','X','X','Y']
                        ]
        for r in range(4):
            #in order to know the relative measurement time we need to know at what time stop the round r
            end_time= (r+1)*measurement_per_round
            circuit = stim.Circuit()
            relevant_lads =[h for h, category in lad_centers.items() if category == key_condition[0][r] ]
            detect_count =0
            for h in relevant_lads:   
                record_targets =[]
                count_edge =0
                for a,b in edges_around_lad:
                    q1 = loop(h+a,distance =distance)
                    q2 = loop(h+b,distance =distance)
                    key = frozenset([q2i[q1],q2i[q2],key_condition[count_edge+1][r]])
                    relative_index = (measurement_time[key]-end_time)%measurement_per_cycle - measurement_per_cycle
                    #we measure two times Z checks so we have to take the measurement of the previous Z 2 rounds before and not 4 rounds
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
                circuit.append_operation("DETECTOR",record_targets, [h.real, h.imag,0])
            det_circuit.append(circuit)

    full_circuit = stim.Circuit()
    if before_cycle_1q_depolarization>0:
        full_circuit.append_operation("DEPOLARIZE1",sorted(q2i.values()),before_cycle_1q_depolarization)
    full_circuit = round_circuit[0]+round_circuit[1]+round_circuit[2]+round_circuit[3]
   
    return full_circuit





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

    #We check that the qubit are in the right place
    fused_dict = dict(lad_centers)
    for q in qubit_coordinates:
        fused_dict[q]="q"
    #print("\n")
    #print_2d(fused_dict)

    #sorting the qubits (ordering)
    q2i: dict[complex, int] = {q: i for i, q in enumerate(
        sorted_complex(qubit_coordinates)
    )}

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
    qubits_bottom_leg = sorted([q for q in qubit_coordinates if q.imag ==0], key = lambda v: v.real)
    full_circuit += (
                      round_circuit_no_noise_no_detectors * 2
                      + round_circuit_no_noise_yes_detectors * 2
    )
    if start_of_all_noisy_cycles_1q_depolarization>0:
        full_circuit.append_operation("DEPOLARIZE1",
                                      sorted(q2i.values()),
                                      start_of_all_noisy_cycles_1q_depolarization)
    full_circuit += (
            round_circuit_yes_noise_yes_detectors * cycles
            + round_circuit_no_noise_yes_detectors * 2
            + round_circuit_no_noise_no_detectors * 2
    )
    #finish circuit with data measurement
    qubits_coords_to_measure = [q for q in qubits_bottom_leg]
    qubits_indices_to_measure =[q2i[q] for q in qubits_coords_to_measure]
    order = {q: i for i,q in enumerate(qubits_coords_to_measure)}

    full_circuit.append("M",qubits_indices_to_measure)
    full_circuit.append_operation("OBSERVABLE_INCLUDE",
                                [stim.target_rec(i-len(qubits_indices_to_measure)) for i in order.values()],
                                0)
    

    return full_circuit


def sample_error_rates(*,
                       probabilities: List[float],
                       diameter_factor: List[int],
                       append: bool,
                       path: str,
                       shots: int,
                       noisy_cycles: int,
                       before_parity_measure_2q_depolarization_factor: float,
                       before_round_1q_depolarization_factor: float,
                       before_cycle_1q_depolarization_factor: float,
                       start_of_all_noisy_cycles_1q_depolarization_factor: float):
    if not pathlib.Path(path).exists():
        append = False
    with open(path, "a" if append else "w") as f:
        if not append:
            print("distance,physical_error_rate,num_shots,num_correct", file=f)
        print("diameter_factors", diameter_factor)
        print("probabilities", probabilities)
        print("num_shots", shots)
        for p in probabilities:
            s = f"physical error rate {p}:"
            s = s.rjust(50)
            print(s , end="")
            for d in diameter_factor:
                circuit = generate_circuit(
                    distance=d,
                    cycles=noisy_cycles,
                    before_cycle_1q_depolarization=before_cycle_1q_depolarization_factor*p,
                    before_round_1q_depolarization=before_round_1q_depolarization_factor*p,
                    before_parity_measure_2q_depolarization=before_parity_measure_2q_depolarization_factor*p,
                    start_of_all_noisy_cycles_1q_depolarization=start_of_all_noisy_cycles_1q_depolarization_factor*p,
                )
                num_correct = shots-num_errors(
                    num_shots=shots,
                    circuit=circuit,
                )
                print(f" {shots - num_correct}", end="")
                print(f"{d},{p},{shots},{num_correct}", file=f, flush=True)
            print()


def num_errors(circuit: stim.Circuit, num_shots: int):

    model = circuit.detector_error_model(decompose_errors = True) #,flatten_loops=True
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circuit.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)

   
    num_errors = 0
    for i in range(syndrome.shape[0]):
        predicted_observables = matching.decode(syndrome[i, :])
        num_errors +=  not np.array_equal(actual_observables[i, :], predicted_observables)

    return num_errors

@dataclass
class DistanceExperimentData:
    num_shots: int = 0
    num_correct: int = 0

    @property
    def logical_error_rate(self) -> float:
        return (self.num_shots - self.num_correct) / self.num_shots


def round_adjustment(error_rate: float, rounds: int) -> float:
    randomize_rate = min(1, 2*error_rate)
    round_randomize_rate = 1 - (1 - randomize_rate)**(1 / rounds)
    round_error_rate = round_randomize_rate / 2
    return round_error_rate

def plot_data(path: str, title: str, rounds_per_shot: int):
    distance_to_noise_to_results: Dict[int, Dict[float, DistanceExperimentData]] = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            distance = int(row["distance"])
            physical_error_rate = float(row["physical_error_rate"])
            d1 = distance_to_noise_to_results.setdefault(distance, {})
            d2 = d1.setdefault(physical_error_rate, DistanceExperimentData())
            d2.num_shots += int(row["num_shots"])
            d2.num_correct += int(row["num_correct"])

    markers = "_ov*sp^<>12348PhH+xXDd|"
    for distance in sorted(distance_to_noise_to_results.keys()):
        group = distance_to_noise_to_results[distance]
        xs = []
        ys = []
        for physical_error_rate in sorted(group.keys()):
            data = group[physical_error_rate]
            xs.append(physical_error_rate)
            ys.append(round_adjustment(data.logical_error_rate, rounds=rounds_per_shot))
        plt.plot(xs, ys, label=f"diameter_scale_factor={distance}", marker=markers[distance])

    plt.legend()
    plt.loglog()

    def f(p):
        if abs(p * 100 - int(p * 100)) < 1e-5:
            return str(int(p * 100)) + "%"
        r = f"{p:.3%}"
        while r and r[-2:] == "0%":
            r = r[:-2] + "%"
        return r
    ticks_y = [k*10**-p for k in range(1, 10) for p in range(1, 5) if k*10**-p <= 0.5]
    ticks_x = [k*10**-p for k in range(1, 10) for p in range(1, 5) if k*10**-p <= 0.5]
    ticks_x.extend([p/100 for p in range(12, 20, 2)])
    plt.xticks([x for x in ticks_x], labels=[f(x) for x in ticks_x], rotation=45)
    plt.yticks([y for y in ticks_y], labels=[f(y) for y in ticks_y])
    plt.ylim(0.0001, 0.5)
    plt.xlim(0.001, 0.5)
    plt.title(title)
    plt.ylabel("Logical Error Rate (Vertical Observable)")
    plt.xlabel("Physical Error Rate Parameter")
    plt.grid()
    plt.show()


def sample_single_depolarizing_layer_circuit():
    sample_error_rates(
        shots=20000,
        probabilities=[
            0.001,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.11,
            0.12,
        ],
        before_cycle_1q_depolarization_factor=0,
        before_parity_measure_2q_depolarization_factor=0,
        before_round_1q_depolarization_factor=0,
        noisy_cycles=0,
        start_of_all_noisy_cycles_1q_depolarization_factor=1,
        diameter_factor=[1, 2, 3],
        append=False,
        path="data.csv",
    )


def sample_parity_error_circuit():

    sample_error_rates(
        shots=10000,
        probabilities=[
            0.001,
            #0.003,
            ##0.002,
            #0.005,
            #0.007,
            #0.009,
            #0.01,
            #0.015,
            #0.02,
            #0.025,
            #0.03,
            #0.035,
            #0.04,
            #0.045,
            #0.05,
        ],
        before_cycle_1q_depolarization_factor=0,
        before_parity_measure_2q_depolarization_factor=1,
        before_round_1q_depolarization_factor=0,
        noisy_cycles=6,
        start_of_all_noisy_cycles_1q_depolarization_factor=0,
        diameter_factor=[2],
        append=False,
        path="data_from_parity_errors.csv",
    )

def main():
    #sample_single_depolarizing_layer_circuit()
    sample_parity_error_circuit()
    #plot_data("data.csv",
    #          title="LogLog error rates per round for 6 cycle (18 round) toric no-ancilla circuit with 2q depolarization before parity measurements",
    #          rounds_per_shot=18)



    plot_data("data_from_parity_errors.csv",
              title="LogLog error rates per round for 6 cycle (18 round) toric no-ancilla circuit with 2q depolarization before parity measurements",
              rounds_per_shot=18)


if __name__ =='__main__':
    main()