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
    i =c.imag #the imaginary number will always stay in the unit cell
    return r+i*1j


@dataclass
class EdgeType:
    pauli: str
    size_check: complex #size of the U-U check (U=X,Y,Z)
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
        pair_target = []
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
            #print("KEY",edge_key)
            measurement_time[edge_key] = current_time 
            current_time+=1
  

        #Measure
        circuit.append_operation("M", pair_target[1::2])
        #print("TARGET",pair_target[1::2])
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
        measurement_per_round = len(pair_target)//2
    measurement_per_cycle = 4*measurement_per_round

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


def compute_treshold (probabilities: list[float], distances: list[int]):
    with open("data.csv","w") as f:
        num_shots = 10000
        print("distance,number shots,number correct", file=f)
        print("distances", distances)
        print("probabilities",probabilities)
        print("num_shots",num_shots)
        for p in probabilities:
            print(f"physical error rate {p}: ", end="")
            for d in distances:
                
                circuit = generate_circuit(distance=d, cycles = 5,
                               before_parity_measure_2q_depolarization =0 ,
                               before_round_1q_depolarization = 0,
                               before_cycle_1q_depolarization =p ,
                               start_of_all_noisy_cycles_1q_depolarization = p)
                model = circuit.detector_error_model(decompose_errors = True,flatten_loops=True) 
                matching = pymatching.Matching.from_detector_error_model(model)
                nb_errors = num_errors(circuit,num_shots = num_shots)
                logical_error_rate = (nb_errors)/num_shots
                print(f"{nb_errors}",end=" ")
                print(f"{d},{p},{num_shots},{nb_errors}",file=f)
            print("\n")
    pass


def num_errors(circuit: stim.Circuit, num_shots: int):

    model = circuit.detector_error_model(decompose_errors = True,flatten_loops=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circuit.compile_detector_sampler()
    syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)
    num_errors = 0
    for i in range(syndrome.shape[0]):
        predicted_observables = matching.decode(syndrome[i, :])
        num_errors +=  not np.array_equal(actual_observables[i, :], predicted_observables)

    return num_errors



def main():

    distance = 1
    rounds =50
    circuit = generate_circuit(distance=distance, cycles = rounds,
                               before_parity_measure_2q_depolarization =0.001 ,
                               before_round_1q_depolarization = 0.001,
                               before_cycle_1q_depolarization =0.001 ,
                               start_of_all_noisy_cycles_1q_depolarization = 0.001)

    compute_treshold(np.linspace(0.001,0.009,9),[1,2,3,4,5])

    #samples = circuit.compile_detector_sampler().sample(10)
    #for sample in samples:
    #    print("".join("_1"[e] for e in sample))
    #print(logical_error(circuit,10000))






'''
def test_decoding(circuit: stim.Circuit, num_shots: int):
    m = detector_error_model_to_matching(circuit.detector_error_model())

    detector_samples = circuit.compile_detector_sampler().sample(num_shots, append_observables=True)
    num_correct = 0
    for sample in detector_samples:
        actual_observable = sample[-1]
        detectors_only = sample[:-1]
        predicted_observable = m.decode(detectors_only)[0]
        num_correct += actual_observable == predicted_observable
    print("shots", num_shots)
    print("correct predictions", num_correct)
    print("logical error rate", (num_shots - num_correct) / num_shots)


def detector_error_model_to_matching(model: stim.DetectorErrorModel) -> pymatching.Matching:
    det_offset = 0

    def _iter_model(m: stim.DetectorErrorModel, reps: int, callback: Callable[[float, List[int], List[int]], None]):
        nonlocal det_offset
        for _ in range(reps):
            for instruction in m:
                if isinstance(instruction, stim.DemRepeatBlock):
                    _iter_model(instruction.body_copy(), instruction.repeat_count, callback)
                elif isinstance(instruction, stim.DemInstruction):
                    if instruction.type == "error":
                        dets = []
                        frames = []
                        for t in instruction.targets_copy():
                            v = str(t)
                            if v.startswith("D"):
                                dets.append(int(v[1:]) + det_offset)
                            elif v.startswith("L"):
                                frames.append(int(v[1:]))
                            else:
                                raise NotImplementedError()
                        p = instruction.args_copy()[0]
                        callback(p, dets, frames)
                    elif instruction.type == "shift_detectors":
                        det_offset += instruction.targets_copy()[0]
                    elif instruction.type == "detector":
                        pass
                    elif instruction.type == "logical_observable":
                        pass
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()

    g = nx.Graph()
    num_detectors = model.num_detectors
    for k in range(num_detectors):
        g.add_node(k)
    g.add_node(num_detectors, is_boundary=True)

    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 1:
            dets.append(num_detectors)
        if len(dets) != 2:
            return  # Just ignore correlated error mechanisms (e.g. Y errors / XX errors)
        g.add_edge(*dets, weight=-math.log(p), qubit_id=frame_changes)

    _iter_model(model, 1, handle_error)

    return pymatching.Matching(g)

'''


if __name__ =='__main__':
    main()