from dataclasses import dataclass
import string
from attr import frozen
from regex import E
import stim
import numpy as np

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

def generate_circuit(distance: int, rounds: int)->stim.Circuit:
    
    #we define centers as complex even though they will only be real number
    #it will be easier for us to describe the qubit emplacements (complex numbers)
    lad_centers : dict[complex,int] = {}

    #the unit cell only do XX-YY (2 lad_centers)
    for row in range(2*distance):
        center = 1+row*2 + 1j
        category = row%2
        lad_centers[loop(center,distance=distance)]=category

    #We assume here that we don't know yet if we are in center type 0,1 so edge_type X and Y are the 
    #we will have to define them after by checking if we are in center 0 or 1
    edge_types = [
        EdgeType(pauli = "Z",size_check=2j,lad_to_qubit_delta= -1 - 1j ),
        EdgeType(pauli = "X",size_check=2,lad_to_qubit_delta= -1 - 1j),
        EdgeType(pauli = "Y",size_check=2,lad_to_qubit_delta= -1 - 1j)
    
    ]

    # here we will already have to correct the switch between X and Y otherwise we would add two qubit in 
    # the same emplacement

    qubit_coordinates = set()

    for h in lad_centers:
            for sign in [-1,+1]:
                q = h+edge_types[0].lad_to_qubit_delta*sign
                qubit_coordinates.add(loop(q,distance = distance))


                
#We check that the qubit are in the right place
    fused_dict = dict(lad_centers)
    for q in qubit_coordinates:
        fused_dict[q]="q"
    print_2d(fused_dict)


    #sorting the qubits (ordering)
    q2i: dict[complex, int] = {q: i for i, q in enumerate(
        sorted_complex(qubit_coordinates)
    )}

    edges_around_lad: list[tuple[complex,complex]] = [
        (-1-1j,-1+1j),
        (+1-1j,+1+1j),
        (-1-1j,+1-1j),
        (-1+1j,+1+1j)
    ]

    round_circuit=[]

    measurement_time: dict[frozenset[int], int] ={} 
    current_time = 0
    measurement_per_round: int 

    edge_groups = {"X": [], "Y": [], "Z": []}  


    for r in range(4):

        relevant_lads =[]


        if r<2:
            relevant_lads =[h for h, category in lad_centers.items() if category == 0 ]
        if r>1:
            relevant_lads =[h for h, category in lad_centers.items() if category == 1 ]

        for h in relevant_lads:
            relevant_edge_group=[]
            if r%4 == 0:
                relevant_edge_group = [edge_types[0]]
            if r%2==1:
                if lad_centers[h] == 0:
                    relevant_edge_group = [edge_types[1]]
                if lad_centers[h] == 1:
                    relevant_edge_group = [edge_types[2]]
            for edge_type in relevant_edge_group:
                for sign in [+1,-1]:
                    q1 = loop(h + edge_type.lad_to_qubit_delta*sign, distance=distance)
                    q2 = loop(h + (edge_type.lad_to_qubit_delta + edge_type.size_check)*sign, distance = distance)
                    edge_groups[edge_type.pauli].append(frozenset([q1,q2]))


        circuit = stim.Circuit()


        x_qubits = [q2i[q] for pair in edge_groups["X"] for q in sorted_complex(pair)]
        y_qubits = [q2i[q] for pair in edge_groups["Y"] for q in sorted_complex(pair)]


        #Make all the parity operation Z basis parities
        circuit.append_operation("H", x_qubits)
        circuit.append_operation("H_YZ", y_qubits)

        # Turn parity observables into single qubit observable
        pair_target =[q2i[q] for pair in edge_groups[condition_round(r)] for q in sorted_complex(pair)]
        
        circuit.append_operation("CNOT", pair_target)

        #detector search measurement time 
        for k in range(0,len(pair_target),2):
            edge_key = frozenset([pair_target[k],pair_target[k+1]]) 
            if r%4 ==2 and condition_round(r)=='Z': 
                current_time+=1 
            else:
                measurement_time[edge_key] = current_time 
                current_time+=1
                
        #Measure
        print(condition_round(r),pair_target[1::2])
        circuit.append_operation("M", pair_target[1::2])
        
        #restore qubit bases
        circuit.append_operation("CNOT", pair_target)
        circuit.append_operation("H_YZ", y_qubits)
        circuit.append_operation("H", x_qubits)


        round_circuit.append(circuit)

    measurement_per_round = len(pair_target)//2
    measurement_per_cycle = 4*measurement_per_round



    #===============START DETEC CIRCUIT==================
    #HERE We want to measure first Z type then X type then Z type then Y type
    det_circuit = []



    'PROBLEM WITH R = 3'
    for r in range(4):
        print("========ROUND=========")
        end_time= (r+1)*measurement_per_round
        circuit = stim.Circuit()
        if r==2 or r==1: 
            relevant_lads =[h for h, category in lad_centers.items() if category == 0 ]
        if r==0 or r == 3:
            relevant_lads =[h for h, category in lad_centers.items() if category == 1 ]
        for h in relevant_lads:   
            record_targets =[]
            count_edge =0.
            for a,b in edges_around_lad:
                q1 = loop(h+a,distance =distance)
                q2 = loop(h+b,distance =distance)
                edge_key = frozenset([q2i[q1],q2i[q2]])
                
                if r%4>1 and (count_edge<2):
                    relative_index = (measurement_time[edge_key]+2*measurement_per_round-end_time)%measurement_per_cycle - measurement_per_cycle
                else:
                    relative_index = (measurement_time[edge_key]-end_time)%measurement_per_cycle - measurement_per_cycle
                print(edge_key,relative_index)
                record_targets.append(stim.target_rec(relative_index))  
                record_targets.append(stim.target_rec(relative_index - measurement_per_cycle))
                count_edge+=1                                            
            circuit.append_operation("DETECTOR",record_targets, [h.real, h.imag,0]) 
        det_circuit.append(circuit)

    # ==================END DETECOR CIRCUIT=====================

    full_circuit = stim.Circuit()
    initial_cycle = round_circuit[0]+round_circuit[1]+round_circuit[2]+round_circuit[3]
    stable_cycle = (stim.Circuit()
        + round_circuit[0]+det_circuit[0]
        + round_circuit[1]+det_circuit[1]
        + round_circuit[2]+det_circuit[2]
        + round_circuit[3]+det_circuit[3]
        
    )
    stable_cycle.append_operation("SHIFT_COORDS",[], [0, 0,1])
    for q,i in q2i.items():
        full_circuit.append_operation("QUBIT_COORDS", [i], [q.real,q.imag])
    full_circuit += initial_cycle*10 +stable_cycle *rounds
    return full_circuit



def main():
    circuit = generate_circuit(distance=2,rounds=1)
    print(circuit)
    samples = circuit.compile_detector_sampler().sample(10)
    for sample in samples:
        print("".join("_1"[e] for e in sample))


if __name__ =='__main__':
    main()