from dataclasses import dataclass
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


def generate_circuit(distance: int, rounds: int)->stim.Circuit:
    
    #we define centers as complex even though they will only be real number
    #it will be easier for us to describe the qubit emplacements (complex numbers)
    lad_centers : dict[complex,int] = {}

    #the unit cell only do XX-YY (2 lad_centers)
    for row in range(2*distance):
        center = 1+row*2 + 1j
        category = -row%2
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

    #We need to check which lad_center type we are in order to put the good pauli matrix
    for h in lad_centers:
        for edge_type in edge_types:
            for sign in [-1,+1]:

                if lad_centers[h] == 0 and edge_type.pauli != "Y":
                    q = h + edge_type.lad_to_qubit_delta * sign
                    qubit_coordinates.add(loop(q,distance =distance))
                    
                if lad_centers[h] == 1 and edge_type.pauli != "X":
                    q = h + edge_type.lad_to_qubit_delta * sign
                    qubit_coordinates.add(loop(q,distance =distance))
                    


#We check that the qubit are in the right place
    fused_dict = dict(lad_centers)
    for q in qubit_coordinates:
        fused_dict[q]="q"
    print_2d(fused_dict)


    #sorting the qubits (ordering)
    q2i: dict[complex, int] = {q: i for i, q in enumerate(
        sorted(qubit_coordinates, key =lambda v: (v.real, v.imag))
    )}



    round_circuit=[]
    #r = rounds we begin by 0 type plaquette (Z->X), then type 1 plaquete (Z->Y)
    for r in range(4):
        edge_groups = {"X": [], "Y": [], "Z": []}   
        relevant_lads =[]

        # We want to measure first the checks of type 0 and then the checks of type 1
        if r<2:
            relevant_lads =[h for h, category in lad_centers.items() if category == 0 ]
        if r>1:
            relevant_lads =[h for h, category in lad_centers.items() if category == 1 ]


        for h in relevant_lads:
            
            # We have now four condition if r=0 only Z type if r = 1 only X type ...
            relevant_edge_group=[]
            if r%2 == 0:
                relevant_edge_group = [edge_types[0]]

            if r%2==1:
                if lad_centers[h] == 0:
                    relevant_edge_group = [edge_types[1]]
                      
                if lad_centers[h] == 1:
                    relevant_edge_group = [edge_types[2]]

            for edge_type in relevant_edge_group:
                q1 = loop(h + edge_type.lad_to_qubit_delta, distance=distance)
                q2 = loop(h + edge_type.lad_to_qubit_delta + edge_type.size_check, distance = distance)
                q3 = loop(h - edge_type.lad_to_qubit_delta - edge_type.size_check, distance = distance)
                q4 = loop(h - edge_type.lad_to_qubit_delta, distance = distance)
                edge_groups[edge_type.pauli].append(frozenset([q1,q2]))
                edge_groups[edge_type.pauli].append(frozenset([q3,q4]))



        circuit = stim.Circuit()

        x_qubits = [q2i[q] for pair in edge_groups["X"] for q in sorted_complex(pair)]
        y_qubits = [q2i[q] for pair in edge_groups["Y"] for q in sorted_complex(pair)]

        #Make all the parity operation Z basis parities
        circuit.append_operation("H", x_qubits)
        circuit.append_operation("H_YZ", y_qubits)

        # Turn parity observables into single qubit observable
        pair_target =[q2i[q] for group in edge_groups.values() for pair in group  for q in sorted_complex(pair)]
        circuit.append_operation("CNOT", pair_target)

        #Measure
        circuit.append_operation("M", pair_target[1::2])

        #restaur qubit bases
        circuit.append_operation("H", x_qubits)
        circuit.append_operation("H_YZ", y_qubits)

        round_circuit.append(circuit)
    full_circuit = stim.Circuit()
    cycle = round_circuit[0]+round_circuit[1]+round_circuit[2]+round_circuit[3]
    for q,i in q2i.items():
        full_circuit.append_operation("QUBIT_COORDS", [i], [q.real,q.imag])
    full_circuit += cycle*rounds
    return full_circuit





def main():
    circuit = generate_circuit(distance=3,rounds=6)
    print(circuit)
    samples = circuit.compile_sampler().sample(10)
    for sample in samples:
        print("".join("_1"[e] for e in sample))


if __name__ =='__main__':
    main()