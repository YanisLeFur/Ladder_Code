import stim 
import numpy as np
import shutil
import sinter 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

def rep_code(distance,rounds,noise):
    circuit = stim.Circuit()
    qubits = range(2*distance+1)
    data = qubits[::2]
    measurement = qubits[1::2]
    pairs1 = qubits[:-1]
    pairs2 = qubits[1:][::-1]
    circuit.append_operation("X_ERROR",pairs1,noise)
    circuit.append_operation("CNOT",pairs1)
    #.append_operation("DEPOLARIZE2",pairs1,noise)
    circuit.append_operation("CNOT",pairs2)
    
    circuit.append_operation("MR",measurement)
    for k in range(len(measurement)):
        circuit.append_operation("DETECTOR", [stim.target_rec(-1-k),
                                              stim.target_rec(-1-distance-k)] )
    full_circuit = stim.Circuit()
    full_circuit.append_operation("M",measurement)
    full_circuit += circuit*rounds
    full_circuit.append_operation("M",data)
    for k in range(len(measurement)):
        full_circuit.append_operation("DETECTOR",[stim.target_rec(-1-k),
                                                  stim.target_rec(-2-k),
                                                  stim.target_rec(-2-k-distance)])

    full_circuit.append_operation("OBSERVABLE_INCLUDE",[stim.target_rec(-1)],0)
    return full_circuit

def shot(circuit):
    sample = circuit.compile_sampler().sample(1)[0]
    return "".join("_1"[int(e)] for e in sample)

def detector_shot(circuit):
    sample = circuit.compile_detector_sampler().sample(1,append_observables=True)[0]
    return "".join("_1"[int(e)] for e in sample)


# Generates surface code circuit tasks using Stim's circuit generation.
def generate_example_tasks():
    for p in [0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9]:
        for d in [3, 5, 7, 9]:
            yield sinter.Task(
                circuit = rep_code(distance = d,rounds = d,noise = p),
                json_metadata={
                    'p': p,
                    'd': d,
                },
            )


def main():
    print(rep_code(distance=3,rounds=2,noise = 0.01))
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
