import numpy as np
import sinter
import stim
from dataclasses import dataclass


def generate_circuit_cycle(distance: int, rounds:int):
    
    lad_centers : dict[complex,int] = {}
    for row in range(2*distance):
        center = 1+row*2 + 1j
        category = row%2
        lad_centers[loop(center,distance=distance)]=category
    return full_circuit