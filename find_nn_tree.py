import dill # dill is important for dumping lambda functions....
import numpy as np
from gates import *
from utils import *

# Map to the gates defined in gates.py
gates_dic = {"H" : H, "S": S, "T": T, "C": C, "t": t, "s": s,
'h': H, "X": X, "Y": Y, "Z": Z, "z": Z, "x": X, "y": Y, "I": I, "i": I}

l0 = 8
pin = open(f"list_optimised{l0}.pickle", "rb") 
tree = dill.load(pin) # pickle_in

def find_nn(U):
  min_seen = None
  min_distance = float("inf")
  for gate, sequence in tree:
    calc_distance = distance(U, gate)
    if calc_distance < min_distance:
      min_distance = calc_distance
      min_seen = sequence
  return min_seen, min_distance

def get_basic_approx(U):
  # U  = SU2(U)
  sequence, min_distance = find_nn(U)
  sequence_multiplied = multiply_gates(sequence)
  # print(min_distance)

  # nearest_neighbours = tree.search_nn(matrix_unroll(U))
  # sequence = nearest_neighbours[0].__dict__['data'].__dict__['sequence']
  # If close to the unitary do nothing
  
  return sequence_multiplied, sequence


"""What is the best way to find the neighbours?
How to represent points in this space?
How big does this space get when you add more qubits?

Options M-tree
kd-tree 
"""