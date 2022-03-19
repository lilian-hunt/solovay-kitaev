import numpy as np 
import kdtree 
import dill
from gates import *
from utils import *

# Store the coordinates and sequence 
# class Sequence(object): 
# 	def __init__(self, r1, c1, r2, c2, sequence):
# 		self.coords = (r1,c1,r2,c2)
# 		self.sequence = sequence 
	
# 	def __len__(self): 
# 		return len(self.coords)

# 	def __getitem__(self, i):
# 		return self.coords[i]

# 	def __repr__(self):
# 		return 'Sequence({},{},{},{},{})'.format(self.coords[0], self.coords[1], self.coords[2], self.coords[3], self.sequence)

# def convert_matrix_to_sequence(matrix, sequence):
#   # Unroll the matrix and tuple it 
#   r1 = matrix[0][0].real
#   c1 = matrix[0][0].imag
#   r2 = matrix[0][1].real
#   c2 = matrix[0][1].imag		
	
#   # Create sequence object
#   return Sequence(r1, c1, r2, c2, sequence) 

def generate_sequences(gates, l):
  basic_gates_sequences = []
  
  # Recursive helper function
  def generate_util(gates, sequence, current_matrix, num_gates, l): 
    # Base case
    if (l == 0):    	
      return 

    # Otherwise, append the basis gate
    for gate in gates:
      # sequence + gate cancels with the last term skip, we can skip
      """
      HH -> XX -> YY -> ZZ -> Tt -> tT -> Ss -> sS -> I
      TT -> S
      SS -> Z
      HZH -> X
      HXH -> Z
      XY -> Z
      YZ -> X
      ZX -> Y
      """
      if gate == sequence[-1]:
        # All cancel 
        # HH -> XX -> YY -> ZZ -> I
        continue
      if gate == "H" and (sequence[-2:] == "HZ" or sequence[-2:] == "HX"):
        # if it's HZH and HXH
        # HZH -> X
        # HXH -> Z
        continue
      new_sequence = sequence + gate
      if new_sequence[-2:] == "XY" or new_sequence[-2:] == "YZ" or new_sequence[-2:] == "ZX":
        # XY -> Z
        # YZ -> X
        # ZX -> Y
        continue
      if new_sequence[-2:] == "tT" or new_sequence[-2:] == "Tt" or new_sequence[-2:] == "sS" or new_sequence[-2:] == "Ss":
        # Tt -> tT -> Ss -> sS -> I
        continue
      
      # Only if this is a new sequence add it
      new_matrix = current_matrix @ gates_dic[gate]
      
      points_sequence = (new_matrix, new_sequence)
      basic_gates_sequences.append(points_sequence) 

      generate_util(gates, new_sequence, new_matrix, num_gates, l-1)
  
  # Call the recursive function 
  # I should be the root node but exclude it from the rest of the tree
  generate_util(gates, "I", np.array([[1, 0], [0, 1]]), len(gates), l)
  return basic_gates_sequences

# Create tree and save into pickle file
def build_tree(basis_gates, l0):  
  basic_gates_sequences = generate_sequences(basis_gates, l0)   

  # Create the KDTree, so that we can find most similar gate
  # KD_Tree  = kdtree.create(basic_gates_sequences) 
  
  # Save into pickle file 
  pickle_out = open("list_optimised" + str(l0) + ".pickle", "wb") 
  dill.dump(basic_gates_sequences, pickle_out) 
  pickle_out.close()


# Map to the gates defined in gates.py
gates_dic = {
  "H": H, "S": S, "T": T, "C": C, "X": X,
  "Y": Y, "Z": Z, "I": I, "s": s, "t": t
  }

l0 = 10 # CRASHES AFTER GATE SEQUENCE OF LENGTH 10 
qubit_basis_2x2 = ['H', 'S', 'T', 'X', 'Y', 'Z', 's', 't'] # I should only be at the root of the tree

build_tree(qubit_basis_2x2, l0)
print("done")

# np.kron()

# for more qubits -> the basis gate set will be all gates tensored with I
