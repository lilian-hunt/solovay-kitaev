from dis import dis
from distutils.command.clean import clean
import numpy as np
import cmath
from gates import *
import gates 

def dagger(matrix):
  return matrix.transpose().conjugate()

def determinant(matrix):
  return np.linalg.det(matrix)

"""
U  be any unitary matrix st UU*=I . 
U' be its transform in SU(2).
then U' = [1/det(U)]^0.5
Multipy the matrix by a scalar factor such that det(UU*) = 1
Decompose to a gate in SU(2), unitary with determinant 1
"""
def SU2(U):
  t = complex(0,0) + determinant(U)
  globalPhase = np.sqrt(1/t)  
  return U * globalPhase 

"""Calculate norm for matrix
Args: 
  x: matrix or vector
Returns:
  norm: scalar
"""
def norm(x):
  return np.linalg.norm(x)

# Diagonalize a matrix
def diagonalize(matrix):            
	eigenvalues, eigenvectors = np.linalg.eig(matrix)
	return eigenvalues, eigenvectors

# Check if two matrix are equal within the tolerance
def isEqual(matrix1, matrix2, tolerance=1e-9):  
  r1 = abs(matrix1[0][0] - matrix2[0][0])
  r2 = abs(matrix1[1][0] - matrix2[1][0])
  r3 = abs(matrix1[0][1] - matrix2[0][1])
  r4 = abs(matrix1[1][1] - matrix2[1][1])

  if (r1 < tolerance and r2 < tolerance and r3 < tolerance and r4 < tolerance):
    return True
  return False, r1, r2, r3, r4

"""
Find the square root of a given matrix:

If U = V D V†

Then U^0.5 = V D^0.5 V†

(U^0.5)^2 = (V D^0.5 V†) (V D^0.5 V†)
          = V D^0.5 D^0.5 V†
          = V D V†

Can just take the square root of each element in the diagonal matrix to 
find the square root of the matrix
"""
def matrix_sqr_root(matrix):
  eigenvalues, eigenvectors = np.linalg.eig(matrix)

  diagonal_matrix = np.array([
    [cmath.sqrt(eigenvalues[0]), 0], [0, cmath.sqrt(eigenvalues[1])]
  ]) # diagonal matrix

  V = eigenvectors 

  square_root = V @ diagonal_matrix @ dagger(V)
  return square_root 


"""
Distance between two matrices
"""
def distance(matrix1, matrix2):
  diff = np.round(matrix1, 10) - np.round(matrix2, 10)
  return np.linalg.norm(diff)

"""
Return the complex conjugate of the sequence where each gate is represented
as a single character, and lowercase is the complex conjugate
e.g. HtS --> sTh

Args:
  sequence: string
Returns:
  dagger of sequence: string
"""
def dagger_seq(sequence):
  return sequence.swapcase()[::-1]

gates_dic = {"H": H, "S": S, "T": T, "C": C, "t": t, "s": s,
'h': H, "X": X, "Y": Y, "Z": Z, "z": Z, "x": X, "y": Y, "I": I, "i": I}

def multiply_gates(sequence):
  matrix = I
  for gate in sequence[::-1]:
    matrix = np.matmul(gates_dic[gate], matrix)
  return matrix

# Based on Nielsen & Chuang eq 4.4
def rotate_X(phi):
  V = np.cos(phi / 2) * gates.I  - 1j * np.sin(phi / 2) * gates.X	
  return V

# Based on Nielsen & Chuang eq 4.5 
def rotate_Y(phi):
  W = np.cos(phi / 2) * gates.I  - 1j * np.sin(phi / 2) * gates.Y
  return W  

# Based on Nielsen & Chuang eq 4.6
def rotate_Z(phi):
	Z = np.cos(phi / 2) * gates.I  - 1j * np.sin(phi / 2) * gates.Z	
	return Z 

# make recusive eventually 
def cleanup_sequence(string):
  string = string.replace("i", "").replace("I","").replace("h","H").replace("HH", "").replace("SSSS", "")
  return string


def cleanup_recursive(string):
  """
  Gates that reduce 
    HH -> XX -> YY -> ZZ -> Tt -> tT -> Ss -> sS -> I
    TT -> S
    SS -> Z
    HZH -> X
    HXH -> Z
    XY -> Z
    YZ -> X
    ZX -> Y
  """
  replace_pairs = (
    ("HH", ""), 
    ("XX", ""),
    ("ZZ", ""),
    ("YY", ""),
    ("sS", ""),
    ("Ss", ""),
    ("tT", ""),
    ("Tt", ""),
    ("TT", "S"),
    ("SSSS", ""),
    ("HXH", "Z"),
    ("HZH", "X"),
  )

  # For hermitian matrices replace dagger with the original
  cleaned_string = string.replace("x", "X").replace("y", "Y").replace("z", "Z").replace("h", "H")

  # Remove all the identity matrices
  cleaned_string = cleaned_string.replace("I", "").replace("i", "")

  for pair in replace_pairs:
    cleaned_string = cleaned_string.replace(pair[0], pair[1])

  if string != cleaned_string:
    return cleanup_recursive(cleaned_string)
  return cleaned_string

"""
Check the calculation of theta and phi are correct up to a given 
tolerance, which is by default 1e-9
"""
def check_theta_phi(theta, phi, tolerance=1e-9):
  lhs = np.sin(theta / 2)
  rhs = 2 * (np.sin(phi / 2) ** 2) * (1 - (np.sin(phi / 2) ** 4)) ** 0.5
  
  return abs(lhs-rhs) < tolerance
