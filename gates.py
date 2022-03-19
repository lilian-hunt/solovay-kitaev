import numpy as np


"""ONE QUBIT GATES"""
H = np.array([[1 , 1], [1 , -1]])/np.sqrt(2) # Hadamard
T = np.array([
  [1 , 0], 
  [0 , np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4)]
]) # P-gate with  ϕ = π/4

S = np.array([
  [1 , 0], 
  [0 , 1j]
]) # P-gate with  ϕ = π/2
s = S.transpose().conjugate() # S dagger
t = T.transpose().conjugate() # T dagger

# Pauli Gates
I = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]]) # Identity
X = np.array([[0 + 0j, 1 + 0j], [1 + 0j, 0 + 0j]])
Y = np.array([[0 + 0j, -1j], [1j, 0 + 0j]]) 
Z = np.array([[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]])

"""Multi Qubit Gates"""
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) # CNOT
