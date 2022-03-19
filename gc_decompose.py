import numpy as np
from utils import *

def calculate_phi(theta):
  phi = 2 * np.arcsin((0.5 + 0.5 * np.cos(theta / 2) ** 0.25)) #sin(θ/2) = 2 sin^2(φ/2)(1 − sin4(φ/2))^0.5
  phi = phi % np.pi # since pi and 0 might be an answer, when theta = 0 phi should be 0 too 
  return round(phi, 9)

""" 
Find θ, n_x, n_y, n_z for a given matrix U.

We can decompose a matrix into the form cos(θ/2) I - i sin(θ/2)(n_x X + n_y Y + n_z Z)

Expanding that matrix, we can then solve for cos(θ/2) and sin(θ/2)

Then find theta with arctan2
"""
def bloch_decomposition(U):
  # Ambiguous edge case the identity can be represented by a 0 rotation about any axis
  if distance(U, I) < 1e-10:
    return 0, np.array([1, 0, 0])

  # Matrix needs to be in SU(2) for this to work
  U = SU2(U)

  # To normalise the axis
  N = np.sqrt((U[0][1].imag)**2 + (U[0][1].real)**2 + (U[0][0].imag)**2) # (n_x sin(theta/2)**2 + n_y sin(theta/2)**2  +  n_z sin(theta/2)**2)^0.5  

  n_x = -U[0][1].imag/N
  n_y = -U[0][1].real/N
  n_z = -U[0][0].imag/N

  sin_theta_on_2 = 0 # Im(a) = -sin(theta/2) n_z -> sin(theta/2) = -Im(a)/n_z where a is the element 0,0 in matrix

  find_sin = [(-U[0][0].imag, n_z), (-U[0][1].real, n_y), (-U[1][0].imag, n_x)]
  for options in find_sin:
    if options[1] != 0:
      sin_theta_on_2 = options[0]/options[1]
      break

  cos_theta_on_2 = U[0][0].real # Re(a) = cos(theta/2) -> cos(theta/2) = Re(a)

  theta = 2 * np.arctan2(sin_theta_on_2, cos_theta_on_2)

  return theta, np.array([n_x, n_y, n_z])

"""
Convert θ, n_x, n_y, n_z into a given matrix based on the equation
cos(θ/2) I - i sin(θ/2)(n_x X + n_y Y + n_z Z)

args: 
  theta - angle
  axis - array of size 3 [n_x, n_y, n_z]
"""
def convert_axis_angle_to_matrix(theta, axis):
	theta /= 2

	n_x = axis[0]
	n_y = axis[1]
	n_z = axis[2]

	matrix = np.cos(theta) * I - 1j * np.sin(theta) * (n_x*X + n_y*Y + n_z*Z)

	return SU2(matrix)

"""Round any matrix to 10dp because at that point floating point error
would probably come in"""
def round_for_floating_pt_error(matrix, dp=10):
  return np.round(matrix, dp)
  
"""For 2x2 matrix only 
U must be conjugate to a rotation by θ about the n axis identified in 
the previous two paragraphs, i.e., U = S(VWV*W*)S* it follows that
U = VWV*W* where V' = SVS* ,  W' = SWS*
"""
def gc_decompose_2x2(U): 
  # U is a rotation of theta about some axis --> find theta and the axis
  theta, U_axis = bloch_decomposition(U) 

  # Calculate phi based on sin(θ/2) = 2 sin^2(φ/2)(1 − sin4(φ/2))^0.5
  phi = calculate_phi(theta) 
  # print("phi error (should be close to 0) = " + str( 2*np.sin(phi/2)**2 * np.sqrt(1-np.sin(phi/2)**4) - np.sin(theta/2) ))

  V = rotate_X(np.pi - phi) # not just phi because it was rotating the wrong way
  W = rotate_Y(np.pi - phi)
  
  V_dagger = dagger(V)
  W_dagger = dagger(W)
  # print(theta, phi)
  # print("V", V)
  # print("W", W)
  
  group_commutator = round_for_floating_pt_error(V @ W @ V_dagger @ W_dagger)
  _, group_commutator_axis = bloch_decomposition(group_commutator)

	# Calcualte S, used to rotate U's axis to VWV†W†'s axis and back - similarity transform
  S_axis = np.cross(U_axis, group_commutator_axis)
  S_theta = np.arccos(abs(np.dot(U_axis, group_commutator_axis)) / (np.linalg.norm(U_axis) * np.linalg.norm(group_commutator_axis))) #TODO make sure this is right direction (do I need to negate???) ALSO MAKE SURE ARCCOS DIDN't FUCK EVERYTHING UP B/C OF DOMAIN AND STUFF

  S = convert_axis_angle_to_matrix(S_theta, S_axis)
  S_dagger = dagger(S) #S*

  # Should have U = S VWV*W* S*
  V_tilde = round_for_floating_pt_error(S @ V @ S_dagger) #V' = SVS*
  W_tilde = round_for_floating_pt_error(S @ W @ S_dagger) #W' = SWS*

  # print("\nU", U)
  # print("Delta", (V_tilde @ W_tilde @ dagger(V_tilde) @ dagger(W_tilde)))
  # print("GC_Decompose failed! U != VWV†W†", distance(U, (V_tilde @ W_tilde @ dagger(V_tilde) @ dagger(W_tilde))))
	
  return V_tilde, W_tilde
