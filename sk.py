from utils import *
from gc_decompose import *
from find_nn_tree import *

""" 
Approximate a gate (U) to a given accuracy, as n gets bigger, the accuracy
increases because there are more iterations of the recursive algorithm
"""
def solovay_kitaev_2x2(U, n, eps=1e-3): 
  # TODO: terminate when approx within eps
  if (n == 0):
    # return Basic Approximation to U (a sequence)
    return get_basic_approx(U)
  else:
    U_n_minus_1, U_n_minus_1_seq = solovay_kitaev_2x2(U, n - 1)

    V, W = gc_decompose_2x2(U @ dagger(U_n_minus_1))

    # print("V", V)
    # print("W", W)

    V_n_minus_1, V_n_minus_1_seq = solovay_kitaev_2x2(V, n - 1)
    W_n_minus_1, W_n_minus_1_seq = solovay_kitaev_2x2(W, n - 1)

    U_now = V_n_minus_1 @ W_n_minus_1 @ dagger(V_n_minus_1) @ dagger(W_n_minus_1) @ U_n_minus_1

    U_seq = V_n_minus_1_seq + W_n_minus_1_seq + dagger_seq(V_n_minus_1_seq) + dagger_seq(W_n_minus_1_seq) + U_n_minus_1_seq
    return U_now, U_seq