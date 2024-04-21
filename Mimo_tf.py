import numpy as np
import control as ct

# Create a MIMO transfer function object
# The transfer function from the 2nd input to the 1st output is
# (3s + 4) / (6s^2 + 5s + 4).

num =np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
den = np.array([[[9., 8., 7.], [6., 5., 4.]], [[3., 2., 1.], [-1., -2., -3.]]])
G = ct.tf(num, den)
# print('G(s) = ', G)

# Create a variable 's' to allow algebra operations for SISO systems
s = ct.tf('s')
H = (s + 1)/(s**2 + 2*s + 1)
# print('H(s) = ', H)

# Convert a StateSpace to a TransferFunction object.

sys_ss = ct.ss("1. -2; 3. -4", "5.; 7", "6. 8", "9.")
sys2 = ct.tf(G)
print('G(s) = ', sys2)
