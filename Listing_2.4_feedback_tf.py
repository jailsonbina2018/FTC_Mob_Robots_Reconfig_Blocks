import numpy as np
import control as ct

num = np.array([2])
den = np.array([1, 0])

L = ct.tf(num, den)
H = ct.feedback(L, 1)

print('H(s) = ', H)
