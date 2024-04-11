import numpy as np
import control as ct

num1 = np.array([2])
den1 = np.array([1, 0])

num2 = np.array([3])
den2 = np.array([4, 1])

H1 = ct.tf(num1, den1)
H2 = ct.tf(num2, den2)

H = ct.parallel(H1, H2)
print('H(s) = ', H)


