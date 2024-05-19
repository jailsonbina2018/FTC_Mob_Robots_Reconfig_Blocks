import numpy as np
import control as ct
import matplotlib.pyplot as plt

K1 = 2
K2 = 3
T = 4

num1 = np.array([K1])
den1 = np.array([1, 0])

num2 = np.array([K2])
den2 = np.array([T, 1])

H1 = ct.tf(num1, den1)
H2 = ct.tf(num2, den2)

H = ct.series(H1, H2)
(p, z) = ct.pzmap(H)

print('poles = ', p)
print('zeros = ', z)

plt.show()
# print('H(s) = ', H)
