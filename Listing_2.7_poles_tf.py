import numpy as np
import control as ct
import matplotlib.pyplot as plt

# %% Creating the transfer function:
num = np.array([1, 2])
den = np.array([1, 4])

H = ct.tf(num, den)

(p, z) = ct.pzmap(H)

print('poles = ', p)
print('zeros = ', z)

plt.show()
# plt.savefig('poles_zeros.pdf')
