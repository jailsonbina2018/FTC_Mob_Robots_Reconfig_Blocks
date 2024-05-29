import cvxpy as cp
from numpy import linalg

A = cp.Parameter(
    (2, 2), 
    value= [[1, 0.35],
            [0, 0.9955]]
)

B = cp.Parameter(
    (2, 1), 
    value=[[-0.0019, -0.1078]]
)

Q = cp.Variable(shape=(2,2), symmetric=True)
Y = cp.Variable(shape=(1,2))

constraints = []

constraints = [Q >> 0]

z = cp.bmat([
    [Q, Q.T @ A.T + Y.T @ B.T], 
    [A @ Q + B @ Y, Q]])

constraints += [z >> 0]

obj = cp.Minimize(0)

prob = cp.Problem(obj, constraints)

prob.solve(solver=cp.MOSEK, verbose=False)
# print(f"Valor  ́otimo: {prob.value}")

# print(f"Q={Q.value}")
# print()
# print(f"Y={Y.value}")
# print()
print(f"K = {Y.value @ linalg.inv(Q.value)}")