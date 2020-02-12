import numpy as np
import matplotlib.pyplot as plt
from structures import Matrix, Vector

#! problem setup
A = np.array(
    [[3.0, 2.0],
    [2.0, 6.0]]
)
b = np.array([2.0, -8.0])
# solution is x = [2, -2]

#! initial x
x = np.array([-8.0, 8.0])
all_x = [np.ndarray.copy(x)]


#! algorithm
r = b - np.matmul(A, x)
d = np.ndarray.copy(r)

for _ in range(2):
    α = r.dot(r) / d.dot(np.matmul(A, d))           # calculate step size
    x += α * d                                      # update x

    r_2 = r - α * np.matmul(A, d)                   # calculate new residual
    β = r_2.dot(r_2) / r.dot(r)                     # calculate coefficient for Gram-Schmidt
    d = r_2 + β * d                                 # find new search direction

    r = r_2
    all_x.append(np.ndarray.copy(x))

#! plot contours
delta = 0.01
u = np.arange(-10.0, 10.0, delta)
x, y = np.meshgrid(u, u)
z = 1.5*(x**2) + 2*x*y + 3*(y**2) - 2*x + 8*y

plt.contour(x, y, z)

#! plot iterations
all_x = np.array(all_x).T
plt.plot(all_x[0], all_x[1])

#! graph formatting
plt.title('Conjugate Gradient')
plt.show()