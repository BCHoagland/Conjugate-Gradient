import numpy as np
import matplotlib.pyplot as plt

A = np.array(
    [[3, 2],
    [2, 6]]
)
b = np.array([2, -8])
# solution is x = [2, -2]

num_iter = 10


#! step through iterations
x = np.array([-8, 8]) + np.random.rand(2)                 # initial guess
all_x = [np.ndarray.copy(x)]                                # store past iterates for plotting

for _ in range(num_iter):
    r = b - np.matmul(A, x)                                 # calculate residual
    α = (r.dot(r)) / (r.dot(np.matmul(A, r)))               # calculate optimal step size
    x += α * r                                              # update iterate
    all_x.append(np.ndarray.copy(x))                        # store iterate

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
plt.title('Steepest Descent')
plt.show()