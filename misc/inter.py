import numpy as np
from scipy.interpolate import lagrange, CubicSpline
import matplotlib.pyplot as plt

fun = lambda x: 1/(1+x**2)
xmin = -1.5
xmax = 1.5
x = np.linspace(xmin, xmax, int(xmax - xmin) + 1, endpoint=True)
y = [fun(i) for i in x]

inter = CubicSpline(x,y)
inp = np.linspace(xmin, xmax, 100, endpoint=True)
plt.plot(inp, [fun(i) for i in inp])
plt.plot(inp, [inter(i) for i in inp])
plt.scatter(x, y, c="k")
plt.show()
breakpoint()