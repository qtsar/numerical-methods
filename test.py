import matplotlib.pyplot as plt
import numpy as np 
from numpy import linalg as LA
import scipy as sp 


x_nodes = np.array([-1, -0.83, -0.11, 0.45, 1])
x_st = np.linspace(-1, 1, 1000)


def f(x):
    return x - np.sin(x) + x**5

degree = 3
y_nodes = f(x_nodes)
y_st = f(x_st)



def der_spec_expr(x, k):
    return sp.misc.derivative((1 - x**2)**k, x, order=k)


order = degree + 1
p = 1
pol_lagrange = 0

for n in range(order):
    L_k = 1 / (np.math.factorial(n) * 2**n) * der_spec_expr(x_nodes, n)

    expr_num = lambda x_nodes: p * f(x_nodes) * L_k
    expr_denom = lambda x_nodes: p * L_k**2

    num = sp.integrate.quad(expr_num, -1, 1)
    denom = sp.integrate.quad(expr_denom, -1, 1)

    c_k = num / denom

    pol_lagrange += (c_k * L_k)


plt.plot(x_nodes, y_nodes, 'o')
plt.plot(x_st, y_st, '-b')
plt.plot(x_st, np.polyval(pol_lagrange, x_st), '-r')

plt.grid(True)
plt.show()
