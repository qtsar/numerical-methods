import numpy as np
from numpy import linalg as LA


# check matrix for positive definiteness
def is_positive_definite(matrix):
    return np.all(LA.eigvals(matrix) > 0)


# create random matrixes
def create_matrixes(size):
    while True:
        array = np.random.uniform(-5, 5, (size, size))
        array = array @ array.T

        if is_positive_definite(array):
            break

    vector = np.random.uniform(-5, 5, (size, 1))

    array = np.around(array, 2)
    vector = np.around(vector, 2)

    print("\nA matrix\n", array, "\n\nb vector\n", vector, "\n")
    return array, vector


eps = 10**(-6)
# s = int(input("Enter the size of matrix (one digit): "))
s = 3
A_array, b_vector = create_matrixes(s)


# f(x) value
def f(x):
    return np.asscalar(1/2 * x.T @ A_array @ x + x.T @ b_vector)


# Fastest Gradient Descent Method
def fastest_gradient_descent(A, b, x0):
    print("\nGradient Descent Mtd\n--------------------")

    i = 0
    x_next = x0

    while True:
        i += 1
        x_prev = x_next
        
        q = A @ x_prev + b
        mu = (q.T @ q) / (q.T @ A @ q)

        x_next = x_prev - np.asscalar(mu) * q

        if LA.norm(A @ x_next + b) < eps:
            print(i, "iterations\n")
            print(x_next, "= x\n\n", f(x_next), "= f(x)")
            return x_next, f(x_next)


# Fastest Coordinate Descent Method
def fastest_coordinate_descent(A, b, x0):
    print("\nCoordinate Descent Mtd\n----------------------")

    j = 0
    x_next = x0

    while True:
        j += 1
        x_prev = x_next

        q = A @ x_prev + b

        max_mu = 0
        mem = 0
        for i in range(s):

            mu = np.asscalar(q[i]) / A[i][i]
            print(mu)
            if np.abs(mu) > max_mu:
                max_mu = mu
                mem = i            

        x_next[mem][0] = x_prev[mem][0] - max_mu

        if LA.norm(A @ x_next + b) < eps:
            print(j, "iterations\n")
            print(x_next, "= x\n\n", f(x_next), "= f(x)")
            return x_next, f(x_next)



# Gauss Method
def solve_gauss(A, b):
    print("\nGauss Mtd\n---------")

    n = A.shape[0]

    for i in range(n):
        b[i] = b[i] / A[i][i]
        A[i] = A[i] / A[i][i]

        for j in range(i + 1, n):
            b[j] = b[j] - A[j][i] * b[i]
            A[j] = A[j] - A[j][i] * A[i]

    for i in range(n - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            b[j] -= b[i] * A[j][i]

    print(b, "= x\n\n", f(b), "= f(x)")
    return b, f(b)


# x_0 = np.array([[0], [0]])
x_0 = np.random.uniform(-5, 5, (s, 1))
x_0 = np.around(x_0)
print("Initial value\n", x_0, "\n")

#res1 = fastest_gradient_descent(A_array, b_vector, x_0)[0]
res2 = fastest_coordinate_descent(A_array, b_vector, x_0)[0]
#res3 = solve_gauss(A_array.copy(), -b_vector.copy())[0]

# print("\nDifference\n----------")
# print("Gradient Mtd & Gauss Mtd:", vector_diff(res1, res3))
# print("Coordinate Mtd & Gauss Mtd:", vector_diff(res2, res3))
# print("Gradient Mtd & Coordinate Mtd:", vector_diff(res1, res2))




