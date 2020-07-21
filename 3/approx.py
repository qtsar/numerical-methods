import numpy as np
import matplotlib.pyplot as plt


x_points = np.arange(1, 65, 1, dtype=float)
x_st = np.linspace(1, 72, 1000)

y_points = np.array([684967,
                     716940,
                     688085,
                     666406,
                     703250,
                     724903,
                     729242,
                     736415,
                     758847,
                     884667,
                     960076,
                     951148,
                     928880,
                     1021458,
                     972515,
                     932520,
                     916935,
                     943282,
                     904364,
                     817960,
                     848536,
                     846035,
                     776401,
                     738890,
                     790007,
                     784556,
                     746411,
                     673846,
                     695104,
                     681685,
                     428278,
                     823732,
                     616555,
                     630333,
                     603212,
                     554707,
                     573237,
                     594808,
                     577958,
                     556165,
                     556845,
                     560410,
                     553055,
                     535939,
                     534152,
                     557307,
                     562415,
                     536704,
                     551003,
                     597349,
                     629586,
                     610538,
                     587102,
                     595349,
                     484454,
                     493158,
                     495207,
                     534986,
                     521436,
                     506847,
                     483415,
                     509551,
                     497913,
                     500653,
                     ], dtype=float)



print(x_points,'\n', y_points)


def polyfit_square(xx, yy, deg):

    order = int(deg) + 1
    # set up least squares equation for powers of x
    left_hs = np.vander(xx, order)
    right_hs = yy

    # print('left_hs', left_hs)
    # print('right_hs', right_hs)

    # scale lhs to improve condition number and solve
    scale = np.sqrt((left_hs * left_hs).sum(axis = 0))
    left_hs /= scale
    c = np.linalg.lstsq(left_hs, right_hs, rcond = None)[0]
    c = (c.T / scale).T 
    return c

degree = 15

pol_square = polyfit_square(x_points, y_points, degree) 

plt.plot(x_points, y_points, 'o')
plt.plot(x_st, np.polyval(pol_square, x_st), '-r')

plt.grid(True)
plt.show()

aaaa = np.polyval(pol_square, [65, 66, 67, 68])
print(np.array(aaaa, dtype=int))

