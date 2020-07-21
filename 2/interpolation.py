import numpy as np
import sympy as smp
import matplotlib.pyplot as plt


x = smp.Symbol('x', real=True)  # ТОЛЬКО ВЕЩЕСТВЕННЫЕ

# F_expr = (0.4 - x)**2 + smp.tan(3*x)

F_expr = smp.tan(x) - smp.cos(x) + 0.1

# F_expr = x**8 + 1
H_expr = F_expr * smp.Abs(x)

print(F_expr)
print(H_expr)


def subs_x(value, expr):
    # print(smp.simplify(expr))

    return expr.subs(x, value)


def subs_to_np_expr(value, expr):
    # print(smp.simplify(expr))

    expr = expr.subs(x, value)
    expr_vector = smp.lambdify(x, expr, 'numpy')

    return expr_vector(value)


def omega(x_arr, f_df_d2f_arr):
    res = 1
    for xj in x_arr:
        # на самом то деле надо умножать на (максимальная производная + 1)
        res *= (x - xj)**len(f_df_d2f_arr)

    return res


def get_simmetric_interval(val, nodes_num, name):
    start_val = -val
    end_val = val

    if name == 'Uniform':
        named_x = np.linspace(start_val, end_val, nodes_num)
    elif name == 'Chebyshev':
        named_x = 1/2 * ((end_val - start_val) * np.cos(np.pi * (2 * np.arange(
            0, nodes_num) + 1) / (2 * (nodes_num))) + (end_val + start_val))

    normal_x = np.linspace(start_val, end_val, nodes_num * 1000)

    return named_x, normal_x


def get_hermite_polynom(x_arr, f_df_d2f_arr):
    hermite = 0

    for i in range(len(x_arr)):
        xj = x_arr[i]
        # print()
        # print('xj = {}'.format(xj))

        # print('----------------')
        func = (x - xj)**len(f_df_d2f_arr) / omega(x_arr, f_df_d2f_arr)
        # print(func)

        c = []
        for k in range(len(f_df_d2f_arr)):

            c_func = smp.diff(func, x, k) / smp.factorial(k)
            c.append(subs_x(xj, c_func))

            # print('c{} = {}'.format(k, c[k]))
            # print('\t-----')

        term, q_term, res_q_term = 0, 0, 0

        for q in (range(len(c))):
            q_term += (c[q] * (x-xj)**q)

            res_q_term = q_term * (x-xj)**(len(c) - 1 - q)

            term += (res_q_term *
                     f_df_d2f_arr[len(c) - 1 - q][i] / smp.factorial(len(c) - 1 - q))

            # print(f_df_d2f_arr[len(c) - 1 - q][i] /
            #   smp.factorial(len(c) - 1 - q))
            # print(res_q_term)

        # print()
        # print(smp.simplify(term))
        # print(1 / func)
        hermite += (term * 1 / func)

    # print('----- Hermite polynom -----')
    # print(smp.simplify(hermite))  # сильно тормозит программу
    return hermite


def get_lagrange_polynom(x_arr, f_arr):
    lagrange = 0

    for j in range(len(f_arr)):
        numerator, denominator = 1, 1
        for i in range(len(x_arr)):
            if i == j:
                numerator *= 1
                denominator *= 1
            else:
                numerator *= (x - x_arr[i])
                denominator *= (x_arr[j] - x_arr[i])

        # print('\t-----')
        # print(f_arr[j])
        # print(numerator / denominator)

        lagrange += (f_arr[j] * numerator / denominator)

    # print('----- Lagrange polynom -----')
    # print(smp.simplify(lagrange))  # сильно тормозит программу
    return lagrange


f_df_d2f_array = []


def get_plot(val, nodes_num, nodes_name, func_name, polynom_name, der):

    x_nodes, x_st = get_simmetric_interval(val, nodes_num, nodes_name)

    if func_name == 'f(x)':
        expr = F_expr
    elif func_name == 'h(x)':
        expr = H_expr

    y_nodes = subs_to_np_expr(x_nodes, expr)
    y_st = subs_to_np_expr(x_st, expr)

    # print("----- {} nodes -----".format(nodes_name))
    # print(x_nodes)
    # print("------ {} values in nodes -----".format(func_name))

    if polynom_name == 'Lagrange':
        y_st_polynom = subs_to_np_expr(
            x_st, get_lagrange_polynom(x_nodes, y_nodes))

    elif polynom_name == 'Hermite':
        f_df_d2f_array = np.empty(shape=[0, len(x_nodes)])

        for i in range(der):  # <-- ЗДЕСЬ МЕНЯТЬ КОЛИЧЕСТВО ПРОИЗВОДНЫХ
            line = np.array([])

            if i == 0:
                line = y_nodes
            else:
                for x_val in x_nodes:
                    el = smp.limit(smp.diff(expr, x, i), x, x_val)
                    # if x_val == 0:
                    #     el = smp.limit(smp.diff(expr, x, i), x, x_val)
                    # else:
                    #     el = subs_to_np_expr(x_val, smp.diff(expr, x, i))

                    line = np.append(line, el)

            f_df_d2f_array = np.append(f_df_d2f_array, [line], axis=0)

        # print(f_df_d2f_array)

        y_st_polynom = subs_to_np_expr(
            x_st, get_hermite_polynom(x_nodes, f_df_d2f_array))

    max_diff = max(np.abs(y_st_polynom - y_st))

    print(" {:10} | {:2} | {:8} | {:.15f}".format(
        nodes_name, nodes_num, polynom_name, max_diff))

    plt.figure('{} polynom, {}, {} nodes'. format(
        polynom_name, func_name, nodes_name))
    plt.title('{} polynom, {}, {} nodes'. format(
        polynom_name, func_name, nodes_name))

    plt.plot(x_nodes, y_nodes, 'or', x_st,
             y_st_polynom, '-r', x_st, y_st, ':b')

    # plt.plot(x_st, y_st, '-r')

    plt.legend(('nodes', '{} polynom'.format(polynom_name), 'real function'))
    plt.xlabel('max diff = {:.10f}'.format(max_diff))
    plt.grid(True)


def get_table(value1, func1, name, max_nodes, polynom):
    print(" {} Nodes |  # | Polynom  | Max diff".format(func1))
    print("------------------------------------------------------")
    for index_nodes in range(3, max_nodes + 1):
        get_plot(value1, index_nodes, name, func1, polynom, 3)


cheb = 'Chebyshev'
lagrange = 'Lagrange'
uniform = 'Uniform'
hermite = 'Hermite'


# get_plot(val, nodes_num, nodes_name, func_name, polynom_name, derivatives)
# это просто график чего то одного
# get_plot(1, 3, uniform, 'f(x)', hermite, 3)

# это таблица
get_table(1.5, 'f(x)', cheb, 12, lagrange)
plt.show()