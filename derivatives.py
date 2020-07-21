import sympy as smp
import numpy as np 


x = smp.Symbol('x', real=True)

f = (smp.tan(3*x) + (0.4 - x)**2)*smp.Abs(x)

df_dx = smp.diff(f, x, 1)

d2f_dx2 = smp.diff(f, x , 2)

print('\n', f, '\n', df_dx, '\n', d2f_dx2)


def subs_to_np_expr(value, expr):
    print(smp.simplify(expr))

    expr = expr.subs(x, value)
    expr_vector = smp.lambdify(x, expr, 'numpy')
    return expr_vector(value)

def subs_x(value, expr):
    # print(smp.simplify(expr))

    return expr.subs(x, value)

print("-----------")
print(subs_to_np_expr(np.array([4, 1]), f))
print(subs_to_np_expr(np.array([4, 1]), df_dx))
print(subs_to_np_expr(2, d2f_dx2))



