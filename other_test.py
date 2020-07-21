import sympy as smp 


x = smp.Symbol('x', real = True)


n_expr = 1 - x**2
deg = 3

# for n in range(deg + 1):
#     n_expr = (1 - x**2)**n
#     L_kk = 1 / (smp.factorial(n) * 2**n) * smp.diff(n_expr, x, n)
#     print(L_kk)
#     print('---')

"""
1
---
-x
---
(3*x**2 - 1)/2
---
-x*(5*x**2 - 3)/2

"""
# prev_expr = 1
# der_expr = smp.diff(n_expr, x)
# for n in range(deg + 1):    
#     if n == 0: 
#         L_k = 1
#     else:
#         next_expr = smp.diff(prev_expr, x) * n_expr + der_expr * prev_expr
#         L_k = next_expr / n / 2
#         prev_expr = next_expr 

#     print(L_k)
#     print('---')


for n in range(4):
    print(smp.diff((1 - x**2)**n, x, n))


    