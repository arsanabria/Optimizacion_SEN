# Se importan las librerias
import numpy as np
from scipy.optimize import minimize


# Definicion de la funcion objetivo
def objectivo(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return  x1*x4*(x1+x2+x3) +x3

# Primera funcion restriccion
def CT(x):
    return x[0]*x[1]*x[2]*x[3]-25

# Segunda funcion restriccion
def CT2(x):
    sum_sq = 40
    for i in range(4):
        sum_sq = sum_sq - x[i]**2
    return sum_sq

# Valores iniciales
x0 = [1,5,5,1]


# Limite inferior y superior para las variables x's
b = (1.0, 5.0)
# Tupla que contiene los limites para cada x's
bnds = (b,b,b,b)

# restricciones
# restriccion desigual donde se define que debe ser mayor o igual a cero
con1 = {'type': 'ineq', 'fun': CT}

# restriccion de igualdad donde debe ser igual a cero
con2 = {'type': 'eq', 'fun': CT2}
cons = [con1, con2]


# optimiazcion del problema por medio de minimize
''' método Sequential Least SQuares Programming (SLSQP),
que es un método de optimización no lineal con restricciones '''
sol = minimize(objectivo, x0, method='SLSQP', bounds=bnds,constraints=cons)

print(sol)