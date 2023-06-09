import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


# Definicion de los valores iniciales
x0 = np.array([1, 5, 5, 1])


# Definicion de la funcion
def funcion(x):
    '''
    Definicion de la funcion a optimizar donde se busca encontrar
    los valores x1, x2, x3 y x4 mas adecuados que cumplan con las
    restricciones (contrainsts).
    '''
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return x1*x4*(x1 + x2 + x3) + x3


# Definicion de la primera restriccion
C1 = LinearConstraint(np.eye(4), np.array([1, 1, 1, 1]),
                      np.array([5, 5, 5, 5]))


# Definicion de la segunda restriccion
def restr2(x):
    '''
    Definicion de la segunda restriccion no lineal del sistema,
    dada como la suma de los cuadrados de x1, x2, x3 y x4.
    '''
    return np.sum(x**2)


C2 = NonlinearConstraint(restr2, 40, 40)


# Definicion de la tercera restriccion
def restr3(x):
    '''
    Definicion de la tercera restriccion no lineal del sistema,
    dada como la multiplicacion de x1, x2, x3, x4 mayor o igual a 25.
    '''

    return x[0]*x[1]*x[2]*x[3]


C3 = NonlinearConstraint(restr3, 25, np.inf)

# Obtencion de los valores de x1, x2, x3 y x4 segun las restricciones
respuesta = minimize(funcion, x0, constraints=[C1, C2, C3])

# Imprime los valores
print(respuesta)
