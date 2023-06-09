# Se importan los modulos
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


# Definicion de las variables conocidas
R1 = 0.01
X1 = 0.1
R2 = 0.01
X2 = 0.1
P3 = 5
Q3 = 0

# Definicion de los valores iniciales
Vs = (1, 0, 1, 0, 1, 0)


# Definicion de la funcion objetivo
def objetivo(V):
    '''
    Suma de P1 y P2
    '''
    V1_real = V[0]
    V1_imag = V[1]
    V2_real = V[2]
    V2_imag = V[3]
    V3_real = V[4]
    V3_imag = V[5]

    I1RE = ((V1_real - V3_real)*R1 + (V1_imag - V3_imag)*X1)/(R1**2 + X1**2)

    I1IM = ((V1_imag - V3_imag)*R1 - (V1_real - V3_real)*X1)/(R1**2 + X1**2)

    I2RE = ((V2_real - V3_real)*R2 + (V2_imag - V3_imag)*X2)/(R2**2 + X2**2)

    I2IM = ((V2_imag - V3_imag)*R2 - (V2_real - V3_real)*X2)/(R2**2 + X2**2)

    return (I1RE**2 + I1IM**2)*R1 + (I2RE**2 + I2IM**2)*R2


# Definicion de la funcion para la restriccion 1
def restriccion1(V):
    '''
    Los potencias activas deben ser cero para minimizar perdidas
    '''
    V1_real = V[0]
    V1_imag = V[1]
    V2_real = V[2]
    V2_imag = V[3]
    V3_real = V[4]
    V3_imag = V[5]

    I1RE = ((V1_real - V3_real)*R1 + (V1_imag - V3_imag)*X1)/(R1**2 + X1**2)

    I1IM = ((V1_imag - V3_imag)*R1 - (V1_real - V3_real)*X1)/(R1**2 + X1**2)

    I2RE = ((V2_real - V3_real)*R2 + (V2_imag - V3_imag)*X2)/(R2**2 + X2**2)

    I2IM = ((V2_imag - V3_imag)*R2 - (V2_real - V3_real)*X2)/(R2**2 + X2**2)

    P3sys = (V3_real)*(I1RE +I2RE) + V3_imag*(I1IM+I2IM)

    return P3sys - P3


# Definicion de la funcion para la restriccion 2
def restriccion2(V):
    '''
    Los potencias reactivas deben ser cero para minimizar perdidas
    '''
    V1_real = V[0]
    V1_imag = V[1]
    V2_real = V[2]
    V2_imag = V[3]
    V3_real = V[4]
    V3_imag = V[5]

    I1RE = ((V1_real - V3_real)*R1 + (V1_imag - V3_imag)*X1)/(R1**2 + X1**2)

    I1IM = ((V1_imag - V3_imag)*R1 - (V1_real - V3_real)*X1)/(R1**2 + X1**2)

    I2RE = ((V2_real - V3_real)*R2 + (V2_imag - V3_imag)*X2)/(R2**2 + X2**2)

    I2IM = ((V2_imag - V3_imag)*R2 - (V2_real - V3_real)*X2)/(R2**2 + X2**2)

    Q3sys = (V3_imag)*(I1RE +I2RE) - V3_real*(I1IM+I2IM)

    return Q3sys - Q3


def restriccion3(V):
    '''
    Definicion de la primera restriccion donde
    las magnitudes de las tensiones deben estar
    entre los cuadrados de 0.95 y 1.05
    '''
    V1re = V[0]
    V1im = V[1]
    V2re = V[2]
    V2im = V[3]
    V3re = V[4]
    V3im = V[5]

    return np.array([np.sqrt((V1re**2) + (V1im**2)),
                     np.sqrt((V2re**2) + (V2im**2)),
                     np.sqrt((V3re**2) + (V3im**2))])


# Restricciones

NLC1 = NonlinearConstraint(restriccion1, 0, 0)
NLC2 = NonlinearConstraint(restriccion2, 0, 0)
NLC3 = NonlinearConstraint(restriccion3, (0.95)*np.ones(3), (1.05)*np.ones(3))

# Respuesta de la optimizacion
respuesta = minimize(objetivo, Vs, constraints=[NLC1, NLC2, NLC3])

# Se imprime la solucion de la optimizacion
# print(respuesta)

# Se extraen la parte real e imaginaria de las tensiones
valores = respuesta.x

# Lista vacia para contener magnitud y angulo de cada tension
resultados = []

# Determinacion de la magnitud y angulo de cada tension
for i in range(3):
    parte_real = valores[i*2]  # Valores impares son la parte real
    parte_imag = valores[i*2 + 1]  # Valores pares son la parte imaginaria
    magnitud = np.abs(parte_real + 1j*parte_imag)
    angulo = np.angle(parte_real + 1j*parte_imag)*(180/np.pi)
    resultados.append(magnitud)
    resultados.append(angulo)

# Magnitudes y angulos para cada tension
V1magn = resultados[0]
V1angl = resultados[1]
V2magn = resultados[2]
V2angl = resultados[3]
V3magn = resultados[4]
V3angl = resultados[5]

# Se imprimen los resultados
print('V1 con magnitud', V1magn, 'con angulo de', V1angl)
print('V2 con magnitud', V2magn, 'con angulo de', V2angl)
print('V3 con magnitud', V3magn, 'con angulo de', V3angl)
