#Se importan los modulos
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


# Definicion de las variables conocidas
R1 = 0.01
X1 = 0.1
R2 = 0.01
X2 = 0.1
P3 = 5
Q3 = 0
P4 = 0
Q4 = 0
#P4 = P3/3
#Q4 = Q3/3


# Definicion de los valores iniciales
Vs = (1, 0, 1, 0, 1, 0, 1, 0)


# Definicion de la funcion objetivo
def objetivo(V):
    '''
    Suma de P1, P2 y P4
    '''
    V1_real, V1_imag, V2_real, V2_imag, V3_real, V3_imag, V4_real, V4_imag = V
    # Tensiones
    V1 = V1_real + V1_imag*1j
    V2 = V2_real + V2_imag*1j
    V3 = V3_real + V3_imag*1j
    V4 = V4_real + V4_imag*1j
    # Corrientes
    I13 = (V1 - V3)/(R1 + X1*1j)
    I23 = (V2 - V3)/(R2 + X2*1j)
    I14 = (V1 - V4)/(R1 + X1*1j) 

    return (abs(I13)**2)*R1 + (abs(I23)**2)*R2 + (abs(I14)**2)*R1


# Definicion de la funcion para la restriccion 1
def restriccion1(V):
    '''
    Los potencias activas P3 deben ser cero para minimizar perdidas
    '''
    V1_real, V1_imag, V2_real, V2_imag, V3_real, V3_imag, V4_real, V4_imag = V
    # Tensiones
    V1 = V1_real + V1_imag*1j
    V2 = V2_real + V2_imag*1j
    V3 = V3_real + V3_imag*1j
    # Corrientes
    I13 = (V1 - V3)/(R1 + X1*1j)
    I23 = (V2 - V3)/(R2 + X2*1j)
    I3 = I13 + I23
    # Potencia aparente
    S3 = V3*I3.conjugate()
    # Potencia activa Sys
    P3sys = S3.real
    
    return P3sys - P3


# Definicion de la funcion para la restriccion 2
def restriccion2(V):
    '''
    Los potencias reactivas Q3 deben ser cero para minimizar perdidas
    '''
    V1_real, V1_imag, V2_real, V2_imag, V3_real, V3_imag, V4_real, V4_imag = V
    # Tensiones
    V1 = V1_real + V1_imag*1j
    V2 = V2_real + V2_imag*1j
    V3 = V3_real + V3_imag*1j
    # Corrientes
    I13 = (V1 - V3)/(R1 + X1*1j)
    I23 = (V2 - V3)/(R2 + X2*1j)
    I3 = I13 + I23
    # Potencia aparente
    S3 = V3*I3.conjugate()
    # Potencia reactiva Sys
    Q3sys = S3.imag
    return Q3sys - Q3

# Definicion de la funcion para la restriccion 3
def restriccion3(V):
    '''
    Los potencias activas P4 deben ser cero para minimizar perdidas
    '''
    V1_real, V1_imag, V2_real, V2_imag, V3_real, V3_imag, V4_real, V4_imag = V
    # Tensiones
    V1 = V1_real + V1_imag*1j
    V4 = V4_real + V4_imag*1j
    # Corrientes
    I14 = (V1 - V4)/(R1 + X1*1j)
    # Potencia aparente
    S4 = V4*I14.conjugate()
    # Potencia activa Sys
    P4sys = S4.real
    return P4sys - P4


# Definicion de la funcion para la restriccion 4
def restriccion4(V):
    '''
    Los potencias reactivas Q4 deben ser cero para minimizar perdidas
    '''
    V1_real, V1_imag, V2_real, V2_imag, V3_real, V3_imag, V4_real, V4_imag = V
    # Tensiones
    V1 = V1_real + V1_imag*1j
    V4 = V4_real + V4_imag*1j
    # Corriente
    I14 = (V1 - V4)/(R1 + X1*1j)
    # Potencia aparente
    S4 = V4*I14.conjugate()
    # Potencia activa Sys
    Q4sys = S4.imag
    return Q4sys - Q4

def restriccion5(V):
    '''
    Definicion de la primera restriccion donde
    las magnitudes de las tensiones deben estar
    entre los cuadrados de 0.95 y 1.05
    '''
    V1re, V1im, V2re, V2im, V3re, V3im, V4re, V4im = V

    return np.array([np.sqrt((V1re**2) + (V1im**2)),
                     np.sqrt((V2re**2) + (V2im**2)),
                     np.sqrt((V3re**2) + (V3im**2)),
                     np.sqrt((V4re**2) + (V4im**2))])



# Restricciones

if P4== 0 and Q4 == 0 :
    NLC1 = NonlinearConstraint(restriccion1, 0, 0)
    NLC2 = NonlinearConstraint(restriccion2, 0, 0)
    NLC5 = NonlinearConstraint(restriccion5, (0.95)*np.ones(4), (1.05)*np.ones(4))
    # Respuesta de la optimizacion
    respuesta = minimize(objetivo, Vs, constraints=[NLC1, NLC2, NLC5])
else:
    NLC1 = NonlinearConstraint(restriccion1, 0, 0)
    NLC2 = NonlinearConstraint(restriccion2, 0, 0)
    NLC3 = NonlinearConstraint(restriccion3, 0, 0)
    NLC4 = NonlinearConstraint(restriccion4, 0, 0)
    NLC5 = NonlinearConstraint(restriccion5, (0.95)*np.ones(4), (1.05)*np.ones(4))
    # Respuesta de la optimizacion
    respuesta = minimize(objetivo, Vs, constraints=[NLC1, NLC2, NLC3, NLC4, NLC5])
    
# Se imprime la solucion de la optimizacion
# print(respuesta)

# Se extraen la parte real e imaginaria de las tensiones
valores = respuesta.x

# Lista vacia para contener magnitud y angulo de cada tension
resultados = []

# Determinacion de la magnitud y angulo de cada tension
for i in range(4):
    parte_real = valores[i*2]  # Valores impares son la parte real
    parte_imag = valores[i*2 + 1]  # Valores pares son la parte imaginaria
    magnitud = np.abs(parte_real + 1j*parte_imag)
    angulo = np.angle(parte_real + 1j*parte_imag)*(180/np.pi)
    resultados.append(magnitud)
    resultados.append(angulo)

# Magnitudes y angulos para cada tension
V1magn, V1angl, V2magn, V2angl, V3magn, V3angl, V4magn, V4angl = resultados
# Se imprimen los resultados
print('V1 con magnitud', V1magn, 'con angulo de', V1angl)
print('V2 con magnitud', V2magn, 'con angulo de', V2angl)
print('V3 con magnitud', V3magn, 'con angulo de', V3angl)
print('V4 con magnitud', V4magn, 'con angulo de', V4angl)
