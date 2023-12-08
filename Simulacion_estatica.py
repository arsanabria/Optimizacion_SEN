"""
                    Universidad de Costa Rica
                     Facultad de Ingenieria
                Escuela de Ingeniería Eléctrica
                  IE0499 - Proyecto Eléctrico
                            II-2023

                    Simulació estática de SEP
                    Aroon Sanabria Torres B97205

                Profesor tutor: Dr. Andrés Argüello Guillén
                Profesor colaborador: MSc. Francisco Escobar Prado
                        02 de Diciembre, 2023

Descripción del programa: Realiza la simulación estática de un sistema eléctrico de potencia
mediante el cambio en la tensión y velocidad de referencia de un generador eléctrico para N
unidades presentes en el sistema con el analizar el comportamiento del redespacho de generación
ante el cierre de una línea de transmisión. El programa importa archivos tipo .sav, .dyr y .xlsx
que describen el sistema, los modelos asociados y los cambios de potencia activa y tensión del 
redespacho respectivamente. Con ello, se obtienen la respuesta del sistema mediante la gráfica 
de las variables eléctricas de estudio: Potencia activa y tensión en terminales de los generadores.
"""


# Se importan las librerías
import os
import sys
import matplotlib.pyplot as plt

# Dirección del PATH para importar el PSSE
psse_path = r'C:\Program Files\PTI\PSSE35\35.4\PSSBIN'
os.environ['PATH']=os.environ['PATH']+';'+psse_path
pssepythonpath = r'C:\Program Files\PTI\PSSE35\35.4\PSSPY37'
sys.path.append(pssepythonpath)

# Se importan las librerías de PSS/e-
import psspy
import redirect
import dyntools
import pssarrays
import openpyxl
import re

# Se redirije la salida del PSS/e a Python y viceversa. 
redirect.psse2py()

# Valores por defecto entero, flotante y carácter.
_i=psspy.getdefaultint()
_f=psspy.getdefaultreal()
_c=psspy.getdefaultchar()

class Sistema():
    """
    Contiene la información de todo el sistema eléctrico de potencia.
    """

    def __init__(self) -> None:
        self.Vref = []          # Tensiones de referencia de los generador.
        self.Gref = []          # Velocidades de referencia de los generador.
        self.DeltaVref = []     # Cambios en las tensiones de referencia de los generadores.
        self.DeltaGref =  []    # Cambios en las velocidades de referencia de los generadores.
        self.Buses_PV = []      # Nombre de los buses presentes en el SEP.
        self.Pinicial = []      # Potencia activa inicial de los generadores en MW.
        self.Pfinal = []        # Potencia final inicial de los generadores en MW.
        self.DeltaP = []        # Cambios de potencia activa de los generadores en pu.
        self.Vinicial = []      # Tensión inicial de los generadores en kV.
        self.Vfinal = []        # Tensión final de los generadores en kV.
        self.DeltaV = []        # Cambios de tensión de los generadores en pu.      
        self.Vbase = []         # Tensiones base de los generadores en kV.
        self.Sbase = []         # Potencia aparente base del sistema en MVA.
        self.ID = []            # ID asociados a cada uno de los generadores.
        self.Pgrafica = []      # Potencias activas  de los generadores ante el redespacho de generación.
        self.Vgrafica = []      # Tensiones de los generadores ante el redespacho de generación.
        self.Tgrafica = []      # Tiempo de simulación ante el redespacho de generación.
        self.SAV = None         # Almacena el archivo tipo .sav.
        self.DYR = None         # Almacena el archivo tipo .dyr.
        self.OUT_A = None       # Almacena el archivo .out del punto inicial.
        self.OUT_B = None       # Almacena el archivo .out del punto final.
        

    def importar_datos(self, archivo_sav: str, archivo_dyr: str, archivo_xlsx:str):
        """
        Importa los datos del archivo excel para obtener los cambios en las potencias activas y tensiones del sistema.
        """
        # Referencias a los archivos fisicos de "entrada".
        self.SAV = archivo_sav            
        self.DYR = archivo_dyr    

        # Inicializa el PSS/e para obtener Sb
        ierr =psspy.psseinit(50)                    # Inicializa el PSS/e.
        ierr =psspy.case(self.SAV)                  # Abre el archivo ".sav".
        psspy.ierr = psspy.base_frequency(60)       # Define la frecuencia de sistema en 60 Hz.   

        # Se obtiene el archivo de excel.
        workbook = openpyxl.load_workbook(archivo_xlsx)
        # Se obtiene las hojas del excel.
        sheets = workbook.worksheets

        # Itera sobre las filas y columnas de la hoja de excel.
        for row in sheets[0].iter_rows(min_row=2):
            # Iterador para obtener Vb y Sb.
            i=0    
            if row[1].value != None:
                # Obtiene el numero de bus PV  (primera columna).  
                num_bus = self.numero_bus(row[1].value)
                self.Buses_PV.append(int(num_bus)) 
                # Obtiene la tensión base en kV (novena columna).                             
                self.Vbase.append(float(row[8].value))
                # Obtiene la potencia aparente base en MVA.                
                self.Sbase.append(psspy.sysmva())                         
                # Obtiene la potencia activa inicial en MW (tercer columna).
                self.Pinicial.append(float(row[2].value)) 
                # Obtiene la potencia activa final en MW (cuarta columna).
                self.Pfinal.append(float(row[3].value))
                # Obtiene la diferencia de potencias activas en pu (quinta columna)
                deltaP = self.valores_pu(i = i, P_MW = float(row[4].value))
                self.DeltaP.append(deltaP)
                # Obtiene la tensión inicial en pu (sexta columna)
                self.Vinicial.append(float(row[5].value))                  
                # Obtiene la tensión final en pu (séptima columna)
                self.Vfinal.append(float(row[6].value))   
                # Obtiene la diferencia de tensión en pu (octava columna)
                deltaV = self.valores_pu(i = i, V_kV= float(row[7].value))
                self.DeltaV.append(deltaV)
                                     
                i += 1

        # Se asume que solo hay un generador por cada bus, ID = 1
        for i in range(len(self.Buses_PV)):
            self.ID.append(1)
           
    
    def numero_bus(self, cadena:str):
        """"
        Limpia en el archivo Excel la información de los buses para obtener solo su número de identificación.
        """
        return re.sub("^g", "", cadena)
    

    def valores_pu(self,i: int, V_kV = None, P_MW = None):
        """
        Transforma los valores de kV y MW a pu.
        """
        # Tensión base en kV.
        Vb = self.Vbase
        # Potencia aparente base en MVA.
        Sb = self.Sbase
        if V_kV:
            # Tension del generador a pu.
            V_pu = (V_kV)/(Vb[i])
            return V_pu
        if P_MW:
            # Potencia activa del generador a pu.
            P_pu =(P_MW)/(Sb[i])
            return P_pu
        
    def simulacion_PuntoA(self):
        """
        Realiza la simulación estática para el punto de inicio.
        """
        # Archivos para la simulación.
        SAV = self.SAV                  
        DYR = self.DYR
        self.OUT_A = "SalidaA.out"

        # Inicializa el PSS/e y abre el caso base.
        ierr =psspy.psseinit(50)                    # Inicializa el PSS/e.
        ierr =psspy.case(SAV)                       # Abre el archivo ".sav".
        psspy.ierr = psspy.base_frequency(60)       # Define la frecuencia de sistema en 60 Hz.

        # Corre 3 flujos de carga para "ajustar" las condiciones iniciales de cada barra.
        ierr =psspy.fnsl([0,0,0,1,1,0,0,0])
        ierr =psspy.fnsl([0,0,0,1,1,0,0,0])      
        ierr =psspy.fnsl([0,0,0,1,1,0,0,0])

        # Convierte generadores y cargas (Z constante para P y Z constante para Q).
        ierr =psspy.cong(0)
        ierr =psspy.conl(0,1,1,[0,0],[0.0,100.0,0.0,100.0])
        ierr =psspy.conl(0,1,2,[0,0],[0.0,100.0,0.0,100.0])
        ierr =psspy.conl(0,1,3,[0,0],[0.0,100.0,0.0,100.0])

        # Factoriza la Matriz de Admitancias. Solución para estudios de switcheo.
        ierr =psspy.fact()    
        ierr =psspy.tysl(0)

        # Abre el archivo dinámico ".dyr".
        ierr =psspy.dyre_new([1,1,1,1], DYR,"","","")

        # Configura las características de la simulación dinámica.
        tstep=0.005
        ierr =psspy.dynamics_solution_param_2(intgar=[_i,_i,_i,_i,_i,_i,_i,_i],realar=[_f,_f,tstep,tstep,_f,_f,_f,_f])
        # Creación de los canales de potencia activa y tensión en terminales de los generadores.
        for i in range(len(self.Buses_PV)):
            # Configura los canales de salida de la simulación.
             ierr = psspy.machine_array_channel([i*2+1,2,self.Buses_PV[i]],r"1","")    # Canal 1: Potenca activa de cada generador
             ierr = psspy.voltage_channel([i*2+2,-1,-1,self.Buses_PV[i]],)             # Canala 2: Tensión en bornes de cada generador

        # Inicializa la simulación.
        ierr =psspy.strt(0, self.OUT_A)

        # Si la inicialización no se realizó correctamente detiene ejecución del programa.
        if psspy.okstrt()!=0:
            exit()

        # Corre la simulación dinámica.
        ierr =psspy.run(0, 10, 1000, 1, 1)
 
 
    def valores_dinamicos(self, bus: int, id: str, modelo: str, variable:str):

        """
        Retorna la lista completa de los valores de los modelos de la máquina:
        
        tipo:
        'GEN', 'COMP', 'STAB', 'EXC', 'GOV', 'TLC', 'MINXL', 'MAXXL'
        variable:
        'CON', 'STATE', 'VAR', 'ICON'
        
        """
        # Obtiene los indices del modelo relacionado a la planta.
        ierr, indice = psspy.mdlind(bus, id, modelo, variable)
        # Si hay error, retorne una lista vacía.
        if ierr>0: return []

        # Obtiene la cantidad de valores utilizado por el modelo.
        ierr, ncons = psspy.mdlind(bus, id, modelo, 'N'+variable)
        valores = []
        # Obtiene los valores del modelo especificado.
        for numero_cte in range(ncons):
            # Obtiene los valores dinámicos enteros.
            if variable=='ICON':
                ierr, valor = psspy.dsival(variable, indice+numero_cte)
                # Obtiene los valores dinámicos tipo carácter.
                if ierr==3: ierr, valor = psspy.dscval(variable, indice+numero_cte)
            # Obtiene los valores dinámicos reales.
            else: ierr, valor = psspy.dsrval(variable, indice+numero_cte)
            # Si no hay error encontrado.
            if ierr==0: valores.append(valor)
        return valores
    
    def obtener_Vref(self):
        """
        Obtiene el valor de la tensión de referencia para ser utilizado en la función increment_vref de psspy.
        """
        for bus in range(len(self.Buses_PV)):
            # Obtiene los estados del modelo de excitación o AVR.
            estados = self.valores_dinamicos(bus = int(self.Buses_PV[bus]), id = str(self.ID[bus]), modelo = "EXC", variable = "STATE")
            VR = estados[2]                                   # Tensión de campo del modelo de excitación.
            constantes = self.valores_dinamicos(bus = int(self.Buses_PV[bus]), id = str(self.ID[bus]), modelo = "EXC", variable = "CON")
            KA = constantes[1]                                # Ganancia del modelo de excitación.
            ierr, Vt = psspy.agenbuscplx(sid = -1, flag = 5, string = "VOLTAGE")    # Obtiene la tensión en terminal de todas las barras.
            EC = abs(Vt[0][self.Buses_PV[bus]-1])             # Tensión en terminal del generador.
            self.Vref.append((VR/KA) + EC)                    # Tensión de referencia por medio del estudio del diagrama de bloques.


    def obtener_Gref(self):
        """
        Obtiene el valor de la velocidad de referencia para ser utilizado en la función increment_gref de psspy.
        """
        # Obtener las variables del modelo del gobernador.
        for bus in range(len(self.Buses_PV)):
            variables = self.valores_dinamicos(bus = self.Buses_PV[bus], id = str(self.ID[bus]), modelo = "GOV", variable = "VAR")
            self.Gref.append(variables[0])         # Velocidad de referencia.


    def simulacion_PuntoB(self):
        """
        Realiza la simulación para el punto final.
        """
        # Archivos para la simulación.
        SAV = self.SAV
        DYR = self.DYR
        OUT = "SalidasB.out"   
        
        # Inicializa el PSS/e y abre el caso base.
        ierr = psspy.psseinit(50)                # Inicializa el PSS/e.
        ierr = psspy.case(SAV)                   # Abre el archivo ".sav".
        psspy.ierr = psspy.base_frequency(60)   # Define la frecuencia de sistema en 60 Hz.
        
        for bus in range(len(self.Buses_PV)):
            # Cambio en la potencia activa del generador
            psspy.machine_chng_3(ibus = self.Buses_PV[bus],
                                  id = str(self.ID[bus]),
                                  intgar = [_i, _i, _i, _i, _i, _i, _i],
                                  realar = [self.Pfinal[bus], _f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

            # Cambio de la tension en bornes del generador.
            psspy.plant_chng_4(ibus = self.Buses_PV[bus],
                                inode = 0 , 
                                intgar = [_i, _i], 
                                realar = [self.Vfinal[bus], _f])

        # Corre 3 flujos de potencia para "ajustar" las condiciones iniciales de cada barra.
        ierr = psspy.fnsl([0,0,0,1,1,0,0,0])
        ierr = psspy.fnsl([0,0,0,1,1,0,0,0])      
        ierr = psspy.fnsl([0,0,0,1,1,0,0,0])

        # Convierte generadores y cargas (Z constante para P y Z constante para Q).
        ierr = psspy.cong(0)
        ierr = psspy.conl(0,1,1,[0,0],[0.0,100.0,0.0,100.0])
        ierr = psspy.conl(0,1,2,[0,0],[0.0,100.0,0.0,100.0])
        ierr = psspy.conl(0,1,3,[0,0],[0.0,100.0,0.0,100.0])

        # Factoriza la Matriz de Admitancias. Solución para estudios de switcheo.
        ierr = psspy.fact()    
        ierr = psspy.tysl(0)

        # Abre el archivo dinámico ".dyr".
        ierr = psspy.dyre_new([1,1,1,1], DYR,"","","")

        # Configura las características de la simulación dinámica.
        tstep = 0.005
        ierr =psspy.dynamics_solution_param_2(intgar=[_i,_i,_i,_i,_i,_i,_i,_i],realar=[_f,_f,tstep,tstep,_f,_f,_f,_f])

        # Configura los canales de salida de la simulación.
        ierr = psspy.machine_array_channel([bus*2+1,2,self.Buses_PV[bus]],r"""1""","")         # Canal 1: Potenca activa de cada generador
        ierr = psspy.voltage_channel([bus*2+2,-1,-1,self.Buses_PV[bus]],)                      # Canala 2: Tensión en bornes de cada generador
  
        # Inicializa la simulación.
        ierr =psspy.strt(0, OUT)

        # Si la inicialización no se realizó correctamente detiene ejecución del programa.
        if psspy.okstrt()!=0:
            exit()

        # Corre la simulación dinámica.
        ierr =psspy.run(0, 10, 1000, 1, 1)
        self.obtener_Vref()
        self.obtener_Gref()


    def obtener_deltas(self):
        """
        Obtiene los cambios de la tensión de referencia y velocidad de referencia\n
        ante el cambio en la potencia activa o de la tensión en los bornes del generador.
        """
        
        # Cambio en la tensión de referencia.
        for bus in range(len(self.Buses_PV)):
            self.DeltaVref.append(self.Vref[bus*1 + len(self.Buses_PV)] - self.Vref[bus*1])
            # Cambio en la velocidad de referencia.
            self.DeltaGref.append(self.Gref[bus*1 + len(self.Buses_PV)] - self.Gref[bus*1])
        

    def incremento(self):
        """
        Se realiza un incremento en la tensión de referencia o la velocidad de referencia\n
        según sea el cambio en la variable eléctrica del generador.
        """
        # Para cada generador
        for bus in range(len(self.Buses_PV)):
            # Se corre la simulación en el punto inicial
            self.simulacion_PuntoA()
            # Incremento de Vref y Gref.
            psspy.increment_gref(self.Buses_PV[bus], r"""1""", self.DeltaGref[bus])
            psspy.increment_vref(self.Buses_PV[bus], r"""1""", self.DeltaVref[bus])

            # Corre la simulación dinámica.
            ierr =psspy.run(0, 50, 1000, 1, 1)

            # Exporta la información del archivo .out.
            chnfobj = dyntools.CHNF(self.OUT_A)
            short_title, chanid, chandata = chnfobj.get_data()

            # Obtiene las variables de interés.
            self.Tgrafica.append(chandata['time'])          # Tiempo (s)
            self.Pgrafica.append(chandata[bus*2+1])         # Potencia activa del generador (pu)
            self.Vgrafica.append(chandata[bus*2+2])         # Tension terminal del generador (pu)
        # Elimina los archivos .out para proximas simulaciones.  
        os.remove("SalidaA.out")
        os.remove("SalidasB.out")
        # Grafica las potencias de los generadores.
        for bus in range(len(self.Buses_PV)):
             plt.plot(self.Tgrafica[bus], self.Pgrafica[bus],linestyle='-', linewidth = 1, label =f"Generador {self.Buses_PV[bus]}")
        plt.autoscale()
        plt.title("Potencia activa en función al tiempo")
        plt.grid(linestyle='--', color='grey',linewidth=0.5)
        # Configuración de los ejes.
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Potencia activa (pu)")
        plt.legend()
        plt.show()

        # Grafica las tensiones de los generadores.
        for bus in range(len(self.Buses_PV)):
             plt.plot(self.Tgrafica[bus], self.Vgrafica[bus],linestyle='-', linewidth = 1, label =f"Generador {self.Buses_PV[bus]}")
        plt.autoscale()
        plt.title("Tensión en terminal en función al tiempo")
        plt.grid(linestyle='--', color='grey',linewidth=0.5)
        # Configuración de los ejes.
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Tensión(pu)")
        plt.legend(loc= "upper right")
        plt.show()
        


if __name__ == "__main__":

    #------------------------#
    # Ejecución del programa #
    #------------------------#
    Sys = Sistema()
    # Simulacion al abrir linea 16-17
    #Sys.importar_datos(archivo_sav="NewEngland39.sav", archivo_dyr="Caso_NO_BESS.dyr", archivo_xlsx= "NewEngland16_17.xlsx")

    # Simulacion al abrir linea 26-29
    Sys.importar_datos(archivo_sav="NewEngland39.sav", archivo_dyr="Caso_NO_BESS.dyr", archivo_xlsx= "NewEngland26_29.xlsx")
    
    # Simulacion al abrir linea 01-39
    #Sys.importar_datos(archivo_sav="NewEngland39.sav", archivo_dyr="Caso_NO_BESS.dyr", archivo_xlsx= "NewEngland01_39.xlsx")

    Sys.simulacion_PuntoA() 
    Sys.obtener_Vref()   
    Sys.obtener_Gref()
    Sys.simulacion_PuntoB()
    Sys.obtener_deltas()
    Sys.incremento()