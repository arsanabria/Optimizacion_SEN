"""
Realizado por: MSc. Francisco Escobar Prado
Colaboración: Aroon Sanabria Torres

Descripción del programa: Del programa original que optimiza el redespacho de generación ante el cierre
de una línea de transmisión, se realiza una modificación para minimizar los cambios de potencia activa y
tensión de los generadores bajo restricciones de operación de cada una de las unidades generadoras.
Además crea un archivo .xlsx para obtener las variables de interés y realizar la simulación del SEP.
"""

import pf_static
import records
from collections.abc import Sequence
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import numpy as np
import pandas as pd
import openpyxl


class OptimizableSystem(pf_static.StaticSystem):
    pass

    def define_redispatchable_generators(
        self, generator_names: Sequence
    ) -> None:
        """
        Define the genenerators whose active power can be redispatched.
        """

        self.redispatchable_generators: list[records.Generator] = []
        self.BusesA  = []

        for generator_name in generator_names:
            # We first fetch the associated object:
            generator = self.gen_dict[generator_name]

            # A redispatchable generator cannot be connected to the slack
            # bus, as otherwise we would loose a degree of freedom.
            if generator.bus is self.slack:
                raise ValueError("Cannot redispatch slack generator.")

            # We then store the generator and store its initial power output
            # to use it when evaluating the cost function.
            self.redispatchable_generators.append(generator)
            generator.PG_MW_0 = generator.PG_MW
            generator.V_pu_0 = generator.bus.V_pu




    def specify_synchrocheck(
        self,
        line_name: str,
        bus_name: str,
        V_pu_tol: float,
        theta_degrees_tol: float,
        name_file_xlsx: str
    ) -> None:
        """
        Specify the synchrocheck under study, as well as its settings.
        """

        # To simulate an open line, we will add a fictitious bus that represents
        # the breaker pole that is opposite to the specified bus. The
        # optimization will then try to bring the voltages of those two buses
        # closer together.
        
        self.file_xlsx = name_file_xlsx

        line = self.line_dict[line_name]
        bus = self.bus_dict[bus_name]

        self.SC_bus = bus
        self.fictitious_bus = self.add_PQ(
            PL=0, QL=0, name=f"open end of {line_name}", base_kV=bus.base_kV
        )

        if bus is line.from_bus:
            line.from_bus = self.fictitious_bus
        elif bus is line.to_bus:
            line.to_bus = self.fictitious_bus
        else:
            raise ValueError("The line does not touch the specified bus.")

        # We then save the settings of the synchrocheck.
        self.V_pu_tol = V_pu_tol
        self.theta_degrees_tol = theta_degrees_tol

    def set_generator_limits(
        self, generator_name: str, PG_MW_min: float, PG_MW_max: float
    ) -> None:
        """
        Set the active power limits of a generator.
        """

        self.gen_dict[generator_name].PG_MW_min = PG_MW_min
        self.gen_dict[generator_name].PG_MW_max = PG_MW_max

    def set_bus_limits(
        self, bus_name: str, V_pu_min: float, V_pu_max: float
    ) -> None:
        """
        Set the voltage limits of a bus.
        """

        self.bus_dict[bus_name].V_pu_min = V_pu_min
        self.bus_dict[bus_name].V_pu_max = V_pu_max


    def print_problem_formulation(self) -> None:
        """
        Display the OPF problem formulation.

        The message has the form:

            You want to solve the following OPF problem:

                minimize the sum of: (PG of j - initial PG of j)^2
                over all generators j

                subject to:

                    * PG_min of j <= PG of j <= PG_max of j  for every gen. j,
                    * V_min of i <= V of i <= V_max of i     for every bus  i,
                    * abs(V_pu difference across breaker) <= V_pu_tol,
                    * abs(theta difference across breaker) <= theta_degrees_tol,
                    * and, implicitly, the power flow equations.
        """

        message = "\nYou want to solve the following OPF problem:\n\n"

        message += "    minimize the sum of: (PG of j - initial PG of j)^2\n"
        message += "    over all generators j\n\n"

        message += "    subject to:\n\n"

        for generator in self.redispatchable_generators:
            message += (
                f"        {generator.PG_MW_min:.1f} MW "
                f"<= PG of {generator.name} "
                f"<= {generator.PG_MW_max:.1f} MW\n"
            )

        message += "\n"
        for bus in self.buses:
            message += (
                f"        {bus.V_pu_min:.2f} pu "
                f"<= V of {bus.name} "
                f"<= {bus.V_pu_max:.2f}\n"
            )

        message += "\n"
        message += (
            f"        "
            f"abs(V of {self.fictitious_bus.name} "
            f"- "
            f"V of {self.SC_bus.name}) <= {self.V_pu_tol:.2f} pu\n"
        )
        message += (
            f"        "
            f"abs(theta of {self.fictitious_bus.name} "
            f"- "
            f"theta of {self.SC_bus.name}) "
            f"<= {self.theta_degrees_tol:.2f} degrees\n"
        )
        message += "\n"
        
        message += "        Initial Values\n\n"
        message += "\n"
        message += "        and, implicitly, the power flow equations. ;)\n\n"
        print(message)

        input("Press Enter to continue...")

    def find_optimal_redispatch(self, power_flow_tol: float = 1e-9) -> None:
        """
        Solve the OPF problem and display optimization results.
        """

        # Print the problem formulation:
        self.print_problem_formulation()

        # Define the cost function:
        def fun(x: np.ndarray) -> float:
            cost1 = 0 
            cost2 = 0
            KP = 0.001
            Kv = 1
            for new_PG_MW, generator in zip(x[:len(x)//2], self.redispatchable_generators):
                cost1 += KP*(new_PG_MW - generator.PG_MW_0) ** 2
                
            for new_VPU, generator in zip(x[len(x)//2:], self.redispatchable_generators):
                cost2 += Kv*(new_VPU - generator.V_pu_0) ** 2
               
            cost = cost1 + cost2

            return cost

        # Define the linear constraints on generation output, i.e. lb <= Ax <= ub:
        lbP = np.array(
            [
                generator.PG_MW_min
                for generator in self.redispatchable_generators
            ]
        )
        lbV = np.array(
            [   
                
                generator.bus.V_min_pu
                for generator in self.redispatchable_generators
            ]
        )
        lb = np.concatenate([lbP, lbV])


        ubP = np.array(
            [
                generator.PG_MW_max
                for generator in self.redispatchable_generators
            ]
        )
        ubV = np.array(
            [   
            
                generator.bus.V_max_pu
                for generator in self.redispatchable_generators
            ]
        )
        
        ub = np.concatenate([ubP, ubV])

        A = np.eye(len(ub))
        LC = LinearConstraint(A=A, lb=lb, ub=ub)
        
        # Define the nonlinear constraints on voltage magnitude and angle:
        def voltage_magnitudes(x: np.ndarray) -> np.ndarray:
            """
            Return the voltage magnitudes of all buses.
            """
    
            # We first update the power output of the redispatchable generators:
            for new_PG_MW, generator in zip(x[:len(x)//2], self.redispatchable_generators):
                generator.PG_MW = new_PG_MW

            for new_VPU, generator in zip(x[len(x)//2:], self.redispatchable_generators):
                generator.bus.V_pu= new_VPU

            # We then run the power flow:
            self.run_pf(tol=power_flow_tol)

            # We finally return the voltage magnitudes:
            return np.array([bus.V_pu for bus in self.buses])
        
        

        def differences_across_breaker() -> np.ndarray:
            """
            Return the voltage magnitude and angle differences across the breaker.

            We take advantage of the fact that the power flow has already been
            run when this function is called.
            """

            delta_V = np.abs(self.fictitious_bus.V_pu - self.SC_bus.V_pu)
            delta_theta_degrees = np.abs(
                np.rad2deg(self.fictitious_bus.theta_radians)
                - np.rad2deg(self.SC_bus.theta_radians)
            )

            # We then return the differences:
            return np.array([delta_V, delta_theta_degrees])

        def NLC_fun(x: np.ndarray) -> np.ndarray:
            """
            This simply stacks the outputs of the two functions above.
            """

            mag = voltage_magnitudes(x)
            diff = differences_across_breaker()

            return np.concatenate([mag, diff])

        # The last two numbers are the bounds for the differences across the
        # breaker:
        lb = np.array([bus.V_pu_min for bus in self.buses] + [0, 0])
      
        ub = np.array(
            [bus.V_pu_max for bus in self.buses]
            + [self.V_pu_tol, self.theta_degrees_tol]
        )
       

        NLC = NonlinearConstraint(fun=NLC_fun, lb=lb, ub=ub)

        # Define the initial guess (initial dispatch):
        x0P = np.array(
            [generator.PG_MW for generator in self.redispatchable_generators]
        )
        x0V = np.array(
           [generator.bus.V_pu for generator in self.redispatchable_generators]
        ) 
        
        x0 = np.concatenate([x0P,x0V]) 


        # Solve the problem (I just played around with the solver options until
        # it worked)
        res = minimize(
            x0=x0,
            fun=fun,
            constraints=[LC, NLC], 
            tol=1e-4,
            method="SLSQP",
            options={
                "ftol": 1e-4,
                "eps": 1e-3,
            },
        )
         # Print the results:
        self.print_redispatch()
        self.create_excel()

        # Display the new operating point:
        print("\nThe new operating point is:")
        print(self)

    def create_excel(self) -> None:
        """
        Create a excel with the results of the optimization.
        """
        # Bus of each generator.
        Bus = np.array([
            generator.bus.name for generator in self.redispatchable_generators])
        # Values of generator active power.
        Pinitial = np.array(
            [generator.PG_MW_0 for generator in self.redispatchable_generators])
        Pfinal = np.array(
            [generator.PG_MW for generator in self.redispatchable_generators])
        DeltaP = []
        for i in range(len(Pfinal)):
            DeltaP.append(Pfinal[i] -  Pinitial[i])
        # Values of the generator voltage.
        Vinitial = np.array([
            generator.V_pu_0 for generator in self.redispatchable_generators])
        Vfinal =np.array([
            generator.bus.V_pu for generator in self.redispatchable_generators])
        DeltaV = []
        for i in range(len(Vinitial)):
            DeltaV.append(Vfinal[i] -  Vinitial[i])
        # Voltage base
        Vb =np.array([
            generator.bus.base_kV for generator in self.redispatchable_generators])
        # Power base MVA
        Sb =np.array([
            generator.bus.base_kV for generator in self.redispatchable_generators])

        # Create a excel file.
        df = pd.DataFrame({"Gen ": Bus, "Pi (MW)": Pinitial, "Pf (MW)": Pfinal, "Delta P (MW)": DeltaP, 
                           "Vi (kV)": Vinitial, "Vf (kV)": Vfinal, "Delta V (kV)": DeltaV, "Vb (kV)": Vb,
                           "V mag across breaker (pu)": self.V_line, " Angle across breaker (degrees)":self.Ang_line} )
        print(self.file_xlsx)
        df.to_excel(self.file_xlsx)

        

    def print_redispatch(self) -> None:
        """
        Display the redispatch results.
        """

        # Print the changes: PG_MW_0 -> PG_MW (delta = PG_MW - PG_MW_0)
        print("\nThe redispatch of active power is as follows:\n\n")
        for generator in self.redispatchable_generators:
            print(
                f"{generator.name}: "
                f"{generator.PG_MW_0:.1f} MW -> {generator.PG_MW:.1f} MW "
                f"(delta = { generator.PG_MW - generator.PG_MW_0:.1f} MW)"
            )

        # Print the magnitude and angle differences across the breaker:
        delta_V = np.abs(self.fictitious_bus.V_pu - self.SC_bus.V_pu)
        delta_theta_degrees = np.abs(
            np.rad2deg(self.fictitious_bus.theta_radians)
            - np.rad2deg(self.SC_bus.theta_radians)
        )
        
        print(
            f"\nVoltage magnitude difference across breaker: {delta_V:.2f} pu"
        )

        print(
            f"Voltage angle difference across breaker: "
            f"{delta_theta_degrees:.2f} degrees"
        )
        
        self.V_line = delta_V
        self.Ang_line = delta_theta_degrees

       
        print("\nThe redispatch of voltage is as follows:\n\n")
        for generator in self.redispatchable_generators:
            print(
                f"{generator.name}: "
                f"{generator.V_pu_0} pu -> {round(generator.bus.V_pu,12)} pu "
                f"(delta = {round(generator.bus.V_pu- generator.V_pu_0,4)} pu)"
            ) 
        


if __name__ == "__main__":
    # We test the code with the Nordic test system, which gives us enough
    # degrees of freedom to play with.

    nordic = OptimizableSystem.import_ARTERE(
        filename="NewEngland39.dat",
        system_name="New England",
        base_MVA=100,
        use_injectors=True,
    )
    
    # nordic.run_pf()
    # print(nordic)
    # exit()
    # We then assume that the line 1011-1013 is open and that we want to close
    # the circuit breaker at bus 1013. We choose this line because it is not
    # critical and will not give convergence issues. Regarding the synchrocheck,
    # we assume that it tolerates a voltage magnitude difference of 5% and a
    # voltage angle difference of 10 degrees.  These are the values currently
    # used by the Costa Rican transmission system operator (ICE).

    nordic.specify_synchrocheck(
        line_name="26-29",
        bus_name="26",
        V_pu_tol=0.1,
        theta_degrees_tol=10,
        name_file_xlsx = "NewEngland26_29.xlsx"
    )

    # Finally, we set the voltage limits of the buses. As per Costa Rican
    # regulations, we assume 0.9 pu - 1.1 pu for transmission buses and 0.95 pu
    # - 1.05 pu for distribution buses. A bus belongs to the transmission
    # network (for our purposes) if it does not feed any load. (The
    # expression below is ugly, sorry.)
    distribution_buses = {
        bus
        for bus in nordic.buses
        if any(
            inj.bus is bus
            for inj in nordic.injectors
            if isinstance(inj, records.Load)
        )
    }
    
    generator_names = [f"g{i}" for i in range(30, 40) if i != 31]                           # Number of generators PV
    nordic.define_redispatchable_generators(generator_names=generator_names)    # Redispatchable generators
    
 
    Pgen = []                       # Specific active power of each generator
    
    for generator_name in generator_names:              # Obtain the active power of each generator
        generator = nordic.gen_dict[generator_name]
        Pgen.append(generator.PG_MW)    
        
       
    for generator_name  in generator_names:
        # Fetch the generator:
        generator = nordic.gen_dict[generator_name]
        # Infer power limits according to the assumptions above:
        current_output = generator.PG_MW
        PG_MW_max = current_output / 0.8
        PG_MW_min = 0.6 * PG_MW_max
        # Set the limits:
        nordic.set_generator_limits(
            generator_name=generator_name,
            PG_MW_min=PG_MW_min,
            PG_MW_max=PG_MW_max,
        )

    
    transmission_buses = set(nordic.buses) - distribution_buses

    for bus in distribution_buses:
        nordic.set_bus_limits(bus_name=bus.name, V_pu_min=0.95, V_pu_max=1.05)

    for bus in transmission_buses:
        nordic.set_bus_limits(bus_name=bus.name, V_pu_min=0.9, V_pu_max=1.1)

    # We can now run the redispatch:
    nordic.find_optimal_redispatch(power_flow_tol=1e-9)
    
