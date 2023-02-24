import cantera as ct


class Settings:

    def __init__(self, fuel, P_therm, phi, T_heater, vdot_N2,H2_perc):
        self.P_therm = P_therm  # kW
        self.fuel = fuel  # 'CH4'/'H2'/'CH4+H2'
        self.LHV = 0
        self.phi = phi
        self.AF_stoic = 0
        self.O2_perc = 0.0
        self.mdot_fuel = 0  # kg/s
        self.mdot_fuel_secondary=0 #kg/s
        self.mdot_air = 0 # kg/s
        self.mdot_N2 = 0 # kg/s
        self.mdot_CO2 = 0  # kg/s
        self.N2dilperc=100.0 # volume percentage of N2 amongst diluent= 100: all diluents are N2, 0=All diluent composition is CO2
        self.mdot_total = 0 #kg/s
        self.vdot_fuel = 0 # lnpm
        self.vdot_air = 0 # lnpm
        self.vdot_ch4 = 0 #lnpm
        self.vdot_h2 = 0 #lnpm
        self.vdot_N2 =vdot_N2 # lnpm
        self.vdot_CO2 = 0.0 # lnpm
        self.T_air = T_heater
        self.T_fuel =273.15 + 15
        self.rho_in = 0
        self.T_in = 0
        self.Q_in =0
        self.Y_comp = {}
        self.X_comp={}
        self.enthalpy_inlet = 0
        self.enthalpy_outlet =0
        self.exhaust_gas = 0
        self.H2_perc = H2_perc
        self.X_CH4 = 0
        self.X_H2 = 0
        self.solve_reactor='on'
        self.main()


    def LHV_CH4_calc(self, gas):
        # CH4 + 2 O2 = CO2 + 2 H2O
        gas.TPX = 298, ct.one_atm, 'CH4:1, O2:2'
        h1 = gas.enthalpy_mass
        Y_CH4 = gas['CH4'].Y[0]  # returns an array, of which we only want the first element

        # set state to complete combustion products without changing T or P
        gas.TPX = None, None, 'CO2:1, H2O:2'
        h2 = gas.enthalpy_mass
        lhv = -(h2 - h1) / Y_CH4 / 1e6
        #print('LHV = {:.3f} MJ/kg'.format(-(h2 - h1) / Y_CH4 / 1e6))

        return lhv

    def LHV_H2_calc(self, gas):
        # 2 H2 + O2 = 2 H2O
        gas.TPX = 298, ct.one_atm, 'H2:2, O2:1'
        h1 = gas.enthalpy_mass
        Y_H2 = gas['H2'].Y[0]  # returns an array, of which we only want the first element

        # set state to complete combustion products without changing T or P
        gas.TPX = None, None, 'H2O:2'
        h2 = gas.enthalpy_mass
        lhv = -(h2 - h1) / Y_H2 / 1e6
        #print('LHV = {:.3f} MJ/kg'.format(-(h2 - h1) / Y_H2 / 1e6))

        return lhv

    def main(self):
        gas = ct.Solution('gri30.cti')
        N2_diluent = ct.Solution('gri30.cti')
        CO2_diluent = ct.Solution('gri30.cti')
        air = ct.Solution('air.cti')
        species = gas.species_names
        air_species = air.species_names
        MW_O2 = gas.molecular_weights[species.index('O2')]
        MW_N2 = gas.molecular_weights[species.index('N2')]
        MW_CH4 = gas.molecular_weights[species.index('CH4')]
        MW_H2 = gas.molecular_weights[species.index('H2')]
        if self.fuel=='CH4' or self.fuel=='H2':
            if self.fuel == 'CH4':
                # CH4 + 2 ( O2 + 3.76 N2) = CO2 + 2 H2O + 3.76(2)N2
                self.LHV = self.LHV_CH4_calc(gas)  # MJ/kg
                self.AF_stoic = 2 * (MW_O2 + 3.76 * MW_N2) / MW_CH4
                self.X_CH4 = 1.0

            if self.fuel == 'H2':
                # 2 H2 + ( O2 + 3.76 N2) = 2 H2O + 3.76 N2
                self.LHV = self.LHV_H2_calc(gas)  # MJ/kg
                self.AF_stoic = (MW_O2 + 3.76 * MW_N2) / (2 * MW_H2)
                self.X_H2 = 1.0
            self.mdot_fuel = (self.P_therm/self.LHV)*(1.0/1000.0)  # kg/s
            gas.TPY = 293.15, ct.one_atm, {self.fuel:1.0}  # NTP
            self.vdot_fuel = (self.mdot_fuel/gas.density) * 60000.0  # lnpm
            self.mdot_air = self.AF_stoic * self.mdot_fuel / self.phi
            air.TP = 293.15, ct.one_atm # NTP
            self.vdot_air = (self.mdot_air / air.density) * 60000  # lnpm
            N2_diluent.TPY = 293.15, ct.one_atm, {'N2': 1.0}  # NTP
            self.mdot_N2 = self.vdot_N2 * N2_diluent.density/60000 # kg/s
            self.mdot_total = self.mdot_air + self.mdot_fuel + self.mdot_N2
            air.TP = self.T_air, ct.one_atm
            H_mix = (gas.enthalpy_mass*self.mdot_fuel + air.enthalpy_mass *\
                     self.mdot_air + N2_diluent.enthalpy_mass*self.mdot_N2)/self.mdot_total
            self.enthalpy_inlet = H_mix
            self.Y_comp = {self.fuel: self.mdot_fuel / self.mdot_total,
                           'O2': air.Y[air_species.index('O2')] * self.mdot_air / self.mdot_total,
                           'N2': (air.Y[air_species.index('N2')] * self.mdot_air +self.mdot_N2)/ self.mdot_total,
                           'AR': air.Y[air_species.index('AR')] * self.mdot_air / self.mdot_total}
        else:
            x = self.H2_perc
            self.AF_stoic = ((400.0 - 3.0 * x) / (100.0 * MW_CH4 + x * (MW_H2 - MW_CH4))) * (
                        (air.mean_molecular_weight) / 2.0)
            moles_total =  (self.P_therm / (
                        (self.LHV_CH4_calc(gas) * MW_CH4*((100.0-x)/100.0)) + ((self.LHV_H2_calc(gas) * MW_H2) * x / (100.0 )))) * (
                                    1.0 / 1000.0)
            moles_CH4 =  ((100.0-x)/100.0)*moles_total#(P_therm / (
                        #(self.LHV_CH4_calc(gas) * MW_CH4) + ((self.LHV_H2_calc(gas) * MW_H2) * x / (100.0 - x)))) * (
                        #            1.0 / 1000.0)  # kmoles/s
            moles_H2 = moles_total * x / (100.0)
            self.LHV = (self.LHV_CH4_calc(gas) * MW_CH4 * moles_CH4 + self.LHV_H2_calc(gas) * moles_H2 * MW_H2) / (
                    moles_CH4 * MW_CH4 + moles_H2 * MW_H2)
            self.X_CH4 = moles_CH4 / (moles_CH4 + moles_H2)
            self.X_H2 = moles_H2 / (moles_CH4 + moles_H2)
            gas.TPX = 293.15, ct.one_atm, {'CH4': 1.0}  # NTP
            self.vdot_ch4 = (moles_CH4 * MW_CH4 / gas.density) * 60000
            gas.TPX = 293.15, ct.one_atm, {'H2': 1.0}  # NTP
            self.vdot_h2 = (moles_H2 * MW_H2 / gas.density) * 60000
            self.mdot_fuel = (moles_CH4 * MW_CH4) + (moles_H2 * MW_H2)
            gas.TPX = 293.15, ct.one_atm, {'CH4': moles_CH4, 'H2': moles_H2}  # NTP
            self.vdot_fuel = (self.mdot_fuel / gas.density) * 60000.0  # lnpm
            gas.TP = self.T_fuel, ct.one_atm
            # self.mdot_air = self.AF_stoic * self.mdot_fuel / self.phi
            self.mdot_air = ((400.0 - 3.0 * x) / (100.0)) * moles_total * 4.76 * (air.mean_molecular_weight) / (
                        2.0 * self.phi)
            """((400 - 3 * x) / (100 - x)) * moles_CH4 * 4.76 * (air.mean_molecular_weight) / (
                        2.0 * phi)"""
            air.TP = 293.15, ct.one_atm  # NTP
            self.vdot_air = (self.mdot_air / air.density) * 60000.0  # lnpm
            N2_diluent.TPY = 293.15, ct.one_atm, {'N2': 1.0}  # NTP
            CO2_diluent.TPY = 293.15, ct.one_atm, {'CO2':1.0}  # NTP
            MW_N2 =gas.molecular_weights[species.index('N2')]
            MW_CO2 = gas.molecular_weights[species.index('CO2')]
            if self.O2_perc!=0.0:
                molar_rate_diluent = ((400.0 - 3.0 * x) / (100.0))*moles_total*((100.0-4.76*self.O2_perc)/self.O2_perc)/(2.0*self.phi)
                self.mdot_N2 = MW_N2*(self.N2dilperc/100.0)*molar_rate_diluent
                self.mdot_CO2 = MW_CO2*(1.0-(self.N2dilperc/100.0))*molar_rate_diluent
                self.vdot_N2 = self.mdot_N2 * 60000.0 / N2_diluent.density
                self.vdot_CO2 = self.mdot_CO2 * 60000.0 / CO2_diluent.density
            else:
                self.mdot_N2 = self.vdot_N2 * N2_diluent.density / 60000.0  # kg/s
                self.mdot_CO2 = self.vdot_CO2 * CO2_diluent.density / 60000.0  # kg/s
            N2_diluent.TP = self.T_air, ct.one_atm
            CO2_diluent.TP = self.T_air, ct.one_atm
            self.mdot_total = self.mdot_air + self.mdot_fuel + self.mdot_N2 + self.mdot_CO2
            air.TP = self.T_air, ct.one_atm
            H_mix = (gas.enthalpy_mass * self.mdot_fuel + air.enthalpy_mass * \
                     self.mdot_air + N2_diluent.enthalpy_mass * self.mdot_N2 + CO2_diluent.enthalpy_mass * self.mdot_CO2) / self.mdot_total
            self.enthalpy_inlet = H_mix
            self.Y_comp = {'CH4': moles_CH4 * MW_CH4 / self.mdot_total, 'H2': moles_H2 * MW_H2 / self.mdot_total,
                           'O2': air.Y[air_species.index('O2')] * self.mdot_air / self.mdot_total,
                           'N2': (air.Y[air_species.index('N2')] * self.mdot_air + self.mdot_N2) / self.mdot_total,
                           'CO2': self.mdot_CO2 / self.mdot_total,
                           'AR': air.Y[air_species.index('AR')] * self.mdot_air / self.mdot_total}


        gas.HPY = H_mix, ct.one_atm, self.Y_comp
        self.X_comp = gas.X
        self.rho_in = gas.density
        self.Q_in = H_mix*self.mdot_total
        self.T_in = gas.T

        if self.solve_reactor=='on':
            res1 = ct.Reservoir(gas)
            gas.TPY = 2000,ct.one_atm, self.Y_comp
            react = ct.IdealGasReactor(gas, energy='on')
            vol_comb = (3.14 / 4.0) * (0.215 ** 2.0) * 0.49  # m^3
            react.volume = vol_comb/12.0
            mfc1 = ct.MassFlowController(res1,react,mdot = self.mdot_total)
            res2 = ct.Reservoir(gas)
            mfc2 = ct.MassFlowController(react,res2,mdot = self.mdot_total)
            net = ct.ReactorNet([react])
            dt = 1e-4
            tf = dt
            net.advance_to_steady_state()
            """for i in range(int(5e5)):
                net.advance(tf)
                tf = tf + dt"""

            self.enthalpy_outlet = react.thermo.enthalpy_mass
            self.exhaust_gas = react

            print(react.thermo.T)
            print("O2=",react.thermo.X[species.index('O2')])
            print("CO2=",react.thermo.X[species.index('CO2')])
            print("CO=", react.thermo.X[species.index('CO')])
            print("Volume settings reactor=",react.volume)











