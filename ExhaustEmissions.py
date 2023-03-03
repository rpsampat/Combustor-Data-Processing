import os
import cantera as ct
import numpy as np
import pickle
import matplotlib.pyplot as plt
#import Thermocouple as TC
from labview_extract import DataExtract
from nptdms import TdmsFile
from scandir import scandir


class ExhaustEmissions:
    def __init__(self):
        self.drive = 'O:/'
        self.folder = "FlamelessCombustor_Jan2023/Labview_data/ExpCampH2admix_2023/OldTube_DSLR_UV_GasAnalyser/"
        self.dataset = {}
        self.dataset_temperature = {}
        self.quant_list = ["Measured - Main Air", "Measured - Pilot Air", "Measured - Cooling Air",
                           "Measured - CH4 Main","Measured - H2", "'Raw TC Data'/'TC15'", "'Raw TC Data'/'Main Air'",
                           "'Raw TC Data'/'TC10'",
                           "'Raw TC Data'/'Cooling Air'", "'Raw TC Data'/'Fuel Temp'", "Offset",
                           "Position", "Avg Exhaust Temp", "'Pressure & GA'/'CH4'", "'Pressure & GA'/'CO2'",
                           "'Pressure & GA'/'CO'", "'Pressure & GA'/'NO'", "'Pressure & GA'/'NO2'",
                           "'Pressure & GA'/'O2'"]
        self.quant = {'mdot_air': "Measured - Main Air", 'mdot_pilotair': "Measured - Pilot Air",
                      'mdot_coolair': "Measured - Cooling Air", 'mdot_ch4': "Measured - CH4 Main",
                      'mdot_h2': "Measured - H2",
                      'TC15': "'Raw TC Data'/'TC15'", 'Temp_mainair': "'Raw TC Data'/'Main Air'",
                      'Temp_flange': "'Raw TC Data'/'TC10'", 'Temp_coolair': "'Raw TC Data'/'Cooling Air'",
                      'Temp_fuel': "'Raw TC Data'/'Fuel Temp'", 'y_pos': "'Traverse System'/'Offset'",
                      'x_pos': "'Traverse System'/'Position'",
                      'T_exhaust': "Avg Exhaust Temp", 'CH4': "'Pressure & GA'/'CH4'", 'CO2': "'Pressure & GA'/'CO2'",
                      'CO': "'Pressure & GA'/'CO'", 'NO': "'Pressure & GA'/'NO'", 'NO2': "'Pressure & GA'/'NO2'",
                      'O2': "'Pressure & GA'/'O2'"}
        gas = ct.Solution('gri30.yaml')
        N2_diluent = ct.Solution('gri30.yaml')
        CO2_diluent = ct.Solution('gri30.yaml')
        air = ct.Solution('air.yaml')
        species = gas.species_names
        air_species = air.species_names
        gas.TPX = 293.15, ct.one_atm, {'CH4': 1.0}  # NTP
        self.rho_n_CH4 = gas.density
        gas.TPX = 293.15, ct.one_atm, {'H2': 1.0}  # NTP
        self.rho_n_H2 = gas.density
        air.TP = 293.15, ct.one_atm  # NTP
        self.rho_n_air = air.density

        self.MW_air = 28.97 #kg/kmol
        self.MW_O2 = gas.molecular_weights[species.index('O2')]
        self.MW_N2 = gas.molecular_weights[species.index('N2')]
        self.MW_CH4 = gas.molecular_weights[species.index('CH4')]
        self.MW_H2 = gas.molecular_weights[species.index('H2')]


    def read_tdms(self,file):
        tdms_file = TdmsFile.read(file)
        group = tdms_file._channel_data
        return group

    def key_extract(self, header, group):
        for i in group.keys():
            res = header in i
            if res == True:
                break

        return i

    def tdms_file_search(self):
        path = self.drive + self.folder
        identifiers=["NV100_","H2_","_phi",".tdms"]
        identifier_exclude = ["_index"]
        subdir_list = next(os.walk(path))[1] # list of immediate subdirectories within the steel chamber data directory
        data={}
        for subdir in subdir_list:
            path_file = path+'/'+subdir
            filenames = next(os.walk(path_file))[2]
            print(filenames)
            for names in filenames:
                check_id = [(x in names) for x in identifiers]
                check_id_exclude = [(x in names) for x in identifier_exclude]
                isfalse = False in check_id
                isfalse_exclude = False in check_id_exclude
                if not(isfalse) and isfalse_exclude:
                    data_dict={}
                    group = self.read_tdms(path_file + '/' + names)
                    for q in self.quant:
                        key = self.key_extract(self.quant[q], group)
                        data_dict[q] = group[key].data
                    data[names] = data_dict
        with open(self.drive + self.folder + "Unified_dataset_ExhaustEmissions", 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def spec_corr(self,X_O2, X_CO,X_CO2,X_CH4, X_NO, X_NO2,excess_o2, n_o2_calc, n_o2_excess, n_t_dry, n_h2o):
        x_O2 = 15 # % O2 corrected
        n_t_wet = n_t_dry + n_h2o
        X_O2_red_dry = n_o2_calc/(n_t_dry-4.76*n_o2_excess)
        X_O2_red_wet = n_o2_calc/(n_t_wet-4.76*n_o2_excess)
        corr_fact_dry = (n_t_dry/(n_t_dry-4.76*n_o2_excess))*(20.9-(x_O2/100.0))/(20.9-X_O2_red_dry)
        corr_fact_wet = (n_t_wet / (n_t_wet - 4.76 * n_o2_excess)) * (20.9 - (x_O2 / 100.0)) / (20.9 - X_O2_red_wet)
        X_CO_corr = X_CO*corr_fact_dry*1e6 # dry at x% O2 (ppm)
        X_CO2_corr = X_CO2 * corr_fact_dry*1e2 # dry at x% O2 (%)
        X_CH4_corr = X_CH4 * corr_fact_dry*1e6 # dry at x% O2 (ppm)
        wet = 'y'
        if wet == 'y':
            X_NO_corr = X_NO * corr_fact_wet * 1e6  # (X_NO*n_t_wet/n_corr)*1e6 # dry at x% O2 (ppm)
            X_NO2_corr = X_NO2 * corr_fact_wet * 1e6  # (X_NO2 * n_t_wet / n_corr)*1e6 # dry at x% O2 (ppm)
        else:
            X_NO_corr = (X_NO * corr_fact_dry) * 1e6  # dry at x% O2 (ppm)
            X_NO2_corr = (X_NO2 * corr_fact_dry) * 1e6  # dry at x% O2 (ppm)

        return X_CO_corr, X_CO2_corr, X_CH4_corr, X_NO_corr, X_NO2_corr







    def excess_O2_archive(self,X_O2, X_CO,X_CO2,X_CH4, mdot_CH4, mdot_H2, mdot_air_main, mdot_air_pilot):
        """
        Calculate excess O2 % based on measured dry values of carbon species, inlet fuel and air mass flows.
        :param X_O2:
        :param X_CO:
        :param X_CO2:
        :param X_CH4:
        :param mdot_CH4:
        :param mdot_H2:
        :param mdot_air_main:
        :param mdot_air_pilot:
        :return:
        """
        n_t_dry = (mdot_CH4/self.MW_CH4)/(X_CO+X_CO2+X_CH4)
        n_h2 = mdot_H2/self.MW_H2
        mdot_air_tot = mdot_air_main + mdot_air_pilot
        n_h2o = n_h2 + 2*X_CO*n_t_dry + 2*X_CO2*n_t_dry
        n_h2o_nt_dry = (n_h2/n_t_dry) + 2*X_CO + 2*X_CO2
        n_o2_in = (mdot_air_tot/self.MW_air)*(1/4.76)
        X_o2_wet = X_O2/(1+n_h2o_nt_dry)
        n_o2_wet = ((X_CO+2*X_CO2+2*X_O2)*n_t_dry/2.0) + (n_h2o/2.0)
        excess_o2 = 100.0*(n_o2_wet-n_o2_in)/n_o2_in

        return excess_o2

    def excess_O2(self,X_O2, X_CO,X_CO2,X_CH4, mdot_CH4, mdot_H2, mdot_air_main, mdot_air_pilot):
        """
        Calculate excess O2 % based on measured dry values of carbon species, inlet fuel and air mass flows.
        :param X_O2:
        :param X_CO:
        :param X_CO2:
        :param X_CH4:
        :param mdot_CH4:
        :param mdot_H2:
        :param mdot_air_main:
        :param mdot_air_pilot:
        :return: excess_o2, n_o2_calc, n_o2_excess, n_t_dry, n_h2o
        """
        n_h2_in = mdot_H2 / self.MW_H2
        n_ch4_in = mdot_CH4/self.MW_CH4
        mdot_air_tot = mdot_air_main + mdot_air_pilot
        n_o2_in = (mdot_air_tot/self.MW_air)/4.76
        c_denom = X_CO+X_CO2+X_CH4
        try:
            c_ratio = (X_CO2/(X_CO+X_CO2+X_CH4))
        except:
            c_ratio = 0.0
        n_o2_calc = (2.0*n_o2_in-2.0*n_ch4_in*(1.0+c_ratio)-n_h2_in)/2.0
        n_n2_calc = 3.76*n_o2_in
        n_o2_excess = (n_o2_calc*(1-X_O2)-X_O2*(n_ch4_in+n_n2_calc))/(4.76*X_O2-1.0)
        n_t_dry = n_ch4_in+n_o2_calc+n_n2_calc+4.76*n_o2_excess
        n_ch4_meas = X_CH4*n_t_dry
        n_h2o = n_h2_in+2*n_ch4_in-2*n_ch4_meas

        excess_o2 = 100.0*(n_o2_excess)/n_o2_calc

        return excess_o2, n_o2_calc, n_o2_excess, n_t_dry, n_h2o

    def data_plot(self):
        pathsave = self.drive + self.folder + "/Results_wet/"
        with open(self.drive + self.folder + "Unified_dataset_ExhaustEmissions", 'rb') as f:
            dataset = pickle.load(f)

        conditions = dataset.keys()
        # port=2
        H2_perc = 80
        identifiers = ['phi', 'H2']
        H2_perc_list = [0,10,50,80,100]
        phi_list = [0.3, 0.35,0.5,0.6,0.7, 0.8,0.9, 1.0]
        ident_excl = ['N2_', '_CO2_','_turbgrid']
        marker_list = ["o", "x", "x"]
        color_list = ['m', 'r', 'k', 'g','b']
        count = 0
        xlegend = []
        fig, ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        mkr_sz = 30
        for perc in H2_perc_list:
            id_list0 = np.append(identifiers, '_H2_' + str(perc)+'_')
            xlegend.append(str(perc)+"%")
            phi_plot=[]
            X_CO_list = []
            X_CO2_list = []
            X_NO_list = []
            X_NO2_list = []
            X_O2_list = []
            X_CH4_list = []
            O2_excess_list=[]
            for phi in phi_list:
                id_list = np.append(id_list0, '_phi_' + str(phi))
                id_list2 = np.append(id_list0, '_phi' + str(phi))
                for name in conditions:
                    check_id = [(x in name) for x in id_list]
                    check_id2 = [(x in name) for x in id_list2]
                    check_id_exclude = [(x in name) for x in ident_excl]
                    isfalse = False in check_id
                    isfalse2 = False in check_id2
                    istrue_excl = True in check_id_exclude
                    if (not (isfalse) and not (istrue_excl)) or (not (isfalse2) and not (istrue_excl)):
                        X_CO=np.mean(dataset[name]['CO'])*1e-6
                        X_CO2=np.mean(dataset[name]['CO2'])*1e-2
                        X_NO=np.mean(dataset[name]['NO'])*1e-6
                        X_NO2=np.mean(dataset[name]['NO2'])*1e-6
                        X_O2=np.mean(dataset[name]['O2'])*1e-2
                        X_CH4=np.mean(dataset[name]['CH4'])*1e-6
                        mdot_air_main = np.mean(dataset[name]['mdot_air'])*self.rho_n_air/60000.0
                        mdot_air_pilot = (np.mean(dataset[name]['mdot_pilotair']*1.1381-0.6213))*self.rho_n_air/60000.0
                        mdot_ch4 = np.mean(dataset[name]['mdot_ch4'])*self.rho_n_CH4/60000.0
                        mdot_h2 = np.mean(dataset[name]['mdot_h2'])*self.rho_n_H2/60000.0
                        excess_o2, n_o2_calc, n_o2_excess, n_t_dry, n_h2o = \
                            self.excess_O2(X_O2, X_CO,X_CO2,X_CH4, mdot_ch4, mdot_h2, mdot_air_main, mdot_air_pilot)
                        X_CO_corr, X_CO2_corr, X_CH4_corr, X_NO_corr, X_NO2_corr =\
                            self.spec_corr(X_O2, X_CO,X_CO2,X_CH4, X_NO, X_NO2, excess_o2, n_o2_calc, n_o2_excess, n_t_dry, n_h2o)
                        X_CO_list.append(X_CO_corr)
                        X_CO2_list.append(X_CO2_corr)
                        X_NO_list.append(X_NO_corr)
                        X_NO2_list.append(X_NO2_corr)
                        X_O2_list.append(X_O2*1e2)
                        X_CH4_list.append(X_CH4_corr)
                        O2_excess_list.append(excess_o2)
                        phi_plot.append(phi)
                        break

            xax = phi_plot
            ax.scatter(xax, X_CO_list, s=mkr_sz,
                       color=color_list[count])  # ,marker=marker_list[0],color=color_list[count])
            ax1.scatter(xax,  X_CO2_list, s=mkr_sz, color=color_list[count])
            ax2.scatter( xax, X_NO_list, s=mkr_sz, color=color_list[count])
            ax3.scatter( xax, X_NO2_list, s=mkr_sz, color=color_list[count])
            ax4.scatter( xax, X_O2_list, s=mkr_sz, color=color_list[count])
            ax5.scatter( xax, X_CH4_list, s=mkr_sz, color=color_list[count])
            ax6.scatter(xax, O2_excess_list, s=mkr_sz, color=color_list[count])


            count += 1

        if count == 0:
            return 0

        mkr_sz_leg = 1.0
        leg_title = 'H$_2$ %'
        ax.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax.set_ylabel("CO dry at 15% O2 (ppm)")
        ax.set_xlabel("Equivalence Ratio ($\phi$)")
        #ax.set_yscale('log')
        fig_name = "CO_exhaust_quartz"+ "_vs_phi"
        fig.tight_layout()
        fig.savefig(pathsave + fig_name + '.pdf')
        fig.savefig(pathsave + fig_name + '.png')

        ax1.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax1.set_ylabel("CO2 dry at 15% O2 (%)")
        ax1.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "CO2_exhaust_quartz" + "_vs_phi"
        fig1.tight_layout()
        fig1.savefig(pathsave + fig_name + '.pdf')
        fig1.savefig(pathsave + fig_name + '.png')

        ax2.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax2.set_ylabel("NO wet at 15% O2 (ppm)")
        ax2.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "NO_exhaust_quartz" + "_vs_phi"
        fig2.tight_layout()
        fig2.savefig(pathsave + fig_name + '.pdf')
        fig2.savefig(pathsave + fig_name + '.png')

        ax3.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax3.set_ylabel("NO2 wet at 15% O2 (ppm)")
        ax3.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "NO2_exhaust_quartz" + "_vs_phi"
        fig3.tight_layout()
        fig3.savefig(pathsave + fig_name + '.pdf')
        fig3.savefig(pathsave + fig_name + '.png')

        ax4.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax4.set_ylabel("O2 (%)")
        ax4.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "O2_exhaust_quartz" + "_vs_phi"
        fig4.tight_layout()
        fig4.savefig(pathsave + fig_name + '.pdf')
        fig4.savefig(pathsave + fig_name + '.png')

        ax5.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax5.set_ylabel("CH4 dry at 15% O2 (ppm)")
        ax5.set_xlabel("Equivalence Ratio ($\phi$)")
        ax5.set_yscale('log')
        fig_name = "CH4_exhaust_quartz" + "_vs_phi"
        fig5.tight_layout()
        fig5.savefig(pathsave + fig_name + '.pdf')
        fig5.savefig(pathsave + fig_name + '.png')

        ax6.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax6.set_ylabel("Excess O$_2$ (%)")
        ax6.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "O2_excess_exhaust_quartz" + "_vs_phi"
        fig6.tight_layout()
        fig6.savefig(pathsave + fig_name + '.pdf')
        fig6.savefig(pathsave + fig_name + '.png')

if __name__=="__main__":
    EE = ExhaustEmissions()
    #EE.tdms_file_search()
    EE.data_plot()
