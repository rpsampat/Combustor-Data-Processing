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
                           "Measured - CH4 Main", "'Raw TC Data'/'TC15'", "'Raw TC Data'/'Main Air'",
                           "'Raw TC Data'/'TC10'",
                           "'Raw TC Data'/'Cooling Air'", "'Raw TC Data'/'Fuel Temp'", "Offset",
                           "Position", "Avg Exhaust Temp", "'Pressure & GA'/'CH4'", "'Pressure & GA'/'CO2'",
                           "'Pressure & GA'/'CO'", "'Pressure & GA'/'NO'", "'Pressure & GA'/'NO2'",
                           "'Pressure & GA'/'O2'"]
        self.quant = {'mdot_air': "Measured - Main Air", 'mdot_pilotair': "Measured - Pilot Air",
                      'mdot_coolair': "Measured - Cooling Air", 'mdot_ch4': "Measured - CH4 Main",
                      'TC15': "'Raw TC Data'/'TC15'", 'Temp_mainair': "'Raw TC Data'/'Main Air'",
                      'Temp_flange': "'Raw TC Data'/'TC10'", 'Temp_coolair': "'Raw TC Data'/'Cooling Air'",
                      'Temp_fuel': "'Raw TC Data'/'Fuel Temp'", 'y_pos': "'Traverse System'/'Offset'",
                      'x_pos': "'Traverse System'/'Position'",
                      'T_exhaust': "Avg Exhaust Temp", 'CH4': "'Pressure & GA'/'CH4'", 'CO2': "'Pressure & GA'/'CO2'",
                      'CO': "'Pressure & GA'/'CO'", 'NO': "'Pressure & GA'/'NO'", 'NO2': "'Pressure & GA'/'NO2'",
                      'O2': "'Pressure & GA'/'O2'"}


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

    def data_plot(self):
        pathsave = self.drive + self.folder + "/Results/"
        with open(self.drive + self.folder + "Unified_dataset_ExhaustEmissions", 'rb') as f:
            dataset = pickle.load(f)

        conditions = dataset.keys()
        # port=2
        H2_perc = 80
        identifiers = ['phi', 'H2']
        H2_perc_list = [0,10,50,80,100]
        phi_list = [0.3, 0.6, 0.8, 1.0]
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
        mkr_sz = 30
        for perc in H2_perc_list:
            id_list0 = np.append(identifiers, '_H2_' + str(perc))
            xlegend.append(str(perc)+"%")
            phi_plot=[]
            X_CO = []
            X_CO2 = []
            X_NO = []
            X_NO2 = []
            X_O2 = []
            X_CH4 = []
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
                        X_CO.append(np.mean(dataset[name]['CO']))
                        X_CO2.append(np.mean(dataset[name]['CO2']))
                        X_NO.append(np.mean(dataset[name]['NO']))
                        X_NO2.append(np.mean(dataset[name]['NO2']))
                        X_O2.append(np.mean(dataset[name]['O2']))
                        X_CH4.append(np.mean(dataset[name]['CH4']))
                        phi_plot.append(phi)
                        break

            xax = phi_plot
            ax.scatter(xax, X_CO, s=mkr_sz,
                       color=color_list[count])  # ,marker=marker_list[0],color=color_list[count])
            ax1.scatter(xax,  X_CO2, s=mkr_sz, color=color_list[count])
            ax2.scatter( xax, X_NO, s=mkr_sz, color=color_list[count])
            ax3.scatter( xax, X_NO2, s=mkr_sz, color=color_list[count])
            ax4.scatter( xax, X_O2, s=mkr_sz, color=color_list[count])
            ax5.scatter( xax, X_CH4, s=mkr_sz, color=color_list[count])


            count += 1

        if count == 0:
            return 0

        mkr_sz_leg = 1.0
        ax.legend(xlegend, markerscale=mkr_sz_leg)
        ax.set_ylabel("CO dry at 15% O2 (ppm)")
        ax.set_xlabel("Equivalence Ratio ($\phi$)")
        ax.set_yscale('log')
        fig_name = "CO_exhaust_quartz"+ "_vs_phi"
        fig.savefig(pathsave + fig_name + '.pdf')
        fig.savefig(pathsave + fig_name + '.png')

        ax1.legend(xlegend, markerscale=mkr_sz_leg)
        ax1.set_ylabel("CO2 dry at 15% O2 (%)")
        ax1.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "CO2_exhaust_quartz" + "_vs_phi"
        fig1.savefig(pathsave + fig_name + '.pdf')
        fig1.savefig(pathsave + fig_name + '.png')

        ax2.legend(xlegend, markerscale=mkr_sz_leg)
        ax2.set_ylabel("NO dry at 15% O2 (ppm)")
        ax2.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "NO_exhaust_quartz" + "_vs_phi"
        fig2.savefig(pathsave + fig_name + '.pdf')
        fig2.savefig(pathsave + fig_name + '.png')

        ax3.legend(xlegend, markerscale=mkr_sz_leg)
        ax3.set_ylabel("NO2 dry at 15% O2 (ppm)")
        ax3.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "NO2_exhaust_quartz" + "_vs_phi"
        fig3.savefig(pathsave + fig_name + '.pdf')
        fig3.savefig(pathsave + fig_name + '.png')

        ax4.legend(xlegend, markerscale=mkr_sz_leg)
        ax4.set_ylabel("O2 (%)")
        ax4.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "O2_exhaust_quartz" + "_vs_phi"
        fig4.savefig(pathsave + fig_name + '.pdf')
        fig4.savefig(pathsave + fig_name + '.png')

        ax5.legend(xlegend, markerscale=mkr_sz_leg)
        ax5.set_ylabel("CH4 dry at 15% O2 (ppm)")
        ax5.set_xlabel("Equivalence Ratio ($\phi$)")
        ax5.set_yscale('log')
        fig_name = "CH4_exhaust_quartz" + "_vs_phi"
        fig5.savefig(pathsave + fig_name + '.pdf')
        fig5.savefig(pathsave + fig_name + '.png')

if __name__=="__main__":
    EE = ExhaustEmissions()
    #EE.tdms_file_search()
    EE.data_plot()
