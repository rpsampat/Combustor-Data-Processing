import math
import os
import cantera as ct
import numpy as np
import pickle
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
#import Thermocouple as TC
from labview_extract import DataExtract
from nptdms import TdmsFile
from scandir import scandir


class GasProbeTrav:
    def __init__(self):
        self.drive = 'O:/'
        self.folder = "FlamelessCombustor_Jan2023/Labview_data/ExpCampH2admix_2023/SteelChamber_Zaber/ProcessedData/"
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
        identifiers=["NV100_","H2_","_phi_","GA_","Port",".tdms"]
        identifier_exclude = ["_index"]
        subdir_list = next(os.walk(path))[1] # list of immediate subdirectories within the steel chamber data directory
        data_GA_trav={}
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
                    data_GA_trav[names] = data_dict
        with open(self.drive + self.folder + "Unified_dataset_GA_traverse", 'wb') as file:
            pickle.dump(data_GA_trav, file, pickle.HIGHEST_PROTOCOL)

    def data_plot(self,port,H2_perc):
        pathsave = self.drive + self.folder + "/Results_GA_trav/"
        with open(self.drive + self.folder + "Unified_dataset_GA_traverse", 'rb') as f:
            dataset = pickle.load(f)

        conditions = dataset.keys()
        #port=2
        #H2_perc = 80
        identifiers = ['Port'+str(port),'_phi_','H2_'+str(H2_perc)]
        phi_list=[0.3,0.6,0.8,1.0]
        ident_excl = ['N2_8','_CO2_']
        marker_list = ["o", "x", "x"]
        color_list = ['m', 'r', 'k', 'g']
        count=0
        xlegend=[]
        fig,ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        mkr_sz = 1.0
        for phi in phi_list:
            for name in conditions:
                id_list = np.append(identifiers,'_phi_'+str(phi))
                check_id = [(x in name) for x in id_list]
                check_id_exclude = [(x in name) for x in ident_excl]
                isfalse = False in check_id
                istrue_excl = True in check_id_exclude
                if not(isfalse) and not(istrue_excl):
                    index1 = name.index('_phi_')+len('_phi_')#name.index('NV100_')+len('NV100_')
                    index2 = name.index('_GA')
                    xlegend.append("$\phi$="+name[index1:index2])
                    xpos = dataset[name]['x_pos']
                    wallpos = np.min(xpos)
                    xpos = xpos-wallpos
                    dx = np.max(np.abs(np.diff(xpos)))
                    N_kernel = int(12.0/dx)
                    xpos = np.convolve(xpos, np.ones(N_kernel) / N_kernel, mode='valid')
                    X_CO = np.convolve(dataset[name]['CO'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_CO2 = np.convolve(dataset[name]['CO2'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_NO = np.convolve(dataset[name]['NO'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_NO2 = np.convolve(dataset[name]['NO2'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_O2 = np.convolve(dataset[name]['O2'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_CH4 = np.convolve(dataset[name]['CH4'], np.ones(N_kernel) / N_kernel, mode='valid')

                    ax.scatter(xpos,X_CO, s=mkr_sz,color = color_list[count])#,marker=marker_list[0],color=color_list[count])
                    ax1.scatter(xpos, X_CO2, s=mkr_sz, color = color_list[count])
                    ax2.scatter(xpos, X_NO, s=mkr_sz, color = color_list[count])
                    ax3.scatter(xpos, X_NO2, s=mkr_sz, color = color_list[count])
                    ax4.scatter(xpos, X_O2, s=mkr_sz, color = color_list[count])
                    ax5.scatter(xpos, X_CH4, s=mkr_sz, color=color_list[count])
                    count += 1
                    break




        if count ==0:
            return 0
        mkr_sz_leg = 3.0
        ax.legend(xlegend, markerscale=mkr_sz_leg)
        ax.set_ylabel("CO dry at 15% O2 (ppm)")
        ax.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "CO corrected_Port" +str(port) + "_H2_"+str(H2_perc)+  "_vs_radialloc"
        fig.savefig(pathsave + fig_name + '.pdf')
        fig.savefig(pathsave + fig_name + '.png')

        ax1.legend(xlegend, markerscale=mkr_sz_leg)
        ax1.set_ylabel("CO2 dry at 15% O2 (ppm)")
        ax1.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "CO2 corrected_Port" + str(port) + "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig1.savefig(pathsave + fig_name + '.pdf')
        fig1.savefig(pathsave + fig_name + '.png')

        ax2.legend(xlegend, markerscale=mkr_sz_leg)
        ax2.set_ylabel("NO dry at 15% O2 (ppm)")
        ax2.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "NO corrected_Port" + str(port) + "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig2.savefig(pathsave + fig_name + '.pdf')
        fig2.savefig(pathsave + fig_name + '.png')

        ax3.legend(xlegend, markerscale=mkr_sz_leg)
        ax3.set_ylabel("NO2 dry at 15% O2 (ppm)")
        ax3.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "NO2 corrected_Port" + str(port) + "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig3.savefig(pathsave + fig_name + '.pdf')
        fig3.savefig(pathsave + fig_name + '.png')

        ax4.legend(xlegend, markerscale=mkr_sz_leg)
        ax4.set_ylabel("O2 dry at 15% O2 (ppm)")
        ax4.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "O2 corrected_Port" + str(port) + "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig4.savefig(pathsave + fig_name + '.pdf')
        fig4.savefig(pathsave + fig_name + '.png')

        ax5.legend(xlegend, markerscale=mkr_sz_leg)
        ax5.set_ylabel("CH4 dry at 15% O2 (ppm)")
        ax5.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "CH4 corrected_Port" + str(port) + "_H2_" + str(H2_perc) + "_vs_radialloc"
        fig5.savefig(pathsave + fig_name + '.pdf')
        fig5.savefig(pathsave + fig_name + '.png')

        plt.close('all')

        #plt.show()

        return 0

    def data_plot_leg_port(self,H2_perc):
        pathsave = self.drive + self.folder + "/Results_GA_trav_Portcomp/"
        with open(self.drive + self.folder + "Unified_dataset_GA_traverse", 'rb') as f:
            dataset = pickle.load(f)

        conditions = dataset.keys()
        #port=2
        #H2_perc = 80
        identifiers = ['Port','_phi_1.0','H2_'+str(H2_perc)]
        phi_list=[0.3,0.6,0.8,1.0]
        port_list = [2,3,5]
        ident_excl = ['N2_8','_CO2_']
        marker_list = ["o", "x", "x"]
        color_list = ['m', 'r', 'k', 'g']
        count=0
        xlegend=[]
        fig,ax = plt.subplots()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()
        fig5, ax5 = plt.subplots()
        mkr_sz = 1.0
        for port in port_list:
            xlegend.append("Port" + str(port))
            for name in conditions:
                id_list = np.append(identifiers,'Port'+str(port))
                check_id = [(x in name) for x in id_list]
                check_id_exclude = [(x in name) for x in ident_excl]
                isfalse = False in check_id
                istrue_excl = True in check_id_exclude
                if not(isfalse) and not(istrue_excl):
                    xpos = dataset[name]['x_pos']
                    wallpos = np.min(xpos)
                    xpos = xpos-wallpos
                    dx = np.max(np.abs(np.diff(xpos)))
                    N_kernel = int(12.0/dx)
                    # sliding average by convolution operator with kernel size=N_kernel
                    xpos = np.convolve(xpos, np.ones(N_kernel) / N_kernel, mode='valid')
                    X_CO = np.convolve(dataset[name]['CO'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_CO2 = np.convolve(dataset[name]['CO2'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_NO = np.convolve(dataset[name]['NO'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_NO2 = np.convolve(dataset[name]['NO2'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_O2 = np.convolve(dataset[name]['O2'], np.ones(N_kernel) / N_kernel, mode='valid')
                    X_CH4 = np.convolve(dataset[name]['CH4'], np.ones(N_kernel) / N_kernel, mode='valid')

                    ax.scatter(xpos,X_CO, s=mkr_sz,color = color_list[count])#,marker=marker_list[0],color=color_list[count])
                    ax1.scatter(xpos, X_CO2, s=mkr_sz, color = color_list[count])
                    ax2.scatter(xpos, X_NO, s=mkr_sz, color = color_list[count])
                    ax3.scatter(xpos, X_NO2, s=mkr_sz, color = color_list[count])
                    ax4.scatter(xpos, X_O2, s=mkr_sz, color = color_list[count])
                    ax5.scatter(xpos, X_CH4, s=mkr_sz, color=color_list[count])
                    count += 1
                    break




        if count ==0:
            return 0
        mkr_sz_leg = 3.0
        ax.legend(xlegend, markerscale=mkr_sz_leg)
        ax.set_ylabel("CO dry at 15% O2 (ppm)")
        ax.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "CO corrected_Port_leg"+"_H2_"+str(H2_perc)+  "_vs_radialloc"
        fig.savefig(pathsave + fig_name + '.pdf')
        fig.savefig(pathsave + fig_name + '.png')

        ax1.legend(xlegend, markerscale=mkr_sz_leg)
        ax1.set_ylabel("CO2 dry at 15% O2 (ppm)")
        ax1.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "CO2 corrected_Port_leg"+ "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig1.savefig(pathsave + fig_name + '.pdf')
        fig1.savefig(pathsave + fig_name + '.png')

        ax2.legend(xlegend, markerscale=mkr_sz_leg)
        ax2.set_ylabel("NO dry at 15% O2 (ppm)")
        ax2.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "NO corrected_Port_leg" + "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig2.savefig(pathsave + fig_name + '.pdf')
        fig2.savefig(pathsave + fig_name + '.png')

        ax3.legend(xlegend, markerscale=mkr_sz_leg)
        ax3.set_ylabel("NO2 dry at 15% O2 (ppm)")
        ax3.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "NO2 corrected_Port_leg" + "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig3.savefig(pathsave + fig_name + '.pdf')
        fig3.savefig(pathsave + fig_name + '.png')

        ax4.legend(xlegend, markerscale=mkr_sz_leg)
        ax4.set_ylabel("O2 dry at 15% O2 (ppm)")
        ax4.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "O2 corrected_Port_leg" + "_H2_"+str(H2_perc)+ "_vs_radialloc"
        fig4.savefig(pathsave + fig_name + '.pdf')
        fig4.savefig(pathsave + fig_name + '.png')

        ax5.legend(xlegend, markerscale=mkr_sz_leg)
        ax5.set_ylabel("CH4 dry at 15% O2 (ppm)")
        ax5.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "CH4 corrected_Port_leg" + "_H2_" + str(H2_perc) + "_vs_radialloc"
        fig5.savefig(pathsave + fig_name + '.pdf')
        fig5.savefig(pathsave + fig_name + '.png')

        plt.close('all')

        #plt.show()

        return 0







if __name__=="__main__":
    GPT = GasProbeTrav()
    #GPT.tdms_file_search()
    port_list = [2,3,5]
    H2_list = [0,50,80]
    """for port in port_list:
        for H2 in H2_list:
            GPT.data_plot(port,H2)"""

    for H2 in H2_list:
        GPT.data_plot_leg_port(H2)