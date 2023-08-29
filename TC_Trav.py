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
import ExhaustEmissions as ExhEmiss


class TC_Trav:
    def __init__(self):
        self.drive = 'P:/'
        self.folder = "FlamelessCombustor_Jan2023/Labview_data/ExpCampH2admix_2023/SteelChamber_Zaber/ProcessedData/"
        self.dataset = {}
        self.dataset_temperature = {}
        self.quant_list = ["Measured - Main Air", "Measured - Pilot Air", "Measured - Cooling Air",
                           "Measured - CH4 Main", "Measured - H2", "'Raw TC Data'/'TC15'", "'Raw TC Data'/'Main Air'",
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

        self.MW_air = 28.97  # kg/kmol
        self.MW_O2 = gas.molecular_weights[species.index('O2')]
        self.MW_N2 = gas.molecular_weights[species.index('N2')]
        self.MW_CH4 = gas.molecular_weights[species.index('CH4')]
        self.MW_H2 = gas.molecular_weights[species.index('H2')]
        self.dia_tc = 0.003 #m
        self.epsi_tc = 0.4  # thermocouple emissivity (ceramic alumina)
        self.epsi_gas = 0.05  #
        self.epsi_wall = 0.79  # combustor wall emissivity
        self.sigma = 5.6704e-8  # stefan-boltzman constant
        self.Pr_air = 0.7
        self.dia_burner = 206.0  # mm


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
        identifiers=["NV100_","H2_","_phi_","TC_","Port",".tdms"]
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
        with open(self.drive + self.folder + "Unified_dataset_TC_traverse", 'wb') as file:
            pickle.dump(data_GA_trav, file, pickle.HIGHEST_PROTOCOL)

    def data_plot(self,port,H2_perc):
        pathsave = self.drive + self.folder + "/Results_TC_trav/"
        with open(self.drive + self.folder + "Unified_dataset_TC_traverse", 'rb') as f:
            dataset = pickle.load(f)

        conditions = dataset.keys()
        #port=3
        identifiers = ['Port' + str(port), '_phi_', 'H2_' + str(H2_perc)+ '_']
        phi_list = [0.3, 0.6, 0.8, 1.0]
        ident_excl = ['N2_8','_CO2_']
        marker_list = ["o", "x", "x"]
        color_list = ['m', 'r', 'k', 'g']
        count = 0
        xlegend=[]
        fig,ax = plt.subplots()
        """fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        fig4, ax4 = plt.subplots()"""
        mkr_sz = 1.0
        for phi in phi_list:
            for name in conditions:
                id_list = np.append(identifiers,'_phi_'+str(phi)+'_')
                check_id_exclude = [(x in name) for x in ident_excl]
                check_id = [(x in name) for x in id_list]
                isfalse = False in check_id
                istrue_excl = True in check_id_exclude
                if not(isfalse) and not(istrue_excl):
                    index1 = name.index('_phi_') + len('_phi_')  # name.index('NV100_')+len('NV100_')
                    index2 = name.index('_TC')
                    xlegend.append(name[index1:index2])
                    xpos = dataset[name]['x_pos']
                    wallpos = np.min(xpos)
                    xpos = xpos-wallpos
                    xpos_diff = np.diff(xpos)
                    # check sign change of 1st 1/3rd of xpos range
                    sign_change = xpos[int(len(xpos)/3)]-xpos[0]
                    xpos_diff_prod = np.multiply(xpos_diff,sign_change)
                    ind_slice = np.where(xpos_diff_prod<0.0)
                    if len(ind_slice[0])==0:
                        ind_slice = -1
                    else:
                        ind_slice = ind_slice[0][0]
                    dx = np.max(np.abs(np.diff(xpos)))
                    N_kernel = int(6.0/dx)
                    # sliding average by convolution operator with kernel size=N_kernel
                    xpos = np.convolve(xpos[0:ind_slice], np.ones(N_kernel) / N_kernel, mode='valid')
                    TC15_avg = np.average(dataset[name]['TC15'].reshape(-1, 10), axis=1)
                    TC15 = np.convolve(TC15_avg[0:ind_slice], np.ones(N_kernel) / (N_kernel), mode='valid')
                    ax.scatter(xpos,TC15, s=mkr_sz,color = color_list[count])#,marker=marker_list[0],color=color_list[count])
                    count+=1
                    break

        if count ==0:
            return 0
        mkr_sz_leg = 3.0
        ax.legend(xlegend, markerscale=mkr_sz_leg)
        ax.set_ylabel("Temperature (C)")
        ax.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "TC15_Port" +str(port) + "_H2_"+str(H2_perc)+  "_vs_radialloc"
        fig.savefig(pathsave + fig_name + '.pdf')
        fig.savefig(pathsave + fig_name + '.png')

        """ax1.legend(xlegend, markerscale=3.0)
        ax1.set_ylabel("CO2 dry at 15% O2 (ppm)")
        ax1.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "CO2 corrected_Port" + str(port) + "_vs_radialloc"
        fig1.savefig(pathsave + fig_name + '.pdf')
        fig1.savefig(pathsave + fig_name + '.png')

        ax2.legend(xlegend, markerscale=3.0)
        ax2.set_ylabel("NO dry at 15% O2 (ppm)")
        ax2.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "NO corrected_Port" + str(port) + "_vs_radialloc"
        fig2.savefig(pathsave + fig_name + '.pdf')
        fig2.savefig(pathsave + fig_name + '.png')

        ax3.legend(xlegend, markerscale=3.0)
        ax3.set_ylabel("NO2 dry at 15% O2 (ppm)")
        ax3.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "NO2 corrected_Port" + str(port) + "_vs_radialloc"
        fig3.savefig(pathsave + fig_name + '.pdf')
        fig3.savefig(pathsave + fig_name + '.png')

        ax4.legend(xlegend, markerscale=3.0)
        ax4.set_ylabel("O2 dry at 15% O2 (ppm)")
        ax4.set_xlabel("Distance from wall/Radius of Chamber")
        fig_name = "O2 corrected_Port" + str(port) + "_vs_radialloc"
        fig4.savefig(pathsave + fig_name + '.pdf')
        fig4.savefig(pathsave + fig_name + '.png')"""



        #plt.show()

        return 0
    def spec_corr(self,X_O2, X_CO,X_CO2,X_CH4, X_NO, X_NO2, mdot_CH4, mdot_H2, mdot_air_main, mdot_air_pilot):
        x_O2 = 15 # % O2 corrected
        X_CO_corr = (X_CO*(1-x_O2/100.0)/(1-X_O2))*1e6 # dry at x% O2 (ppm)
        X_CO2_corr = (X_CO2 * (1 - x_O2 / 100.0) / (1 - X_O2))*1e2 # dry at x% O2 (%)
        X_CH4_corr = (X_CH4 * (1 - x_O2 / 100.0) / (1 - X_O2))*1e6 # dry at x% O2 (ppm)
        n_t_dry = (mdot_CH4 / self.MW_CH4) / (X_CO + X_CO2 + X_CH4)
        n_h2 = mdot_H2 / self.MW_H2
        mdot_air_tot = mdot_air_main + mdot_air_pilot
        n_h2o = n_h2 + 2 * X_CO * n_t_dry + 2 * X_CO2 * n_t_dry
        n_t_wet = n_t_dry + n_h2o
        n_corr = n_t_dry/(1-x_O2/100.0)
        wet='y'
        if wet == 'y':
            X_NO_corr = (X_NO * (1 - x_O2 / 100.0) / (1 - X_O2))*1e6#(X_NO*n_t_wet/n_corr)*1e6 # dry at x% O2 (ppm)
            X_NO2_corr = (X_NO2 * (1 - x_O2 / 100.0) / (1 - X_O2))*1e6#(X_NO2 * n_t_wet / n_corr)*1e6 # dry at x% O2 (ppm)
        else:
            X_NO_corr = (X_NO*n_t_wet/n_corr)*1e6 # dry at x% O2 (ppm)
            X_NO2_corr = (X_NO2 * n_t_wet / n_corr)*1e6 # dry at x% O2 (ppm)



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

    def convection_thermocouple(self, u, Temp):
        """
        Convection between thermocouple and external cooling flow modelled as cross flow around cylinder
        :return:
        """
        air = ct.Solution('air.yaml')
        air.TP = Temp + 273.15, ct.one_atm
        if u < 0:
            h = 50.0
            return h
        else:
            try:
                Re_tc = air.density * u * (self.dia_tc) / air.viscosity
                Nu_lam = 0.664 * (Re_tc ** 0.5) * (self.Pr_air ** 0.333)
                Nu_turb = 0.037 * (Re_tc ** 0.8) * self.Pr_air / (
                        1 + 2.443 * (Re_tc ** -0.1) * (self.Pr_air ** (2.0 / 3.0) - 1))
                Nu = 0.3 + math.sqrt(Nu_lam ** 2 + Nu_turb ** 2)
            except:
                Nu=0.3
            h = Nu * air.thermal_conductivity / self.dia_tc
            # print "Nu=", Nu

            return h

    def heat_balance(self, T_wall, area_cond):
        """
        A T^4 + B T + C = 0
        :return:
        """
        dx = 0.001
        h = self.convection_thermocouple()
        area_rad_tc = (math.pi / 4.0) * self.dia_tc ** 2 + math.pi * self.dia_tc * 0.10
        area_rad_chamber = math.pi * self.dia_chamber * 0.1

        area_conv = math.pi * self.dia_tc * 0.10
        A = -1.0 * self.epsi_tc * self.sigma * area_rad_tc
        B = (-self.k_steel / dx) * area_cond - h * area_conv
        C = self.epsi_wall * self.sigma * T_wall ** 4.0 * area_rad_tc + (self.k_steel / dx) * T_wall * area_cond \
            + h * self.T_air * area_conv
        coeff = [A, 0.0, 0.0, B, C]
        roots = np.roots(coeff)
        root_eff = roots.real[(abs(roots.imag) < 1e-05) & (roots.real > 0.0)]
        Q_rad = -1.0 * self.epsi_tc * self.sigma * area_rad_tc * root_eff[0] ** 4.0 \
                + self.epsi_wall * self.sigma * T_wall ** 4.0 * area_rad_tc
        Q_rad_chamber = self.epsi_wall * self.sigma * T_wall ** 4.0 * area_rad_chamber
        return root_eff, Q_rad, Q_rad_chamber

    def thermocouple_corr(self, Temp_C, xloc, u,T_surr):
        Temp = Temp_C + 273.15  # K
        h = self.convection_thermocouple(u, Temp_C)
        #T_surr = 15 + 273.15  # K
        """if xloc > self.dia_burner / 2.0:
            T_corr = Temp + self.sigma * (self.epsi_tc * Temp ** 4.0 - self.epsi_gas * T_surr ** 4.0) / h
        else:"""
        A = self.sigma * self.epsi_gas
        B = h
        C = -h * Temp - self.sigma * self.epsi_tc * Temp ** 4.0
        coeff = [A, 0.0, 0.0, B, C]
        #try:
        roots = np.roots(coeff)
        """except:
            print("h=",h)
            print("vel=",u)
            print("Temp=",Temp_C)
            print(A)
            print(B)
            print(C)"""
        # print Temp
        # print roots
        root_eff = roots.real[(abs(roots.imag) < 1e-08) & (roots.real > 0.0)]
        T_corr = root_eff

        return T_corr

    def TC_raw_to_corr(self,xpos,TC,y_vel,u_vel):
        T_wall = TC[np.where(xpos==0.0)[0]]+273.15 # K
        """fig,ax = plt.subplots()
        ax.scatter(range(len(y_vel)),y_vel)
        plt.show()"""
        for x in range(len(xpos)):
            vel_ind = np.where(np.abs(y_vel-xpos[x])<self.dia_tc*1000.0/2.0)[0]
            if len(vel_ind)==0:
                u_conv = 0.0
            else:
                u_conv = np.mean(u_vel[vel_ind])
            T_corr = self.thermocouple_corr(TC[x],xpos[x],u=u_conv, T_surr = T_wall)

            #T_corr, Q_rad, Q_rad_chamber = self.heat_balance(T_wall, area_cond=0.0)
            #if len(T_corr)>1:
             #   print(T_corr[0])
            try:
                T_corr_arr=np.append(T_corr_arr,T_corr[0]-273.15)
            except:
                try:
                    T_corr_arr = np.array([T_corr[0]-273.15])
                except:
                    T_corr_arr = np.array([450])

        return T_corr_arr



    def data_comparison(self):
        """
        Plotting a concise comparison of images of different conditions of operation
        :return:
        """
        H2_perc = [0,50,80,100]#[0,80]#
        port = [2,3,5,6]#[2,3,5,6]#[2,3,5]#
        N2_perc = [0,15,11]
        pathsave = self.drive + self.folder + "/Results_TC_trav/"
        with open(self.drive + self.folder + "Unified_dataset_TC_traverse", 'rb') as f:
            dataset = pickle.load(f)

        conditions = dataset.keys()
        # port=2
        # H2_perc = 80

        phi_list = [0.3, 0.6, 0.8, 1.0]#[0.6,1.0]#
        ident_excl = ['N2_', '_CO2_']
        marker_list = ["o", "x", "x"]
        color_list = ['m', 'r', 'k', 'g']
        count = 0
        EE = ExhEmiss.ExhaustEmissions()
        import matplotlib as mpl
        mpl.rcParams['axes.linewidth'] = 0.1  # set the value globally
        #mpl.rcParams['legend.frameon'] = 'False'
        mkr_sz = 0.8
        label_size = 10.0
        tick_size = 8.0
        figure, ax = plt.subplots(len(H2_perc), len(port), sharex=False, sharey=False, dpi=300,
                                  gridspec_kw={'wspace': 0.27, 'hspace': 0.05})

        for i in range(len(port)):
            for j in range(len(H2_perc)):
                xlegend = []
                count =0
                for phi in phi_list:
                    name=0
                    identifiers = ['Port' + str(port[i]), '_phi_', 'H2_' + str(H2_perc[j]) + '_']
                    for name_search in conditions:
                        id_list = np.append(identifiers, '_phi_' + str(phi) + "_")
                        check_id = [(x in name_search) for x in id_list]
                        check_id_exclude = [(x in name_search) for x in ident_excl]
                        isfalse = False in check_id
                        istrue_excl = True in check_id_exclude
                        if not (isfalse) and not (istrue_excl):
                            index1 = name_search.index('_phi_') + len('_phi_')  # name.index('NV100_')+len('NV100_')
                            index2 = name_search.index('_TC')
                            xlegend.append(name_search[index1:index2])
                            name = name_search
                            break
                    if name == 0:
                        continue
                    else:
                        count +=1
                        xpos = dataset[name]['x_pos']
                        r_chamber = 103.25
                        wallpos = np.min(xpos)
                        xpos = (xpos - wallpos)  # /r_chamber
                        xpos_diff = np.diff(xpos)
                        # check sign change of 1st 1/3rd of xpos range
                        sign_change = xpos[int(len(xpos) / 3)] - xpos[0]
                        xpos_diff_prod = np.multiply(xpos_diff, sign_change)
                        ind_slice = np.where(xpos_diff_prod < 0.0)
                        if len(ind_slice[0]) == 0:
                            ind_slice = -1
                        else:
                            ind_slice = ind_slice[0][0]
                        dx = np.max(np.abs(np.diff(xpos)))
                        N_kernel = int(6.0 / dx)
                        # sliding average by convolution operator with kernel size=N_kernel
                        xpos = np.convolve(xpos[0:ind_slice], np.ones(N_kernel) / N_kernel, mode='valid')
                        TC15_avg = np.average(dataset[name]['TC15'].reshape(-1, 10), axis=1)
                        TC15 = np.convolve(TC15_avg[0:ind_slice], np.ones(N_kernel) / (N_kernel), mode='valid')
                        phi_ind = phi_list.index(phi)
                        if port[i] == 6:
                            with open("phi08" + "_port" + str(5), 'rb') as f:
                                dict_port = pickle.load(f)
                        else:
                            with open("phi08"+"_port"+str(port[i]), 'rb') as f:
                                dict_port = pickle.load(f)
                        T_corr = self.TC_raw_to_corr(xpos,TC15,y_vel = dict_port["y"],u_vel = np.abs(dict_port["u"]))
                        xpos = xpos / r_chamber
                        dict_Tcorr={"T":T_corr,"xpos":xpos}


                        with open(pathsave+"Tcorrected_"+"phi"+str(phi)+"_port"+str(port[i]), 'wb') as f:
                            pickle.dump(dict_Tcorr, f, pickle.HIGHEST_PROTOCOL)
                        try:
                            ax[j, i].scatter(xpos, T_corr, s=mkr_sz, linewidths=0.0,color = color_list[phi_ind])
                        except:
                            print(len(T_corr))
                            print(len(xpos))
                            print('Port' + str(port[i])+ '_phi_'+str(phi)+'H2_' + str(H2_perc[j]))
                        ax[j,i].set_ylim(450,2200)

                if count==0:
                    ax[j, i].axis('off')
                    ax[j, i].set_frame_on(False)
                    if j == 0:
                        ax[j, i].set_title(str(H2_perc[j]))
                    if i == 0:
                        ax[j, i].set_ylabel("H$_2$="+str(H2_perc[j]) + "%\nCO dry at 15% O2 (ppm)", fontsize=label_size)
                        ax[j, i].axis('on')
                        # ax[j, i].set_xticks([])
                        # ax[j, i].set_yticks([])
                        # ax[j, i].xaxis.set_tick_params(labelbottom=False)
                        # ax[j, i].yaxis.set_tick_params(labelleft=False)
                else:
                    ax[j, i].tick_params(axis='both', labelsize=tick_size,width=1.0)
                    # ax[j, i].axis('off')
                    # ax[j,i].set_frame_on(False)
                    #tick_locs = ax[j,i].get_xticks()
                    #tick_labels = ax[j, i].get_xticklabels()

                    if j == 0:
                        ax[j, i].set_title("Port " + str(port[i]), fontsize=label_size*1.5)
                    if i == 0:
                        ax[j, i].set_ylabel("H$_2$=" + str(H2_perc[j]) + "%\n"+"Temperature (C)",
                                            fontsize=label_size*0.75)
                        ax[j, i].axis('on')

                    if j == len(H2_perc) - 1:
                        ax[j, i].set_xlabel("r$_{wall}$/R$_0$", fontsize=label_size)#"Distance from wall/Radius of Chamber"
                        #ax[j, i].set_xticks(ticks=tick_locs)#,labels=tick_labels)
                        #ax[j, i].xaxis.set_tick_params(labelbottom=True)
                    else:
                        ax[j, i].set_xticks([])
                        ax[j, i].xaxis.set_tick_params(labelbottom=False)
                    mkr_sz_leg = 4.0
                    leg = ax[j, i].legend(xlegend, title='$\phi$',title_fontsize=9.0,markerscale=mkr_sz_leg, fontsize=8.0,fancybox = False)
                    leg.get_frame().set_linewidth(0.1)
                    leg.get_frame().set_edgecolor("black")

        #figure.tight_layout()
        #plt.show()
        plt.savefig(pathsave + 'TC_comparison.png', bbox_inches='tight')
        plt.savefig(pathsave + 'TC_comparison.pdf', bbox_inches='tight')
        #plt.show()
        plt.close(figure)
    def data_plot_gascomp_exh(self):
        pathsave = self.drive + self.folder + "/Results_TC_gascomp_wet/"
        with open(self.drive + self.folder + "Unified_dataset_TC_traverse", 'rb') as f:
            dataset = pickle.load(f)

        conditions = dataset.keys()
        # port=2
        H2_perc = 80
        identifiers = ['phi', 'H2','Port3']
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
        fig7, ax7 = plt.subplots()
        mkr_sz = 30
        EE = ExhEmiss.ExhaustEmissions()
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
            Air_cool_list = []
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
                        mdot_air_cool = np.mean(dataset[name]['mdot_coolair']) * self.rho_n_air / 60000.0
                        mdot_air_pilot = (np.mean(dataset[name]['mdot_pilotair'])*1.1381-0.6213)*self.rho_n_air/60000.0
                        mdot_ch4 = np.mean(dataset[name]['mdot_ch4'])*self.rho_n_CH4/60000.0
                        mdot_h2 = np.mean(dataset[name]['mdot_h2'])*self.rho_n_H2/60000.0
                        excess_o2, n_o2_calc, n_o2_excess, n_t_dry, n_h2o = EE.excess_O2(X_O2, X_CO,X_CO2,X_CH4, mdot_ch4, mdot_h2, mdot_air_main, mdot_air_pilot)
                        X_CO_corr, X_CO2_corr, X_CH4_corr, X_NO_corr, X_NO2_corr = \
                            EE.spec_corr(X_O2, X_CO, X_CO2, X_CH4, X_NO, X_NO2, excess_o2, n_o2_calc, n_o2_excess,
                                           n_t_dry, n_h2o)
                        X_CO_list.append(X_CO_corr)
                        X_CO2_list.append(X_CO2_corr)
                        X_NO_list.append(X_NO_corr)
                        X_NO2_list.append(X_NO2_corr)
                        X_O2_list.append(X_O2*1e2)
                        X_CH4_list.append(X_CH4_corr)
                        O2_excess_list.append(excess_o2)
                        phi_plot.append(phi)
                        Air_cool_list.append(mdot_air_cool)
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
            ax7.scatter(xax, Air_cool_list, s=mkr_sz, color=color_list[count])


            count += 1

        if count == 0:
            return 0

        mkr_sz_leg = 1.0
        leg_title = 'H$_2$ %'
        ax.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax.set_ylabel("CO dry at 15% O2 (ppm)")
        ax.set_xlabel("Equivalence Ratio ($\phi$)")
        #ax.set_yscale('log')
        fig_name = "CO_exhaust_steel"+ "_vs_phi"
        fig.tight_layout()
        fig.savefig(pathsave + fig_name + '.pdf')
        fig.savefig(pathsave + fig_name + '.png')

        ax1.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax1.set_ylabel("CO2 dry at 15% O2 (%)")
        ax1.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "CO2_exhaust_steel" + "_vs_phi"
        fig1.tight_layout()
        fig1.savefig(pathsave + fig_name + '.pdf')
        fig1.savefig(pathsave + fig_name + '.png')

        ax2.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax2.set_ylabel("NO wet at 15% O2 (ppm)")
        ax2.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "NO_exhaust_steel" + "_vs_phi"
        fig2.tight_layout()
        fig2.savefig(pathsave + fig_name + '.pdf')
        fig2.savefig(pathsave + fig_name + '.png')

        ax3.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax3.set_ylabel("NO2 wet at 15% O2 (ppm)")
        ax3.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "NO2_exhaust_steel" + "_vs_phi"
        fig3.tight_layout()
        fig3.savefig(pathsave + fig_name + '.pdf')
        fig3.savefig(pathsave + fig_name + '.png')

        ax4.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax4.set_ylabel("O2 (%)")
        ax4.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "O2_exhaust_steel" + "_vs_phi"
        fig4.tight_layout()
        fig4.savefig(pathsave + fig_name + '.pdf')
        fig4.savefig(pathsave + fig_name + '.png')

        ax5.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax5.set_ylabel("CH4 dry at 15% O2 (ppm)")
        ax5.set_xlabel("Equivalence Ratio ($\phi$)")
        ax5.set_yscale('log')
        fig_name = "CH4_exhaust_steel" + "_vs_phi"
        fig5.tight_layout()
        fig5.savefig(pathsave + fig_name + '.pdf')
        fig5.savefig(pathsave + fig_name + '.png')

        ax6.legend(xlegend, markerscale=mkr_sz_leg, title= leg_title)
        ax6.set_ylabel("Excess O$_2$ (%)")
        ax6.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "O2_excess_exhaust_steel" + "_vs_phi"
        fig6.tight_layout()
        fig6.savefig(pathsave + fig_name + '.pdf')
        fig6.savefig(pathsave + fig_name + '.png')

        ax7.legend(xlegend, markerscale=mkr_sz_leg, title=leg_title)
        ax7.set_ylabel("Cooling air (kg/s)")
        ax7.set_xlabel("Equivalence Ratio ($\phi$)")
        fig_name = "Coolingair_exhaust_quartz" + "_vs_phi"
        fig7.tight_layout()
        fig7.savefig(pathsave + fig_name + '.pdf')
        fig7.savefig(pathsave + fig_name + '.png')


if __name__=="__main__":
    TCT = TC_Trav()
    #TCT.tdms_file_search()
    port_list = [2, 3, 5]
    H2_list = [0, 50, 80,100]
    """for port in port_list:
        for H2 in H2_list:
            TCT.data_plot(port, H2)"""

    #TCT.data_plot_gascomp_exh()
    TCT.data_comparison()