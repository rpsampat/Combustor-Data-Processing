import math
import cantera as ct
import numpy as np
import pickle
from scandir import scandir, walk
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,KMeans
import EdgeDetection as ED
from SavitzkyGolay2D import sgolay2d

class ImageProcessing:
    def __init__(self):
        # 'TC9'=='Flang Temp', 'TC11'=='Frame Temp'
        self.drive = 'O:/'#'P:/'#
        self.folder = "FlamelessCombustor_Jan2023/DSLR/OldTube_DSLR_UV_GasAnalyser/"#"ExhaustModification_June2022/ModifiedExhaust_20220602/DSLR_ExhaustModification_2022_06_02/"#
        self.scaling_folder = "Scaling images with light on/"
        self.scale = self.scaling()

    def search_file(self,path):
        file_list = []
        for entry in scandir(path):
            names= entry.name
            if (".jpg" in names):
                file_list.append(names)

        return file_list

    def scaling(self):
        """
        Manually entered scaling value by giving (x1,y1) and (x2,y2) from a calibration image and entering the known distance between them.
        :return:
        """
        """path = self.drive + self.folder + "Calibration_20230130/"
        name="calibration_0129.jpg"
        img = cv2.imread(path + name)  # bgr
        img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plt.subplots()
        plt.imshow(img)
        plt.show()"""

        x1 = 155.0#801.139
        y1 = 1921.0#2565.29
        x2 = 3431.0#3672.99
        y2 = 2191.0#2569.72
        dist_px = math.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0)
        dist_div = 200#50.0
        div2mm = 29.0 / 33.0
        dist_mm = dist_div * div2mm
        scale = dist_mm / dist_px
        return scale

    def save_image_named(self,img,name,path):
        shp = np.shape(img)
        cv2.imwrite(path + name + '.jpg', img)

    def image_dir_list(self):
        """
        Identify and extract image directory list for processing
        :return:
        """
        path = self.drive + self.folder
        identifiers = ["NV100_", "H2_100","_phi_"]
        identifier_exclude = ["_index","N2","CO2","_phi_1.0"]
        identifier_optional = ["H2_"]
        subdir_list = next(os.walk(path))[1]  # list of immediate subdirectories within the data directory
        sub_list=[]
        for subdir in subdir_list:
            check_id = [(x in subdir) for x in identifiers]
            check_id_exclude = [(x in subdir) for x in identifier_exclude]
            check_id_optional = [(x in subdir) for x in identifier_optional]
            isfalse = False in check_id
            isTrue_exclude = True in check_id_exclude
            isTrue = True in check_id_optional
            if not(isfalse) and not(isTrue_exclude) and isTrue:
                sub_list.append(subdir)

        print(sub_list)
        return sub_list



    def main(self):
        # name = 'Img1311.jpg'
        path = self.drive + self.folder
        #sub_list = self.image_dir_list()#['NV100_H2_0_phi_0.6','NV100_H2_10_phi_0.6','NV100_H2_50_phi_0.6','NV100_H2_80_phi_0.6','NV100_H2_100_phi_0.6']#
        sub_list = self.image_dir_list()  #['60_kW_phi0.9_ss_5000']#'60_kW_phi0.5_ss_1250']
        exception = 0
        folder_save = 'Variance'#'MinSub'/'Variance'
        for subdir in sub_list:
            path_file = path  + subdir+ '/'
            print(path_file)
            filenames = next(os.walk(path_file))[2]
            if len(filenames)<5:
                continue
            if not os.path.exists(path_file+folder_save):
                os.makedirs(path_file+folder_save)
            # Iterating through files for a particular case
            count = 0
            for name in filenames:
                if "Cluster_stats" in name:
                    continue
                count += 1
                img = cv2.imread(path_file + name)  # bgr

                shp = np.shape(img)
                img[:, :, 1] = np.zeros((shp[0], shp[1]))
                img[:, :, 2] = np.zeros((shp[0], shp[1]))
                #if count==1:
                 #   min_arr = np.ones((shp[0], shp[1]))*255
                """img[:,:,1] = np.zeros((shp[0],shp[1]))
                img[:, :, 2] = np.zeros((shp[0], shp[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"""
                #xlim = img.shape[1]
                # img = img[:,xlim/5:xlim/2,:]
                #cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
                #cv2.imshow('original',img)
                img_thresh = img#self.image_enhancement(img, False,lower_lim=230)
                blue_img = np.array(img_thresh, dtype=int)  # change dtype to int, which is 64 bit integer
                # for numpy else the default uint8 of opencv will cause an overflow while adding images for averages.
                try:
                    if folder_save=='Variance':
                        avg_img = np.add(avg_img, blue_img)
                    else:
                        min_ind = np.where(blue_img < avg_img)
                        avg_img[min_ind] = blue_img[min_ind]

                except:
                    avg_img = blue_img
                    exception += 1
                """cv2.namedWindow('original', cv2.WINDOW_NORMAL)
                cv2.imshow('original', img)
                cv2.waitKey(0)"""

                """if count==30:
                    break"""



            if count==0:
                continue
            avg_img = avg_img/(count)
            """cv2.namedWindow('original', cv2.WINDOW_NORMAL)
            cv2.imshow('original', avg_img)
            cv2.waitKey(0)"""
            self.save_image_named(avg_img, 'avg_'+str(count), path_file + folder_save+'/')
            count = 0
            for name in filenames:
                if "Cluster_stats" in name:
                    continue
                count += 1
                img = cv2.imread(path_file + name)  # bgr
                img[:, :, 1] = np.zeros((shp[0], shp[1]))
                img[:, :, 2] = np.zeros((shp[0], shp[1]))
                """img[:, :, 1] = np.zeros((shp[0], shp[1]))
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)"""
                # cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('original',img)
                blue_img = np.array(img, dtype=int)  # [:,:,0]
                var_img = np.subtract(blue_img, avg_img)
                try:
                    stdv_img = np.add(stdv_img, np.power(var_img,2.0))
                except:
                    stdv_img = np.power(var_img,2.0)
                self.save_image_named(var_img, 'var' + str(count), path_file + folder_save+'/')

                """if count==3:
                    break"""

            stdv_img = np.power(stdv_img/count,1/2.0)
            self.save_image_named(stdv_img, 'stdv_'+str(count), path_file + folder_save+'/')

            """cv2.namedWindow('Avg', cv2.WINDOW_NORMAL)
            cv2.imshow('Avg',stdv_img)
            cv2.waitKey(0)"""

    def main_comparison(self):
        """
        Plotting a concise comparison of images of different conditions of operation
        :return:
        """
        H2_perc = [0, 10,50,80,100]
        phi = [0.6,0.8,1.0]#[0.3,0.35,0.5,0.6,0.7, 0.8, 0.9,1.0]
        N2_perc = [0,15,11]
        px = 1.0/300.0
        figure,ax = plt.subplots(len(phi),len(H2_perc),sharex=True, sharey=True, dpi=300, gridspec_kw={'wspace': 0.1, 'hspace': 0.0025}, figsize=(10,4.5))#(24,18))#(10,4.5))#,
                         #gridspec_kw={'wspace': 0.01, 'hspace': 0.01})# figsiz=
        path = self.drive + self.folder
        sub_list = self.image_dir_list()
        folder = sub_list[0]+ "/" + "Variance/"
        with open(path + folder + 'avg_146.jpg', 'rb') as file:
            img = plt.imread(file)

        tick_size = 4.0
        aspect = 'equal'  # '#float(shp0[0])/float(shp0[1])
        img_blank =img*0+255
        pyr_scale =1
        n = 0
        while (n < pyr_scale-1):
            img_blank = cv2.pyrDown(img_blank)
            n = n + 1
        shp0 = np.shape(img_blank)

        #self.scale=1.0

        for i in range(len(H2_perc)):
            for j in range(len(phi)):
                folder = "NV100_H2_"+str(H2_perc[i])+"_phi_"+str(phi[j])
                """identifier_exclude = ["_index", "N2", "CO2"]
                identifier_optional = ["H2_"]
                folder = 0
                for subdir in sub_list:
                    check_id = [(x in subdir) for x in identifiers]
                    check_id_exclude = [(x in subdir) for x in identifier_exclude]
                    check_id_optional = [(x in subdir) for x in identifier_optional]
                    isfalse = False in check_id
                    isfalse_exclude = False in check_id_exclude
                    isTrue = True in check_id_optional
                    if not (isfalse) and isfalse_exclude and isTrue:
                        folder = subdir
                        break"""
                try:
                    filenames = next(os.walk(path + folder))[2]
                    count_files=0
                    for nm in filenames:
                        if '.jpg' in nm:
                            count_files+=1
                    if count_files<5:
                        raise Exception
                    folder = folder + "/" + "Variance/"

                    with open(path+ folder + 'avg_'+str(count_files)+'.jpg', 'rb') as file:
                        img = plt.imread(file)
                    #img_thresh,img_gray = self.image_enhancement(img,False,220)
                    n = 0
                    while (n < pyr_scale-1):
                        img = cv2.pyrDown(img)
                        n = n + 1
                    shp = np.shape(img)
                    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) # image normalisation to improve visibility of final plot
                    ax_img = ax[j, i].imshow(img_normalized,vmin=0,vmax=255, aspect=aspect,origin='lower',extent=[0,shp[1],0,shp[0]])#[0:-48,85:-1,:]
                    #ax[j, i].set_xlim((0, shp[1]))
                    #img_ex = ax_img.extent
                    ax[j, i].axis('off')
                    ax[j,i].set_frame_on(False)
                    if j==0:
                        ax[j, i].set_title(str(H2_perc[i]))
                    if i == 0 and j < len(phi) - 1:
                        #ax[j, i].set_frame_on(True)
                        ax[j, i].set_ylabel(str(phi[j])+"\n"+"$\\regular_{Y (mm)}$")
                        ax[j, i].axis('on')
                        ticky0 = np.linspace(0,shp[0],len(ax[j, i].get_yticks()))
                        #ticky0_min = np.min(ticky0)
                        ticky1 = np.round((ticky0) * self.scale*pyr_scale, decimals=2)
                        #ticky0 = ticky0 - ticky0_min
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=1.0)
                        ax[j, i].xaxis.set_tick_params(labelbottom=False,width=0)
                        #ax[j, i].yaxis.set_tick_params(labelleft=False,width=0)
                    if i>0 and j==len(phi)-1:
                        #ax[j, i].set_frame_on(True)
                        ax[j, i].set_xlabel("$\\regular_{X (mm)}$")
                        ax[j, i].axis('on')
                        tickx0 = np.linspace(0,shp[1],len(ax[j, i].get_xticks()))
                        #tickx0_min = np.min(tickx0)
                        tickx1 = np.round((tickx0) * self.scale*pyr_scale, decimals=1)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        #tickx0 = tickx0 - tickx0_min
                        ax[j, i].set_xticks(ticks=tickx0, labels=tickx1)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=1.0)
                        ax[j, i].yaxis.set_tick_params(labelleft=False, width=0)
                    if i==0 and j==len(phi)-1:
                        #ax[j, i].set_frame_on(True)
                        ax[j, i].set_ylabel(str(phi[j])+"\n"+"$\\regular_{Y (mm)}$")
                        ax[j, i].set_xlabel("$\\regular_{X (mm)}$")
                        ax[j, i].axis('on')
                        tickx0 = np.linspace(0,shp[1],len(ax[j, i].get_xticks()))
                        #tickx0_min = np.min(tickx0)
                        tickx1 = np.round((tickx0) * self.scale*pyr_scale,decimals=1)
                        #tickx0=tickx0-tickx0_min
                        ticky0 = np.linspace(0,shp[0],len(ax[j, i].get_yticks()))
                        #ticky0_min = np.min(ticky0)
                        ticky1 = np.round((ticky0) * self.scale*pyr_scale, decimals=1)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        #ticky0 = ticky0 - ticky0_min
                        ax[j, i].set_xticks(ticks=tickx0,labels=tickx1)
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].tick_params(axis='both',which='both', labelsize=tick_size, width=1.0)

                except:
                    ax[j, i].imshow(img_blank, cmap='gray', aspect=aspect,origin='lower',extent=[0,shp0[1],0,shp0[0]])  # , aspect='auto')#[0:-48,85:-1,:]
                    ax[j, i].axis('off')
                    ax[j, i].set_frame_on(False)
                    if j == 0:
                        ax[j, i].set_title(str(H2_perc[i]))
                    if i == 0 and j < len(phi) - 1:
                        # ax[j, i].set_frame_on(True)
                        ax[j, i].set_ylabel(str(phi[j]) + "\n" + "$\\regular_{Y (mm)}$")
                        ax[j, i].axis('on')
                        ticky0 = np.linspace(0, shp[0], len(ax[j, i].get_yticks()))
                        # ticky0_min = np.min(ticky0)
                        ticky1 = np.round((ticky0) * self.scale * pyr_scale, decimals=1)
                        # ticky0 = ticky0 - ticky0_min
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=1.0)
                        ax[j, i].xaxis.set_tick_params(labelbottom=False, width=0)
                        # ax[j, i].yaxis.set_tick_params(labelleft=False,width=0)
                    if i > 0 and j == len(phi) - 1:
                        # ax[j, i].set_frame_on(True)
                        ax[j, i].set_xlabel("$\\regular_{X (mm)}$")
                        ax[j, i].axis('on')
                        tickx0 = np.linspace(0, shp[1], len(ax[j, i].get_xticks()))
                        # tickx0_min = np.min(tickx0)
                        tickx1 = np.round((tickx0) * self.scale * pyr_scale, decimals=1)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        # tickx0 = tickx0 - tickx0_min
                        ax[j, i].set_xticks(ticks=tickx0, labels=tickx1)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=1.0)
                        ax[j, i].yaxis.set_tick_params(labelleft=False, width=0)
                    if i == 0 and j == len(phi) - 1:
                        # ax[j, i].set_frame_on(True)
                        ax[j, i].set_ylabel(str(phi[j]) + "\n" + "$\\regular_{Y (mm)}$")
                        ax[j, i].set_xlabel("$\\regular_{X (mm)}$")
                        ax[j, i].axis('on')
                        tickx0 = np.linspace(0, shp[1], len(ax[j, i].get_xticks()))
                        # tickx0_min = np.min(tickx0)
                        tickx1 = np.round((tickx0) * self.scale * pyr_scale, decimals=1)
                        # tickx0=tickx0-tickx0_min
                        ticky0 = np.linspace(0, shp[0], len(ax[j, i].get_yticks()))
                        # ticky0_min = np.min(ticky0)
                        ticky1 = np.round((ticky0) * self.scale * pyr_scale, decimals=1)
                        ax[j, i].set_xlim((tickx0[0], tickx0[-1]))
                        # ticky0 = ticky0 - ticky0_min
                        ax[j, i].set_xticks(ticks=tickx0, labels=tickx1)
                        ax[j, i].set_yticks(ticks=ticky0, labels=ticky1)
                        ax[j, i].tick_params(axis='both', which='both', labelsize=tick_size, width=1.0)

                    continue
        figure.suptitle("Hydrogen %")
        figure.text(0.025,0.33,"Equivalence Ratio ($\phi$)", rotation='vertical')
        plt.savefig(path + 'avg_comparison_reduced.png', bbox_inches='tight')
        #plt.show()



    def cluster(self):
        """
        Identify clusters using the DBSCAN algorithm. This process can either be done on the variance images or
        the raw images directly, depending on the folder chosen and the name condition.
        :return:
        """
        #dict_pdf={'volume':[],'x_com':[],'y_com':[],'Lxx':[],'Lyy':[],'xmin':[],'spacing':[],'hydrau_dia':[],'aspect_ratio':[]}
        path = self.drive + self.folder
        #file_loc = path+'NV100_H2_100_phi_1.0/'
        #path = self.drive + self.folder
        sub_list = self.image_dir_list()  # ['NV100_H2_0_phi_0.6','NV100_H2_10_phi_0.6','NV100_H2_50_phi_0.6','NV100_H2_80_phi_0.6','NV100_H2_100_phi_0.6']#
        #sub_list = ['60_kW_phi0.9_ss_5000']
        exception = 0
        plot_cluster_image='y'
        settings={0:{'thresh':60,'eps':8.0,'minpts':20},10:{'thresh':60,'eps':8.0,'minpts':20},
                  50:{'thresh':60,'eps':8.0,'minpts':150},80:{'thresh':20,'eps':8.0,'minpts':100},
                  100:{'thresh':50,'eps':8.0,'minpts':125}}#125
        for subdir in sub_list:
            file_loc = path + subdir + '/' + 'Variance/'
            print(file_loc)
            filenames = next(os.walk(file_loc))[2]
            if len(filenames) < 5:
                continue
            check_clust = ["Cluster_stats" in x for x in filenames]
            """if plot_cluster_image=='n':
                if True in check_clust:
                    continue"""
            for ind in settings.keys():
                check_id = '_H2_'+str(ind)+'_'
                if check_id in subdir:
                    break
            if not os.path.exists(file_loc+'dbscan_img'):
                os.makedirs(file_loc+'dbscan_img')
            #Stats
            stdv_name = np.array([filenames[i].find('stdv_') for i in range(len(filenames))])
            stdv_ind = np.where(stdv_name == 0)[0][0]
            stdv = cv2.imread(file_loc+filenames[stdv_ind])
            stdv_max = np.max(stdv)
            print("Stdv max=",stdv_max)
            stdv_mean = np.mean(np.mean(stdv[:,:,0],axis=0))
            stdv_std = np.std(np.std(stdv[:,:,0],axis=0))
            print("Stdv of stdv=",stdv_std)
            print("Stdv mean=",stdv_mean)
            if stdv_std>stdv_mean/3.0:
                thresh_criteria = int(stdv_mean-stdv_std)
            else:
                thresh_criteria = int(stdv_mean/4.0)
            # Iterating through files for a particular case
            count = 0
            dict_pdf = {'volume': [], 'x_com': [], 'y_com': [], 'Lxx': [], 'Lyy': [], 'xmin': [], 'spacing': [],
                        'hydrau_dia': [], 'aspect_ratio': [], 'rect_properties':[],'num_cluster':[]}
            for name in filenames:
                """if count ==6:
                    break"""
                if not ('var' in name):
                    if "Cluster_stats" in name:
                        os.remove(file_loc+name)
                    continue
                """if "Cluster_stats" in name:
                    os.remove(file_loc+name)
                    continue"""
                count += 1
                print("Count=", count)
                """
                The blue component of the image is chosen. Next it is converted to grey scale and then
                normalised by its own min-max values.
                """
                img = cv2.imread(file_loc+name)
                shp = np.shape(img)
                if check_id=="_H2_100_":
                    #Including green and red channel for 100% H2 case due to faint signal
                    pass
                else:
                    img[:, :, 1] = np.zeros((shp[0], shp[1]))
                    img[:, :, 2] = np.zeros((shp[0], shp[1]))
                # Equalising histogram and extracting points above a certain threshold which are
                #  expected to represent flame structures
                #img_thresh = self.image_enhancement(img,False,lower_lim=150)
                #img_ind = np.where(img_thresh>0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_thresh_loc = np.where(img<thresh_criteria)
                img[img_thresh_loc] = 0
                n = 0
                while (n <2):
                    img = cv2.pyrDown(img)
                    n = n + 1
                #img = sgolay2d(img, window_size=3, order=2)
                #img = ED.arr2img(img)
                #img = cv2.normalize((img), None, 0, 255, cv2.NORM_MINMAX)
                img_normalized = cv2.normalize((img), None, 0, 255, cv2.NORM_MINMAX)#255-img
                if plot_cluster_image=='y' and count<5:
                    fig,ax = plt.subplots(dpi=600)
                    ax.imshow(img,cmap='gray')
                    fig.savefig(file_loc +'dbscan_img/'+ 'gray_' + subdir + '_' + name, bbox_inches='tight')
                #plt.show()
                otsu_threshold, otsu_image_result = cv2.threshold(img_normalized, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
                print("Cluster Otsu=", otsu_threshold)
                try:
                    img_cluster_mask,img_ind,cluster_ind_all,num_cluster,unique_counts,db = ED.dbscan(img_normalized, red_level=0, dbscan_thresh=otsu_threshold,
                                               epsilon=settings[ind]['eps'], minpts=settings[ind]['minpts'], plot_img=plot_cluster_image)
                except:
                    continue
                """img_cluster_mask2,img_ind,cluster_ind,num_cluster,unique_counts,db = ED.dbscan(img_cluster_mask, red_level=0, dbscan_thresh=125, epsilon=settings[ind]['eps'], minpts=settings[ind]['minpts'],
                                                plot_img=plot_cluster_image)
                img_proc3 = ED.arr2img(img_cluster_mask2)
                cnt_max, contours, edges, img_blur = ED.edge_extract(img_proc3, kernel_blur=1, plot_img=plot_cluster_image)"""
                #print("name=",name)
                if plot_cluster_image=='y' and count<5:
                    sc = ax.scatter(img_ind[1][cluster_ind_all], img_ind[0][cluster_ind_all], c=db.labels_[cluster_ind_all], s=0.001)
                    cax = fig.add_axes(
                        [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
                    fig.colorbar(sc, cax = cax)
                    fig.savefig(file_loc +'dbscan_img/'+ 'dbscan_'+subdir+'_'+name, bbox_inches='tight')

                    #fig2, ax2 = plt.subplots()
                    fig3, ax3 = plt.subplots(dpi=600)
                    #img_draw = np.copy(img)
                    img_draw2 = np.copy(img_normalized)
                    """for i in range(len(contours)):
                        # if i==0 :
                        #   continue
                        # plt.plot(edge_contour[i][:,0],edge_contour[i][:,1])
                        img_cnt = cv2.drawContours(img_draw2, [contours[i]], 0, (255, 0, 255), 3)
                        rect = cv2.minAreaRect(contours[i])
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        img_box = cv2.drawContours(img_draw2, [box], 0, (0, 0, 255), 2)"""
                    #ax2.imshow(img_cnt,cmap='gray')
                    #ax3.imshow(img_box,cmap='gray')
                    #plt.show()
                #print("Unique clusters=",num_cluster)
                #print("Unique counts=",unique_counts)
                core_samp = list(db.core_sample_indices_)
                dict_pdf['num_cluster'].append(len(num_cluster))
                for i in range(len(num_cluster)):
                    # iterating over each cluster detected in this image
                    #ignoring cluster id -1
                    if i==0:
                        continue
                    clust_id = num_cluster[i]
                    clust = unique_counts[i]
                    dict_pdf['volume'].append(clust)
                    cluster_ind = np.where(db.labels_== clust_id)
                    edge_ind = set(list(cluster_ind[0]))-set(core_samp)
                    xval = img_ind[1][cluster_ind] # xlocation of all points in this cluster
                    yval = img_ind[0][cluster_ind] # ylocation of all points in this
                    coords = list(zip(xval,yval))
                    coords = np.reshape(coords,(len(xval),1,2))
                    rect = cv2.minAreaRect(coords)#((xcenter,ycenter),(width,height),angle of rotation)
                    dict_pdf['rect_properties'].append(rect)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    if plot_cluster_image == 'y' and count < 5:
                        img_box = cv2.drawContours(img_draw2, [box], 0, (255,0, 255), 3)
                    x_loc = np.mean(xval)
                    y_loc = np.mean(yval)
                    dict_pdf['x_com'].append(x_loc)
                    dict_pdf['y_com'].append(y_loc)
                    Lxx = np.max(xval)-np.min(xval)#math.sqrt((box[0][0]-box[1][0])**2.0+(box[0][1]-box[1][1])**2.0)#
                    Lyy =  np.max(yval)-np.min(yval)#math.sqrt((box[1][0]-box[2][0])**2.0+(box[1][1]-box[2][1])**2.0)#
                    dict_pdf['Lxx'].append(Lxx)
                    dict_pdf['Lyy'].append(Lyy)
                    try:
                        dict_pdf['aspect_ratio'].append(float(Lxx)/float(Lyy))
                    except:
                        dict_pdf['aspect_ratio'].append(float(Lxx) / 1.0)
                    #dict_pdf['hydrau_dia'].append(float(clust) / float(len(edge_ind)))
                    dict_pdf['xmin'].append(np.min(xval))
                    for j in range(len(num_cluster)):
                        if j==i:
                            continue
                        clust_id2 = num_cluster[j]
                        cluster_ind2 = np.where(db.labels_== clust_id2)
                        xval2 = img_ind[1][cluster_ind2]  # xlocation of all points in this cluster
                        yval2 = img_ind[0][cluster_ind2]  # ylocation of all points in this cluster
                        x_loc2 = np.mean(xval2)
                        y_loc2 = np.mean(yval2)
                        spacing = math.sqrt((x_loc-x_loc2)**2.0 + (y_loc-y_loc2)**2.0)
                        dict_pdf["spacing"].append(spacing)

                if plot_cluster_image == 'y' and count < 5:

                    sc = ax3.scatter(img_ind[1][cluster_ind_all], img_ind[0][cluster_ind_all],
                                    c=db.labels_[cluster_ind_all], s=0.3)
                    cax = fig.add_axes(
                        [ax3.get_position().x1 + 0.01, ax3.get_position().y0, 0.02, ax3.get_position().height])
                    ax3.imshow(img_box, cmap='gray')
                    fig3.colorbar(sc, cax=cax)
                    fig3.savefig(file_loc + 'dbscan_img/' + 'gray_box_' + subdir + '_' + name, bbox_inches='tight')
                    #plt.show()
                plt.close()



            """plt.scatter(img_ind[1][cluster_ind],img_ind[0][cluster_ind],c=db.labels_[cluster_ind],s=1.0)
            plt.colorbar()"""
            #plt.imshow(np.uint8(db.labels_))

            with open(file_loc + "Cluster_stats_"+str(count), 'wb') as file:
                pickle.dump(dict_pdf, file, pickle.HIGHEST_PROTOCOL)
            """fig,ax= plt.subplots()
            hist, bin_edge = np.histogram(dict_pdf['volume'],bins=50)
            ax.plot(hist)
            ax.set_xlabel("Volume")
            ax.set_ylabel("Frequency")
            fig2, ax2 = plt.subplots()
            hist2, bin_edge2 = np.histogram(dict_pdf['x_com'], bins=50)
            ax2.plot(hist2)
            ax2.set_xlabel("X location")
            ax2.set_ylabel("Frequency")
            fig3, ax3 = plt.subplots()
            hist3, bin_edge3 = np.histogram(dict_pdf['y_com'], bins=50)
            ax3.plot(hist3)
            ax3.set_xlabel("Y location")
            ax3.set_ylabel("Frequency")
            fig4, ax4 = plt.subplots()
            hist4, bin_edge4 = np.histogram(dict_pdf['Lxx'], bins=50)
            ax4.plot(hist4)
            ax4.set_xlabel("Cluster expanse in X direction")
            ax4.set_ylabel("Frequency")
            fig5, ax5 = plt.subplots()
            hist5, bin_edge5 = np.histogram(dict_pdf['Lyy'], bins=50)
            ax5.set_xlabel("Cluster expanse in y direction")
            ax5.set_ylabel("Frequency")
            ax5.plot(hist5)
            fig6, ax6 = plt.subplots()
            hist6, bin_edge6 = np.histogram(dict_pdf['xmin'], bins=50)
            ax6.plot(hist6)
            ax6.set_xlabel("Cluster minimum X location")
            ax6.set_ylabel("Frequency")
            plt.show()"""

            """plt.imshow(np.uint8(db.labels_.reshape((shp[0],shp[1]))))
            plt.show()"""

    def pdf_comparison_h2percwise(self, pdf_param):
        """
               Plotting a concise comparison of pdfs of different conditions of operation
               :return:
               """
        H2_perc = [0, 10, 50, 80, 100]
        phi = [0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        param_label={'aspect_ratio':'Aspect Ratio', 'x_com':'X$_{COM}$ (pixels)', 'Lxx':'L$_{xx}$ (pixels)',
                     'Lyy':'L$_{yy}$ (pixels)','y_com':'Y$_{COM}$ (pixels)','spacing':'Spacing',
                     'volume':'Volume(pixels)','hydrau_dia':'Hydraulic Diameter','xmin':'X$_{min}$',
                     'rect_properties':"Rect Prop",'num_cluster':"Number of Clusters","rot_angle":"Rotation Angle(deg"}
        N2_perc = [0, 15, 11]
        path = self.drive + self.folder
        #sub_list = self.image_dir_list()
        label_size = 18.0
        tick_size = 12.0
        dict_data={}
        dict_x = {}
        dict_leg={}
        minx = 0
        maxx = 0
        x_arr=[]
        pyr_scale=3
        for i in range(len(H2_perc)):
            #figure, ax = plt.subplots(dpi=600)
            leg=[]
            dict_data[H2_perc[i]]= []
            dict_x[H2_perc[i]] = []
            dict_leg[H2_perc[i]]=[]
            for j in range(len(phi)):
                folder = "NV100_H2_" + str(H2_perc[i]) + "_phi_" + str(phi[j])
                folder = folder + "/" + "Variance/"
                try:
                    filenames = next(os.walk(path + folder))[2]
                    for nm in filenames:
                        if 'Cluster_stats' in nm:
                            break
                    with open(path + folder + nm, 'rb') as file:
                        dict_pdf = pickle.load(file)
                    #minval = np.min(dict_pdf[pdf_param])
                    #maxval = np.max(dict_pdf[pdf_param])
                    red_vol = np.array(dict_pdf['volume'])
                    red_Lxx = np.array(dict_pdf['Lxx'])
                    red_Lyy = np.array(dict_pdf['Lyy'])
                    red_ind = np.where(red_vol>500)[0]#10000np.where(red_Lxx>100)[0]#
                    red_ycom = np.array(dict_pdf['y_com'])
                    red_ind2 = np.where(red_Lxx>50)[0]#np.where(red_ycom>0)[0]
                    red_ind_tot = list(set.intersection(set(red_ind),set(red_ind2)))
                    bin=20
                    if pdf_param == 'aspect_ratio':
                        rect_prop = np.array(dict_pdf['rect_properties'])
                        unzipped = list(zip(*rect_prop[:,1]))
                        width = np.array(unzipped[0])
                        height = np.array(unzipped[1])
                        red_val = np.array(width/height)[red_ind_tot]
                        ind_invt = np.where(red_val<1.0)[0]
                        red_val[ind_invt] = 1.0/red_val[ind_invt]
                    elif pdf_param=='rot_angle':
                        rect_prop = np.array(dict_pdf['rect_properties'])
                        angle = np.array(rect_prop[:, 2])
                        red_val = angle[red_ind_tot]
                    elif pdf_param=='x_com':
                        #rect_prop = np.array(dict_pdf['rect_properties'])
                        #unzipped = list(zip(*rect_prop[:,1]))
                        #red_val = np.array(unzipped[0])[red_ind_tot]*self.scale*pyr_scale
                        red_val = np.array(dict_pdf[pdf_param])[red_ind_tot]*self.scale*pyr_scale
                    elif pdf_param=='num_cluster':
                        red_val = np.array(dict_pdf[pdf_param])
                        bin=10
                    else:
                        red_val = np.array(dict_pdf[pdf_param])[red_ind_tot]
                    red_vol_hist = red_vol[red_ind_tot]
                    hist, bin_edge = np.histogram(red_val, bins=bin,
                                                  density=True)#, weights=red_vol[red_ind_tot])
                    hist_vol, bin_edge_vol = np.histogram(red_vol_hist, bins=50,
                                                  density=True)
                    pdf_x = (bin_edge[0:-1] + bin_edge[1:]) / 2.0
                    pdf_x_vol = (bin_edge_vol[0:-1] + bin_edge_vol[1:]) / 2.0
                    try:
                        dict_data[H2_perc[i]] = np.dstack((dict_data[H2_perc[i]],hist))
                        dict_x[H2_perc[i]] = np.dstack((dict_x[H2_perc[i]],pdf_x))
                    except:
                        dict_data[H2_perc[i]] = np.array(hist)
                        dict_x[H2_perc[i]] = np.array(pdf_x)
                    #dict_data[H2_perc[i]][phi(j)] = np.array(hist)
                    #dict_x[H2_perc[i]][phi(j)] = np.array(pdf_x)
                    dict_leg[H2_perc[i]].append(phi[j])
                    x_arr.append(pdf_x)


                except:
                    #print("Phi=",phi[j])
                    continue

        for i in range(len(H2_perc)):
            figure, ax = plt.subplots(dpi=600)
            xset = np.squeeze(np.array(dict_x[H2_perc[i]]))
            yset = np.squeeze(np.array(dict_data[H2_perc[i]]))
            minx = 0#np.min(x_arr)
            maxx = np.max(x_arr)
            ax.plot(xset,yset)
            mkr_sz_leg = 2.0
            ax.set_xlim((minx,maxx))
            ax.legend(dict_leg[H2_perc[i]], title='$\phi$',title_fontsize=13.0,markerscale=mkr_sz_leg, fontsize=12.0,)
            ax.set_ylabel("Probability density", fontsize=label_size)
            ax.set_xlabel(param_label[pdf_param], fontsize=label_size)
            ax.tick_params(axis='both', labelsize=tick_size, width=3.0)
            figure.tight_layout()
            #plt.show()
            figure.savefig(path+'Cluster_varimage/'+'y_com_and_volume_condition/' + 'pdf_' + pdf_param + '_H2_'+str(H2_perc[i])+'.png', bbox_inches='tight',dpi=600)
            plt.close(figure)

        # ax.set_xlabel("X location")
        # ax.set_ylabel("Frequency")
        # figure.tight_layout()
        #plt.savefig(path + 'pdf_' + pdf_param + '_variance_comparison.png', bbox_inches='tight')  # volumeweighted_
        # plt.show()

    def pdf_comparison_subplots(self, pdf_param):
        """
        Plotting a concise comparison of pdfs of different conditions of operation
        :return:
        """
        H2_perc = [0, 10, 50, 80,100]
        phi = [0.3,0.35,0.5,0.6,0.7, 0.8, 0.9,1.0]
        N2_perc = [0,15,11]
        figure,ax = plt.subplots(len(phi),len(H2_perc),sharex=True, sharey=False, dpi=300, gridspec_kw={'wspace': 0.5, 'hspace': 0.05})#, figsize=(96,18))#,
                         #gridspec_kw={'wspace': 0.01, 'hspace': 0.01})# figsiz=(24,18)
        plt.rcParams['xtick.labelsize']= 0.01
        plt.rcParams['ytick.labelsize'] = 0.01
        path = self.drive + self.folder
        sub_list = self.image_dir_list()
        folder = sub_list[0]+ "/"
        """with open(path + folder + 'avg_146.jpg', 'rb') as file:
            img = plt.imread(file)
        img_blank =img*0"""

        for i in range(len(H2_perc)):
            for j in range(len(phi)):
                folder = "NV100_H2_"+str(H2_perc[i])+"_phi_"+str(phi[j])
                folder = folder + "/" + "Variance/"
                try:
                    filenames = next(os.walk(path + folder))[2]
                    for nm in filenames:
                        if 'Cluster_stats' in nm:
                            break
                    with open(path+ folder + nm, 'rb') as file:
                        dict_pdf = pickle.load(file)
                    hist, bin_edge = np.histogram(dict_pdf[pdf_param], bins=50, density=True)#, weights=dict_pdf['volume'])
                    pdf_x = (bin_edge[0:-1] + bin_edge[1:]) / 2.0
                    ax[j, i].plot(pdf_x,hist)
                    ax[j, i].xaxis.set_tick_params(labelsize=2)
                    ax[j, i].yaxis.set_tick_params(labelsize=2)
                    if j==0:
                        ax[j, i].set_title(str(H2_perc[i]), fontsize=3)
                    if i==0:
                        ax[j, i].set_ylabel(str(phi[j]), fontsize=3)
                        #ax[j, i].set_ylabel("Frequency")
                    if j== len(phi)-1:
                        #ax[j, i].set_xlabel("X location")
                        #ax[j, i].axis('on')
                        """ax[j, i].set_xticks([])
                        ax[j, i].set_yticks([])"""


                except:
                    ax[j, i].xaxis.set_tick_params(labelsize=2)
                    ax[j, i].yaxis.set_tick_params(labelsize=2)
                    if j==0:
                        ax[j, i].set_title(str(H2_perc[i]), fontsize=3)
                    if i==0:
                        ax[j, i].set_ylabel(str(phi[j]), fontsize=3)
                        #ax[j, i].set_ylabel("Frequency")
                    if j == len(phi) - 1:
                        #ax[j, i].set_xlabel("X location")
                        """ax[j, i].axis('on')
                        ax[j, i].set_xticks([])
                        ax[j, i].set_yticks([])"""
                    continue
        #ax.set_xlabel("X location")
        #ax.set_ylabel("Frequency")
        #figure.tight_layout()
        plt.savefig(path + 'pdf_'+pdf_param+'_variance_comparison.png', bbox_inches='tight')#volumeweighted_
        #plt.show()
    def pdf_plot(self):
        path = self.drive + self.folder
        sub_list = ['60_kW_phi0.9_ss_5000','60_kW_phi0.5_ss_500']
        file_loc = path + sub_list[0]+'/Variance/'#'NV100_H2_100_phi_1.0/'
        filenames = next(os.walk(file_loc))[2]
        label_size = 15.0
        tick_size = 9.0
        for name in filenames:
            if "Cluster_stats" in name:
                break
        with open(file_loc + name, 'rb') as file:
            dict_pdf = pickle.load(file)
        param_label = {'aspect_ratio': 'Aspect Ratio', 'x_com': 'X$_{COM}$ (pixels)', 'Lxx': 'L$_{xx}$ (pixels)'}
        for pdf_param in param_label:
            figure, ax = plt.subplots(dpi=600)
            red_vol = np.array(dict_pdf['volume'])
            red_ind = np.where(red_vol > 10)[0]  # 10000
            red_ycom = np.array(dict_pdf['y_com'])
            red_ind2 = np.where(red_ycom > 0)[0]
            red_ind_tot = list(set.intersection(set(red_ind), set(red_ind2)))
            red_val = np.array(dict_pdf[pdf_param])[red_ind_tot]
            hist, bin_edge = np.histogram(red_val, bins=50,
                                          density=True)  # , weights=red_vol[red_ind_tot])
            pdf_x = (bin_edge[0:-1] + bin_edge[1:]) / 2.0
            ax.plot(pdf_x, hist)
            ax.set_ylabel("Probability density", fontsize=label_size)
            ax.set_xlabel(param_label[pdf_param], fontsize=label_size)
            ax.tick_params(axis='both', labelsize=tick_size, width=3.0)
            figure.tight_layout()
            # plt.show()
            #figure.savefig(
             #   path + 'Cluster_varimage/' + 'y_com_and_volume_condition/' + 'pdf_' + pdf_param + '.png', bbox_inches='tight', dpi=300)
            #plt.close(figure)
        """fig, ax = plt.subplots()
        hist, bin_edge = np.histogram(dict_pdf['volume'], bins=50, density=True)
        #print(len(bin_edge))
        pdf_x = (bin_edge[0:-1] + bin_edge[1:]) / 2.0
        ax.plot(pdf_x,hist)
        ax.set_xlabel("Volume")
        ax.set_ylabel("Frequency")
        fig2, ax2 = plt.subplots()
        hist2, bin_edge2 = np.histogram(dict_pdf['x_com'], bins=50, density=True)
        pdf_x = (bin_edge2[0:-1] + bin_edge2[1:]) / 2.0
        ax2.plot(pdf_x,hist2)
        ax2.set_xlabel("X location")
        ax2.set_ylabel("Frequency")
        fig3, ax3 = plt.subplots()
        hist3, bin_edge3 = np.histogram(dict_pdf['y_com'], bins=50, density=True)
        pdf_x = (bin_edge3[0:-1] + bin_edge3[1:]) / 2.0
        ax3.plot(pdf_x,hist3)
        ax3.set_xlabel("Y location")
        ax3.set_ylabel("Frequency")
        fig4, ax4 = plt.subplots()
        hist4, bin_edge4 = np.histogram(dict_pdf['Lxx'], bins=50, density=True)
        pdf_x = (bin_edge4[0:-1] + bin_edge4[1:]) / 2.0
        ax4.plot(pdf_x,hist4)
        ax4.set_xlabel("Cluster expanse in X direction")
        ax4.set_ylabel("Frequency")
        fig5, ax5 = plt.subplots()
        hist5, bin_edge5 = np.histogram(dict_pdf['Lyy'], bins=50, density=True)
        pdf_x = (bin_edge5[0:-1] + bin_edge5[1:]) / 2.0
        ax5.set_xlabel("Cluster expanse in y direction")
        ax5.set_ylabel("Frequency")
        ax5.plot(pdf_x,hist5)
        fig6, ax6 = plt.subplots()
        hist6, bin_edge6 = np.histogram(dict_pdf['xmin'], bins=50, density=True)
        pdf_x = (bin_edge6[0:-1] + bin_edge6[1:]) / 2.0
        ax6.plot(pdf_x,hist6)
        ax6.set_xlabel("Cluster minimum X location")
        ax6.set_ylabel("Frequency")"""
        plt.show()

    def image_enhancement(self,img,show,lower_lim):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist(gray_img, [0], None, [256], [0, 256])
        gray_img_eqhist = cv2.equalizeHist(gray_img)
        hist_eqhist = cv2.calcHist(gray_img_eqhist, [0], None, [256], [0, 256])
        #otsu_threshold, otsu_image_result = cv2.threshold(gray_img, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        #print(otsu_threshold)
        ret, thresh4 = cv2.threshold(gray_img_eqhist, lower_lim, 255, cv2.THRESH_TOZERO)# use 150 for cluster
        if show==True:
            plt.subplot(121)
            plt.title("Image1")
            plt.xlabel('bins')
            plt.ylabel("No of pixels")
            plt.plot(hist)
            plt.subplot(122)
            plt.plot(hist_eqhist)
            plt.show()
            plt.show()
            cv2.namedWindow('Enhanced image', cv2.WINDOW_NORMAL)
            cv2.imshow('Enhanced image', thresh4)
            cv2.waitKey(0)
            return thresh4,gray_img_eqhist
        else:
            return thresh4,gray_img_eqhist

    def edge_detect(self,img):
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY,dstCn=256)
        except:
            img_gray = img
            print( "Exception gray")

        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
        # Use a bimodal image as an input.
        # Optimal threshold value is determined automatically.
        otsu_threshold, otsu_image_result = cv2.threshold(img_gray, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        new_img = np.multiply(otsu_image_result,img_gray)
        #normalizedImg = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX)
        hist = cv2.calcHist(new_img, [0], None, [256], [0, 256])
        gray_img_eqhist = cv2.equalizeHist(new_img)
        hist_eqhist = cv2.calcHist(gray_img_eqhist, [0], None, [256], [0, 256])
        """plt.subplot(121)
        plt.title("Image1")
        plt.xlabel('bins')
        plt.ylabel("No of pixels")
        plt.plot(hist)
        plt.subplot(122)
        plt.plot(hist_eqhist)
        plt.show()
        cv2.namedWindow('Equi hist', cv2.WINDOW_NORMAL)
        cv2.imshow('Equi hist', gray_img_eqhist)"""
        #self.save_image_named(new_img, 'otsu' + str(count), path + '/Otsu_Filtered/')


        #Improved Otsu

        #Adaptive threshold
        adaptive_thresh_result = cv2.adaptiveThreshold(img_blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        # Sobel XY edge detection
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1,
                            ksize=5)  # Combined X and Y Sobel Edge Detection
        # Canny Edge Detection
        edges = cv2.Canny(image=img_gray, threshold1=10,threshold2=50)
        kernel3 = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
        sharp_img = cv2.filter2D(src=img_gray, ddepth=-1, kernel=kernel3)
        #Gradient
        # set the kernel size, depending on whether we are using the Sobel
        # operator of the Scharr operator, then compute the gradients along
        # the x and y axis, respectively
        ksize = 3
        gX = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
        gY = cv2.Sobel(img_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
        # the gradient magnitude images are now of the floating point data
        # type, so we need to take care to convert them back a to unsigned
        # 8-bit integer representation so other OpenCV functions can operate
        # on them and visualize them
        gX = cv2.convertScaleAbs(gX)
        gY = cv2.convertScaleAbs(gY)
        # combine the gradient representations into a single image
        combined = cv2.pow(cv2.addWeighted(gX, 0.5, gY, 0.5, 0),2.0)
        #laplacian
        laplacian = cv2.Laplacian(img_gray,ddepth=cv2.CV_64F)

        cv2.namedWindow('Original Gray Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Gray Image', img_gray)
        # Display Canny Edge Detection Image
        cv2.namedWindow('Canny Edge Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Canny Edge Detection', edges)
        cv2.namedWindow('Otsu Edge Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Otsu Edge Detection', otsu_image_result)
        cv2.namedWindow('Otsu Filtered', cv2.WINDOW_NORMAL)
        cv2.imshow('Otsu Filtered', new_img)
        #cv2.namedWindow('Gradient', cv2.WINDOW_NORMAL)
        #cv2.imshow('Gradient', combined)
        #cv2.namedWindow('Adaptive threshold', cv2.WINDOW_NORMAL)
        #cv2.imshow('Adaptive threshold', adaptive_thresh_result)
        #cv2.namedWindow('Sobel Edge Detection', cv2.WINDOW_NORMAL)
        #cv2.imshow('Sobel Edge Detection', sobelxy)
        #cv2.namedWindow('Sharpened image', cv2.WINDOW_NORMAL)
        #cv2.imshow('Sharpened image', sharp_img)
        cv2.waitKey(0)


if __name__=="__main__":
    ImgProc = ImageProcessing()
    #ImgProc.main()
    #ImgProc.main_comparison()
    #ImgProc.cluster()
    #ImgProc.pdf_plot()
    pdf_param =['x_com']#['rot_angle','num_cluster','volume', 'x_com', 'y_com', 'Lxx', 'Lyy', 'xmin','spacing','hydrau_dia','aspect_ratio']#['aspect_ratio', 'x_com', 'Lxx']#['aspect_ratio']#
    for param in pdf_param:
        ImgProc.pdf_comparison_h2percwise(param)


