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
class ImageProcessing:
    def __init__(self):
        # 'TC9'=='Flang Temp', 'TC11'=='Frame Temp'
        self.drive = 'O:/'
        self.folder = "FlamelessCombustor_Jan2023/DSLR/OldTube_DSLR_UV_GasAnalyser/"
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
        x1 = 801.139
        y1 = 2565.29
        x2 = 3672.99
        y2 = 2569.72
        dist_px = math.sqrt((x2-x1)**2.0+(y2-y1)**2.0)
        dist_mm = 240.0
        scale = dist_mm/dist_px
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
        identifiers = ["NV100_", "H2_","_phi_",]
        identifier_exclude = ["_index","N2","CO2"]
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
        sub_list = self.image_dir_list()#['NV100_H2_0_phi_0.6','NV100_H2_10_phi_0.6','NV100_H2_50_phi_0.6','NV100_H2_80_phi_0.6','NV100_H2_100_phi_0.6']#
        exception = 0
        for subdir in sub_list:
            path_file = path  + subdir+ '/'
            print(path_file)
            filenames = next(os.walk(path_file))[2]
            if len(filenames)<5:
                continue
            if not os.path.exists(path_file+'Variance'):
                os.makedirs(path_file+'Variance')
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
                    avg_img = np.add(avg_img, blue_img)
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
            self.save_image_named(avg_img, 'avg_'+str(count), path_file + 'Variance/')
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
                self.save_image_named(var_img, 'var' + str(count), path_file + 'Variance/')

                """if count==3:
                    break"""

            stdv_img = np.power(stdv_img/count,1/2.0)
            self.save_image_named(stdv_img, 'stdv_'+str(count), path_file + 'Variance/')

            """cv2.namedWindow('Avg', cv2.WINDOW_NORMAL)
            cv2.imshow('Avg',stdv_img)
            cv2.waitKey(0)"""

    def main_comparison(self):
        """
        Plotting a concise comparison of images of different conditions of operation
        :return:
        """
        H2_perc = [0, 10,50,80,100]
        phi = [0.3,0.35,0.5,0.6,0.7, 0.8, 0.9,1.0]
        N2_perc = [0,15,11]
        figure,ax = plt.subplots(len(phi),len(H2_perc),sharex=True, sharey=True, dpi=300, gridspec_kw={'wspace': 0.01, 'hspace': 0.01})#, figsize=(96,18))#,
                         #gridspec_kw={'wspace': 0.01, 'hspace': 0.01})# figsiz=(24,18)
        path = self.drive + self.folder
        sub_list = self.image_dir_list()
        folder = sub_list[0]+ "/" + "Variance/"
        with open(path + folder + 'avg_146.jpg', 'rb') as file:
            img = plt.imread(file)
        img_blank =img*0+255

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
                    img_thresh,img_gray = self.image_enhancement(img,False,220)
                    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) # image normalisation to improve visibility of final plot
                    ax[j, i].imshow(img_normalized,vmin=0,vmax=255, aspect='auto')#[0:-48,85:-1,:]
                    ax[j, i].axis('off')
                    ax[j,i].set_frame_on(False)
                    if j==0:
                        ax[j, i].set_title(str(H2_perc[i]))
                    if i==0:
                        ax[j, i].set_ylabel(str(phi[j]))
                        ax[j, i].axis('on')
                        ax[j, i].set_xticks([])
                        ax[j, i].set_yticks([])
                        ax[j, i].xaxis.set_tick_params(labelbottom=False)
                        ax[j, i].yaxis.set_tick_params(labelleft=False)


                except:
                    ax[j, i].imshow(img_blank, cmap='gray', aspect='auto')  # , aspect='auto')#[0:-48,85:-1,:]
                    ax[j, i].axis('off')
                    ax[j, i].set_frame_on(False)
                    if j==0:
                        ax[j, i].set_title(str(H2_perc[i]))
                    if i==0:
                        ax[j, i].set_ylabel(str(phi[j]))
                        ax[j, i].axis('on')
                        ax[j, i].set_xticks([])
                        ax[j, i].set_yticks([])
                        ax[j, i].xaxis.set_tick_params(labelbottom=False)
                        ax[j, i].yaxis.set_tick_params(labelleft=False)
                    continue
        #figure.tight_layout()
        plt.savefig(path + 'avg_comparison.png', bbox_inches='tight')
        plt.show()

    def cluster(self):
        """
        Identify clusters using the DBSCAN algorithm. This prcoess can either be done on the variance images or
        the raw images directly, depending on the folder chosen and the name condition.
        :return:
        """
        dict_pdf={'volume':[],'x_com':[],'y_com':[],'Lxx':[],'Lyy':[],'xmin':[],'spacing':[],'hydrau_dia':[],'aspect_ratio':[]}
        path = self.drive + self.folder
        #file_loc = path+'NV100_H2_100_phi_1.0/'
        #path = self.drive + self.folder
        sub_list = self.image_dir_list()  # ['NV100_H2_0_phi_0.6','NV100_H2_10_phi_0.6','NV100_H2_50_phi_0.6','NV100_H2_80_phi_0.6','NV100_H2_100_phi_0.6']#
        exception = 0
        plot_cluster_image='n'
        settings={0:{'thresh':60,'eps':8.0,'minpts':20},10:{'thresh':60,'eps':8.0,'minpts':20},
                  50:{'thresh':60,'eps':8.0,'minpts':150},80:{'thresh':20,'eps':8.0,'minpts':100},
                  100:{'thresh':50,'eps':8.0,'minpts':125}}
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
                check_id = '_H2_'+str(ind)
                if check_id in subdir:
                    break
            # Iterating through files for a particular case
            count = 0
            for name in filenames:
                """if count ==6:
                    break"""
                if not ('var' in name):
                    #if "Cluster_stats" in name:
                     #   os.remove(file_loc+name)
                    continue
                count += 1
                print("Count=", count)
                """
                The blue componenet of the image is chosen. Next it is converted to grey scale and then
                normalised by its own min-max values.
                """
                img = cv2.imread(file_loc+name)
                shp = np.shape(img)
                img[:, :, 1] = np.zeros((shp[0], shp[1]))
                img[:, :, 2] = np.zeros((shp[0], shp[1]))
                # Equalising histogram and extracting points above a certain threshold which are
                #  expected to represent flame structures
                #img_thresh = self.image_enhancement(img,False,lower_lim=150)
                #img_ind = np.where(img_thresh>0)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                n = 0
                while (n < 2):
                    img = cv2.pyrDown(img)
                    n = n + 1
                img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                if plot_cluster_image=='y':
                    fig,ax = plt.subplots()
                    ax.imshow(img,cmap='gray')
                    fig.savefig(path + 'vargray_' + subdir + '_' + name, bbox_inches='tight')
                #plt.show()
                img_ind = np.where(img_normalized >settings[ind]['thresh'])
                #self.edge_detect(img)
                # meshgrid
                shp = img.shape
                print("Shape=",shp)
                x0 = range(shp[1])
                y0 = range(shp[0])
                xv, yv = np.meshgrid(x0,y0)
                #print("Shape xv=",xv.shape)
                """
                Array created by depth stacking the x and y coordinates. The array is further reduced by only choosing
                the parts of the array for which the corresponding intensities satisfy a threshold value.
                """
                dbscan_arr = np.dstack((xv,yv))
                #dbscan_arr = np.dstack((dbscan_arr, img_normalized))
                dbscan_arr=dbscan_arr[img_ind[0],img_ind[1]]
                print("Dbscan arr shape=",dbscan_arr.shape)
                Z = np.reshape(dbscan_arr,[-1, 2])
                #print("Z shape =",Z.shape)

                eps=settings[ind]['eps']
                ms=settings[ind]['minpts']
                db = DBSCAN(eps=eps, min_samples=ms).fit(Z)#, algorithm='ball_tree'
                #km = KMeans(n_clusters=10).fit(Z)
                cluster_ind = np.where(db.labels_>=0)[0]
                num_cluster,unique_counts = np.unique(db.labels_,return_counts=True)
                #print("name=",name)
                if plot_cluster_image=='y':
                    sc = ax.scatter(img_ind[1][cluster_ind], img_ind[0][cluster_ind], c=db.labels_[cluster_ind], s=0.001)
                    cax = fig.add_axes(
                        [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
                    fig.colorbar(sc, cax = cax)
                    fig.savefig(path + 'dbscan_'+subdir+'_'+name, bbox_inches='tight')
                    #plt.show()
                #print("Unique clusters=",num_cluster)
                #print("Unique counts=",unique_counts)
                core_samp = list(db.core_sample_indices_)
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
                    x_loc = np.mean(xval)
                    y_loc = np.mean(yval)
                    dict_pdf['x_com'].append(x_loc)
                    dict_pdf['y_com'].append(y_loc)
                    Lxx = np.max(xval)-np.min(xval)
                    Lyy = np.max(yval)-np.min(yval)
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



            """plt.scatter(img_ind[1][cluster_ind],img_ind[0][cluster_ind],c=db.labels_[cluster_ind],s=1.0)
            plt.colorbar()"""
            #plt.imshow(np.uint8(db.labels_))
            with open(file_loc + "Cluster_stats2_"+str(count), 'wb') as file:
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
        N2_perc = [0, 15, 11]
        path = self.drive + self.folder
        sub_list = self.image_dir_list()
        for i in range(len(H2_perc)):
            figure, ax = plt.subplots()
            leg=[]
            for j in range(len(phi)):
                folder = "NV100_H2_" + str(H2_perc[i]) + "_phi_" + str(phi[j])
                folder = folder + "/" + "Variance/"
                try:
                    filenames = next(os.walk(path + folder))[2]
                    for nm in filenames:
                        if 'Cluster_stats2' in nm:
                            break
                    with open(path + folder + nm, 'rb') as file:
                        dict_pdf = pickle.load(file)
                    minval = np.min(dict_pdf[pdf_param])
                    maxval = np.max(dict_pdf[pdf_param])
                    hist, bin_edge = np.histogram(dict_pdf[pdf_param], bins=50,
                                                  density=True)# , weights=dict_pdf['volume'])
                    pdf_x = (bin_edge[0:-1] + bin_edge[1:]) / 2.0
                    ax.plot(pdf_x, hist)
                    leg.append(phi[j])


                except:
                    #print("Phi=",phi[j])
                    continue
            ax.legend(leg)
            ax.set_ylabel("Probability density")
            ax.set_xlabel(param)
            #plt.show()
            figure.savefig(path+'Cluster_varimage/' + 'pdf_' + pdf_param + '_variance_H2_'+str(H2_perc[i])+'.png', bbox_inches='tight')
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
        file_loc = path + 'NV100_H2_100_phi_1.0/'
        with open(file_loc + "Cluster_stats_146", 'rb') as file:
            dict_pdf = pickle.load(file)

        fig, ax = plt.subplots()
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
        ax6.set_ylabel("Frequency")
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
    ImgProc.main_comparison()
    #ImgProc.cluster()
    #ImgProc.pdf_plot()
    """pdf_param =['volume', 'x_com', 'y_com', 'Lxx', 'Lyy', 'xmin','spacing','hydrau_dia','aspect_ratio']
    for param in pdf_param:
        ImgProc.pdf_comparison_h2percwise(param)"""



