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
        identifiers = ["NV100_", "H2_50", "_phi_",]
        identifier_exclude = ["_index"]
        subdir_list = next(os.walk(path))[1]  # list of immediate subdirectories within the data directory
        sub_list=[]
        for subdir in subdir_list:
            check_id = [(x in subdir) for x in identifiers]
            check_id_exclude = [(x in subdir) for x in identifier_exclude]
            isfalse = False in check_id
            isfalse_exclude = False in check_id_exclude
            if not(isfalse) and isfalse_exclude:
                sub_list.append(subdir)

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
                blue_img = np.array(img_thresh, dtype=int)  #
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

    def cluster(self):
        dict_pdf={'volume':[],'x_com':[],'y_com':[],'Lxx':[],'Lyy':[],'xmin':[]}
        path = self.drive + self.folder
        file_loc = path+'NV100_H2_0_phi_0.6/Variance/'
        filenames = next(os.walk(file_loc))[2]
        # Iterating through files for a particular case
        count = 0
        for name in filenames:
            count += 1
            print("Count=",count)
            if count ==50:
                break
            if not ('var' in name):
                continue
            img = cv2.imread(file_loc+name)
            # Equalising histogram and extracting points above a certain threshold which are
            #  expected to represent flame structures
            img_thresh = self.image_enhancement(img,False,lower_lim=150)
            img_ind = np.where(img_thresh>0)
            #self.edge_detect(img)
            # meshgrid
            shp = img.shape
            x0 = range(shp[1])
            y0 = range(shp[0])
            xv, yv = np.meshgrid(x0,y0)
            Z = np.ndarray(shape=(len(img_ind[0]),2))
            Z[:,0] =img_ind[1]
            Z[:,1] =img_ind[0]
            #Z[:,:,2] = yv
            """cv2.namedWindow('Avg', cv2.WINDOW_NORMAL)
            cv2.imshow('Avg', img)
            cv2.waitKey(0)"""
            #Z = Z.reshape((-1, 3))
            eps=10.0
            ms=200
            db = DBSCAN(eps=eps, min_samples=ms).fit(Z)#, algorithm='ball_tree'
            #km = KMeans(n_clusters=10).fit(Z)
            cluster_ind = np.where(db.labels_>=0)[0]
            num_cluster,unique_counts = np.unique(db.labels_,return_counts=True)
            #print("Unique clusters=",num_cluster)
            #print("Unique counts=",unique_counts)
            for i in range(len(num_cluster)):
                #ignoring cluster id -1
                if i==0:
                    continue
                clust_id = num_cluster[i]
                clust = unique_counts[i]
                dict_pdf['volume'].append(clust)
                cluster_ind = np.where(db.labels_ >= clust_id)[0]
                xval = img_ind[1][cluster_ind]
                yval = img_ind[0][cluster_ind]
                dict_pdf['x_com'].append(np.mean(xval))
                dict_pdf['y_com'].append(np.mean(yval))
                Lxx = np.max(xval)-np.min(xval)
                Lyy = np.max(yval)-np.min(yval)
                dict_pdf['Lxx'].append(Lxx)
                dict_pdf['Lyy'].append(Lyy)
                dict_pdf['xmin'].append(np.min(xval))



        """plt.scatter(img_ind[1][cluster_ind],img_ind[0][cluster_ind],c=db.labels_[cluster_ind],s=1.0)
        plt.colorbar()"""
        #plt.imshow(np.uint8(db.labels_))
        with open(file_loc + "Cluster_stats", 'wb') as file:
            pickle.dump(dict_pdf, file, pickle.HIGHEST_PROTOCOL)
        fig,ax= plt.subplots()
        hist, bin_edge = np.histogram(dict_pdf['volume'],bins=50)
        ax.plot(hist)
        fig2, ax2 = plt.subplots()
        hist2, bin_edge2 = np.histogram(dict_pdf['x_com'], bins=50)
        ax2.plot(hist2)
        fig3, ax3 = plt.subplots()
        hist3, bin_edge3 = np.histogram(dict_pdf['y_com'], bins=50)
        ax3.plot(hist3)
        fig4, ax4 = plt.subplots()
        hist4, bin_edge4 = np.histogram(dict_pdf['Lxx'], bins=50)
        ax4.plot(hist4)
        fig5, ax5 = plt.subplots()
        hist5, bin_edge5 = np.histogram(dict_pdf['Lyy'], bins=50)
        ax5.plot(hist5)
        fig6, ax6 = plt.subplots()
        hist6, bin_edge6 = np.histogram(dict_pdf['xmin'], bins=50)
        ax6.plot(hist6)
        plt.show()

        """plt.imshow(np.uint8(db.labels_.reshape((shp[0],shp[1]))))
        plt.show()"""

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
            return thresh4
        else:
            return thresh4#gray_img_eqhist

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
    ImgProc.main()
    #ImgProc.cluster()



