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


def dbscan(img, red_level, dbscan_thresh, epsilon, minpts, plot_img):
    n = 0
    while (n < red_level):
        img = cv2.pyrDown(img)
        n = n + 1
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img_ind = np.where(img_normalized > dbscan_thresh)
    # self.edge_detect(img)
    # meshgrid
    shp = img.shape
    print("Shape=", shp)
    x0 = range(shp[1])
    y0 = range(shp[0])
    xv, yv = np.meshgrid(x0, y0)
    # print("Shape xv=",xv.shape)
    """
    Array created by depth stacking the x and y coordinates. The array is further reduced by only choosing
    the parts of the array for which the corresponding intensities satisfy a threshold value.
    """
    dbscan_arr = np.dstack((xv, yv))
    # dbscan_arr = np.dstack((dbscan_arr, img_normalized))
    dbscan_arr = dbscan_arr[img_ind[0], img_ind[1]]
    print("Dbscan arr shape=", dbscan_arr.shape)
    Z = np.reshape(dbscan_arr, [-1, 2])
    # print("Z shape =",Z.shape)

    eps = epsilon
    ms = minpts
    db = DBSCAN(eps=eps, min_samples=ms).fit(Z)  # , algorithm='ball_tree'
    # km = KMeans(n_clusters=10).fit(Z)
    cluster_ind_all = np.where(db.labels_ >= 0)
    num_cluster, unique_counts = np.unique(db.labels_, return_counts=True)
    # max_cluster = np.where(unique_counts==max(unique_counts))[0]-1#-1 to offset -1 category which is noise
    cluster_ind = cluster_ind_all  # np.where(db.labels_ == max_cluster[0])[0]

    # print("name=",name)
    if plot_img == 'y':
        fig, ax = plt.subplots()
        sc = ax.scatter(img_ind[1][cluster_ind_all], img_ind[0][cluster_ind_all], c=db.labels_[cluster_ind_all], s=1.0)
        cax = fig.add_axes(
            [ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        fig.colorbar(sc, cax=cax)
        # ax.imshow(img_normalized)
        ax.set_title("Clusters")
        # fig.savefig(file_loc + 'dbscan_img/' + 'dbscan_' + subdir + '_' + name, bbox_inches='tight')
    # print("Unique clusters=",num_cluster)
    # print("Unique counts=",unique_counts)
    core_samp = list(db.core_sample_indices_)
    shp_norm = img_normalized.shape
    img_arr = np.ones((shp_norm[0], shp_norm[1]))
    img_arr[img_ind[0][cluster_ind], img_ind[1][cluster_ind]] = 0

    # currently returning only largest cluster, so most likely upper edge interface

    return img_arr, img_ind, cluster_ind, num_cluster, unique_counts, db


def arr2img(arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    scale_fact = (255 - 0) / (max_val - min_val)
    arr_scale = np.subtract(arr, min_val)
    arr_scale = arr_scale * scale_fact
    arr_scale = (arr_scale).astype('uint8')

    return arr_scale

def edge_extract(img, kernel_blur, plot_img):
    # shp_orig = img.shape
    # extent = [0,shp_orig[1]*self.scale,0,shp_orig[0]*self.scale]
    # pyramidal image reduction
    n = 0
    while (n < 0):
        img = cv2.pyrDown(img)
        n = n + 1
    """plt.subplots()
    plt.imshow(img)
    plt.colorbar()"""
    edges = cv2.Canny(image=img, threshold1=10, threshold2=50, apertureSize=3)

    """edges_copy = np.copy(edges)
    edges_copy[314,:] = 255
    #edges_copy[0,314] = 255
    edge_det = np.where(edges_copy == 255)
    edge_loc = list(zip(edge_det[1],edge_det[0]))
    tri = Delaunay(edge_loc)
    plt.subplots()
    plt.triplot(edge_det[1],edge_det[0],tri.simplices)
    plt.plot(edge_det[1],edge_det[0],'o')
    plt.show()"""
    # edge_blur = cv2.GaussianBlur(edges, (7, 7), 0)
    if plot_img == 'y':
        plt.subplots()
        plt.imshow(img)  # ,extent = extent)
        plt.title("Image for Edge detection")
        plt.colorbar()
        plt.subplots()
        plt.imshow(edges)  # ,extent = extent)
        plt.title("Edges")
        plt.colorbar()
        # plt.show()
    # Contour identification and searching for longest continuous contour line
    cnt_max = 0
    contour_long = []
    edge_draw = np.array(
        edges)  # define another image object as contourdraw function uses the input image as destination image to draw over.
    contours, hierarchy = cv2.findContours(edge_draw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # hiererchy=[Next, Previous, First_Child, Parent]
    # img_cnt = edges
    count_cont = 0
    shp_count = 0
    # plt.subplots()
    for i in range(len(contours)):
        cnt = contours[i]
        if cnt.shape[0] > 2:#and hierarchy[0][i][3]==-1 and hierarchy[0][i][2]==-1: # closed loop contour always has 1 child
            contour_long.append(cnt)
            # img_cnt = cv2.drawContours(edges, [cnt], 0, (255, 0, 0), 1)
            # plt.imshow(img_cnt)
        if cnt.shape[0] > shp_count:
            shp_count = cnt.shape[0]
            cnt_max = cnt
            count_cont += 1
    # img_cnt = cv2.drawContours(img, [cnt_max], 0, (255, 0, 0), 3)
    """edge_contour={}
    edge_Coord = np.where(edges==255)
    x_edge  =edge_Coord[1]
    y_edge = edge_Coord[0]
    shp_contour = np.shape(contour_long)
    for i in range(contour_long.__len__()):
        edge_contour[i]=[]
        for j in range(len(contour_long[i])):
            x_cont = contour_long[i][j][0][0]
            y_cont = contour_long[i][j][0][1]
            xedge_loc = np.where(x_edge==x_cont)[0]
            yedge = y_edge[xedge_loc]
            if y_cont in yedge:
                edge_contour[i].append([y_cont,x_cont])"""

    # plt.imshow(img_cnt)#,extent = extent)
    if plot_img == 'y':
        plt.subplots()
        for i in range(len(contour_long)):
            # if i==0 :
            #   continue
            # plt.plot(edge_contour[i][:,0],edge_contour[i][:,1])
            img_cnt = cv2.drawContours(edge_draw, [contour_long[i]], 0, (255, 0, 255), 5)
        plt.imshow(img_cnt)
        plt.title("Contours")
        # plt.colorbar()

        plt.subplots()
        plt.imshow(edges)  # ,extent = extent)
        plt.title("Edges after contour detect")
        plt.colorbar()
        # plt.show()

    """plt.subplots()
    img_cnt1 = cv2.drawContours(edges, [contour_long[1]], 0, (255, 0, 0), 1)
    plt.imshow(img_cnt1)
    plt.colorbar()
    print(contour_long[0].shape)
    print(contour_long[1].shape"""
    # plt.show()
    # cv2.imshow("Shapes", img_cnt)
    # cv2.waitKey(0)

    return cnt_max, contour_long, edges, img  # , extent