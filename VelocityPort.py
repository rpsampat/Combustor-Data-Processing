"""Extract velocity along radial direction of probe port from measured PIV fields. PIV data extracted from reduced
average images processed fro AGNES Proletariat"""
import numpy as np
import matplotlib.pyplot as plt
import pickle

class VelocityPort:
    def __init__(self):
        self.piv_path = 'C:/Users/rishikeshsampa/Documents/Research/AGNES_proletariat/'
        self.port_dist = {1:40,2:90,3:140,4:190,5:240,6:290} # Axial Distance of port in mm from head end of steel pipe
        self.dia_probe = 3 # mm
        self.case = {1: "phi06", 2: "phi08", 3: "phi08_N2"}

    def velocityextract(self):
        case_num=3
        fig,ax=plt.subplots()
        name = self.case[case_num]
        path = self.piv_path+name+'/'
        with open(path+"velocitymatrix.dat") as ifile:
            count = 0
            for line_count in ifile:
                line = ifile.readline()
                # print line
                if len(line) == 0:
                    print("Finished reading file\n")
                    break
                ln = line.split()
                x1 = float(ln[0])
                y1 = float(ln[1])
                u1 = float(ln[2])
                v1 = float(ln[3])
                try:
                    x = np.append(x,x1)
                    y = np.append(y, y1)
                    u = np.append(u, u1)
                    v = np.append(v, v1)
                except:
                    x = np.array([x1])
                    y = np.array([y1])
                    u = np.array([u1])
                    v = np.array([v1])

                #count = count + 1
        x = x-np.min(x)
        y = y-np.min(y)
        y = np.max(y)-y
        print(self.port_dist.keys())
        for port_num in list(self.port_dist.keys()):
            port_index = np.where(np.abs(x-self.port_dist[port_num])<self.dia_probe/2.0)[0]
            y_ind = y[port_index]
            u_ind = u[port_index]
            v_ind = v[port_index]
            dict_save={"y":y_ind,"u":u_ind,"v":v_ind}
            with open(name+"_port"+str(port_num), 'wb') as file:
                pickle.dump(dict_save, file, pickle.HIGHEST_PROTOCOL)
            ax.scatter(y_ind,u_ind)
        plt.show()

    def velocity_plot(self):
        fig, ax = plt.subplots()
        name = "phi06"
        for port in list(self.port_dist.keys()):
            with open(name + "_port" + str(port), 'rb') as f:
                dict_port = pickle.load(f)
            ax.scatter(dict_port["y"],dict_port["u"])
        ax.legend(list(self.port_dist.keys()), title="Port")
        fig.savefig("velocity_"+name+".png")


if __name__=="__main__":
    VP = VelocityPort()
    #VP.velocityextract()
    VP.velocity_plot()


