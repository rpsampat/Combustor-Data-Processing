from nptdms import TdmsFile
import matplotlib.pyplot as plt
from scandir import scandir, walk

class DataExtract:
    def __init__(self):
        # 'TC9'=='Flang Temp', 'TC11'=='Frame Temp'
        self.drive = "P:/"
        self.folder = "Combustor_Temperature and gas composition/Stype_old_thin_characterisation/"
        self.position_range = 9
        self.start = 2
        self.quant = ["Measured - Air Secondary","Measured - CH4 Secondary","'Raw TC Data'/'TC15'","Offset","Position"]#,"Avg Exhaust Temp"]
        self.addon = "_burnerraised"
        self.template = lambda x:"Pos"+str(x)+"_TC15"
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

    def search_tdms_file(self,path,identifier):
        name_return ="None"
        for entry in scandir(path):
            names= entry.name
            if (".tdms" in names) and (identifier in names) and (self.addon in names) and not("_index" in names):
                name_return = str(names)
                break

        return name_return

    def plot_data(self,data_dict):
        fig, ax = plt.subplots()
        ax.plot(data_dict["Measured - Air Secondary"])
        ax.plot(data_dict["Measured - CH4 Secondary"])
        fig2, ax2 = plt.subplots()
        ax2.plot(data_dict["Position"])
        fig3, ax3 = plt.subplots()
        ax3.plot(data_dict["Raw TC Data/TC15"])

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Temperature (C)")
        ax3.legend(self.quant)
        plt.grid(True)
        plt.show()

    def main(self):
        path = self.drive + self.folder
        data_dict = {}
        for i in range(self.position_range):
            if i<self.start:
                continue
            data_dict[i] = {}
            h = self.template(i)
            filename = self.search_tdms_file(path,self.template(i))
            #print filename
            group = self.read_tdms(path+filename)
            for q in self.quant:
                key = self.key_extract(q,group)
                data_dict[i][q] = group[key].data
        #self.plot_data(data_dict[i])
        return data_dict

if __name__=="__main__":
    extract = DataExtract()
    extract.main()