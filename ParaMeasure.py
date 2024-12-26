import pandas as pd
import imagej
import scyjava as sj
from collections import defaultdict
from pandas import DataFrame
import imagej.doctor
import numpy as np
import math
import torch

# imagej.doctor.checkup()
# imagej.doctor.debug_to_stderr()
pd.set_option('display.max_rows', None)     # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列


# 创建测量盒子维数的盒子划分
def create_string(n):
  nums = [2**i for i in range(n+1)]
  if 4 in nums:
      idx = nums.index(4)
      nums.insert(idx, 3)
  return f"box={','.join(map(str,nums))}"


class PyimageJ():

    def __init__(self, path='C:\\Users\\DELL\\Fiji.app'):
        # initialize ImageJ2
        self.ij = imagej.init(path, mode='interactive')
        print(f"ImageJ2 version: {self.ij.getVersion()}")
        # self.ui_show = True
        self.ui_show = False

    def __del__(self):
        pass

    def boneJ_2D_Thickness(self, img):
        self.ij.IJ.setAutoThreshold(img, "Default dark no-reset")
        self.ij.IJ.run(img, "Convert to Mask", "")
        if self.ui_show:
            self.ij.ui().show(img)

        self.ij.IJ.run(img, "Particle Analyser", "thickness min=0.000 max=Infinity surface_resampling=2 surface=Gradient split=0.000 volume_resampling=2 clear")

        results = self.ij.ResultsTable.getResultsTable()
        stats_ij = defaultdict(list)

        # for column in results.getHeadings() and ['ID', 'Vol. (pixels³)', 'x Cent (pixels)', 'y Cent (pixels)',
        #                                          'Thickness (pixels)', 'SD Thickness (pixels)', 'Max Thickness (pixels)']:
        for column in results.getHeadings() and ['ID', 'Vol. (pixels³)', 'Thickness (pixels)']:
            for i in range(len(results.getColumn(column))):
                stats_ij[column].append(results.getColumn(column)[i])

        # df_ij = DataFrame(stats_ij)
        # print(df_ij)
        Vol_list = stats_ij['Vol. (pixels³)']
        max_index = Vol_list.index(max(Vol_list))
        thickness_2D = stats_ij['Thickness (pixels)'][max_index]

        return thickness_2D


    def open_Img(self, input_tensor):
        if isinstance(input_tensor, torch.Tensor):
            imp = input_tensor.cpu().detach().numpy()
            imp = np.squeeze(imp)
            imp[imp > 0] = 255
            imp[imp <= 0] = 0
            imp = imp.astype(np.uint8)
            # plt.imshow(imp)
            # plt.show()
        else:
            imp = self.ij.io().open(input_tensor.replace('/', '\\'))

        # jimp = self.ij.py.to_imageplus(imp)
        if self.ui_show:
            # ij.py.show(imp)
            self.ij.ui().show(imp)
        return imp

    def get_Bone_Paras(self, imp):
        jimp = self.ij.py.to_imageplus(imp)
        if self.ui_show:
            self.ij.ui().show(jimp)

        # self.ij.IJ.run(jimp, "8-bit", "")

        thickness_2D = self.boneJ_2D_Thickness(jimp.duplicate())

        self.ij.IJ.setAutoThreshold(jimp, "Default no-reset")
        self.ij.IJ.run(jimp, "Convert to Mask", "")

        # Measure each cell, accumulating results into a dictionary.
        self.ij.IJ.run("Set Measurements...", "area feret's redirect=None decimal=3")
        results = self.ij.ResultsTable.getResultsTable()
        stats_ij = defaultdict(list)

        self.ij.IJ.run(jimp, "Analyze Particles...", "clear add")

        rois = self.ij.RoiManager.getRoiManager()
        # print("Number of ROIs:", rois.getCount())

        for roi_index in range(rois.getCount()):
            for column in results.getHeadings() and ['Area', 'Feret', 'MinFeret']:
                stats_ij[column].append(results.getColumn(column)[roi_index])

        for roi_index in range(rois.getCount()):
            new_img = self.ij.IJ.createImage("Untitled", "8-bit black", jimp.getWidth(), jimp.getHeight(), 1)
            if self.ui_show:
                self.ij.ui().show(new_img)

            rois.select(new_img, roi_index)
            self.ij.IJ.run(new_img, "Invert", "")
            img_single_hole = new_img.crop()

            self.ij.ui().show(img_single_hole)

            powers = int(math.log2(max(img_single_hole.getWidth(), img_single_hole.getHeight())))+2
            new_width = 2**powers
            new_height = new_width

            self.ij.IJ.run(img_single_hole, "Canvas Size...",
                      "width=%d height=%d position=Center zero" % (new_width, new_height))

            self.ij.IJ.run(img_single_hole, "Find Edges", "")

            self.ij.IJ.run(img_single_hole, "Fractal dimension", "startboxsize=%d smallestboxsize=2 "
                         "scalefactor=1.2 translations=0 autoparam=true showpoints=false" % int(new_width/2))

            macro_false = """
                selectWindow("DUP_Untitled");
                close();
                """
            macro_true = """
                selectWindow("DUP_Untitled");
                close();
                """
            if self.ui_show:
                self.ij.py.run_macro(macro_true)
            else:
                self.ij.py.run_macro(macro_false)

        sharedtable = sj.jimport('org.bonej.utilities.SharedTable')
        table = sharedtable.getTable()
        print(table[0])

        self.ij.IJ.run("Clear BoneJ results")

        # Display the results.
        # df_ij = DataFrame(stats_ij)
        # print(df_ij)

        return stats_ij, sum(stats_ij["Area"][0:]) / (jimp.getWidth() * jimp.getHeight()), table[0], thickness_2D, rois.getCount()

    def calc_Paras_to_Label(self, input_img):
        # 1.孔隙率
        bone_paras, porosity, frac, thickness, holes_num = self.get_Bone_Paras(input_img)

        feret_list = [math.sqrt(x * y) for x, y in zip(bone_paras["Feret"], bone_paras["MinFeret"])]
        feret_mean = np.mean(feret_list)

        frac_list = np.array(frac)
        frac_mean = np.mean(frac_list)


        return porosity, feret_mean, frac_mean, thickness, holes_num
