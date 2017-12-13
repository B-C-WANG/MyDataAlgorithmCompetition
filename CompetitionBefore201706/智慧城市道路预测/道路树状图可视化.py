import tqdm
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import profile
import cv2







#TODO：     3、建立节点内部的画图方法   4、根据关系找到根节点，由根节点开始索引画图




class DrawRoad():
    '''
    :param:
    '''

    roads = []




    # 先把树的关系建立起来再说
    def __init__(self,road_id,road_length=None,road_width=None,link_in=None,link_out=None,level=None):
        self.road_id = road_id
        self.road_length = road_length
        self.road_width = road_width
        self.link_in = link_in
        self.link_out = link_out
        self.road_level = level # 道路等级



        self.checked = False#在用树作索引的时候，需要检测是否已经checked，否则可能陷入循环，比如一个4结点的矩形
        self.__class__.roads.append(self)


    def update(self,link_in=None,link_out=None):


        self.link_in = link_in
        self.link_out = link_out

    def show_info(self):

        print(">>>>>>>>>>>>>>>>>>>>>")
        print("road ID",self.road_id)
        print("road length",self.road_length)
        print("road width", self.road_width)
        print("road level",self.road_level)

        print("link in", self.link_in)
        print("link out",self.link_out)

        return ""

    @classmethod
    def get_root(cls):
        root = []
        for i in cls.roads:
            if i.link_in == None:
                root.append(i)

                print('find root:>>>>>>>>>>>>>>>',i.show_info())
        return root


    @classmethod
    def find_Road_by_id(cls,id):
        # 需要分别处理list中的id，返回list
        if len(id) == 0 :
            print("No such road")
            return None
        road = []
        print("finding:   ",id)
        print("found")
        for j in id:
            for i in cls.roads:
                if i.road_id == j:
                    print(i.road_id)
                    road.append(i)
        print("----")
        if road == []:
            print("No such road_id!")
        else:
            return road





# 从此以上，已经建立了一个完整的road节点，之后是进行plot

    # 首先对节点进行order设置，这个order(n, m)将根据深度和同一深度上的排序决定每个节点的位置，之后连接是link的事儿
    # 第一步，先统计每个节点到root的距离，根据这个距离设置n，之后对于每个n，排列设置m，n即为 self.plot_y, m = self.plot_x

    # 遇到的问题：会陷入死循环：存在一个矩形连通，





ROAD_DATA_PATH = "gy_contest_link_info.txt"

def read_road_data():
            data = pd.read_csv(ROAD_DATA_PATH, delimiter=";")
            print(data,data.shape)
            return data



ROAD_LINK_DATA_PATH = "gy_contest_link_top.txt"

def read_road_link_data():
            data = pd.read_csv(ROAD_LINK_DATA_PATH, delimiter=";")
            print(data,data.shape)
            return data



def create_Road_objects():
    # 准备文件中的数据
    road_data = read_road_data()
    road_data_id = list(road_data["link_ID"])
    road_data_length = list(road_data["length"])
    road_data_width = list(road_data["width"])
    road_data_level = list(road_data["link_class"])


    link_data = read_road_link_data()
    t_link_data_id = list(road_data["link_ID"])
    t_link_data_in_links = list(link_data["in_links"])
    t_link_data_out_links = list(link_data["out_links"])

    link_data_id = []
    link_data_in_links = []
    link_data_out_links = []
    for i in range(len(t_link_data_id)):
        link_data_id.append(str(t_link_data_id[i]))
        link_data_in_links.append(str(t_link_data_in_links[i]))
        link_data_out_links.append(str(t_link_data_out_links[i]))



    #print(road_data_id,"\n",link_data_id,"\n",road_data_width,"\n",road_data_level,"\n",link_data_in_links)

    _ = []

    for i in range(len(road_data_id)):
        assert road_data_id[i] == link_data_id[i]

        _.append(DrawRoad(road_id=road_data_id[i],road_length = road_data_length[i],
                        road_width = road_data_width[i],
                          level = road_data_level[i]
                ))# 根据id初始化对象，增加初始属性


    for i in range(len(road_data_id)):

        if "#" in  link_data_in_links[i]:
            link_data_in = link_data_in_links[i].split("#")
        else:
            link_data_in = [link_data_in_links[i]]

        if "#" in link_data_out_links[i]:
            link_data_out = link_data_out_links[i].split("#")
        else:
            link_data_out = [link_data_out_links[i]]

        #print(link_data_in,"\n",link_data_out)

        # 更新连接相关属性
        DrawRoad.roads[i].update(
            link_in  =  DrawRoad.find_Road_by_id(link_data_in),
            link_out  = DrawRoad.find_Road_by_id(link_data_out)
        )






def main():
    create_Road_objects()
    roads = DrawRoad.roads
    for i in roads:


        i.show_info()



if __name__=="__main__":
    main()


#对于索引作图，采用层次式，每次用list按顺序存储上一级下面的所有结点，画出点，如果为None则无视，画出点后排列点，再连线