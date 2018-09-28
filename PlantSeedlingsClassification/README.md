- Kaggle种子识别竞赛：[https://www.kaggle.com/c/plant-seedlings-classification](https://www.kaggle.com/c/plant-seedlings-classification)
- coco,config,model,utils以及visualize文件是从kaggle优秀kernel上下载，表头有作者名
## 思路
- 1.直接使用Vgg，详见kaggle
- 2.数据集比较特殊，要识别的目标物体都是绿色，因此可以按rgb设置阈值提取绿色成分，去除周围的泥土等无关特征，具体见KerasGreenMaskVgg
- 3.更进一步，绿色可以直接作为图片的mask，使用Mask-RCNN进行训练，这样得以更加保留形貌信息，能这样做是因为种子是绿色直接可以用来转化为mask标签，数据集本身比较特殊。