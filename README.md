# MyDataAlgorithmCompetition
- 我参加数据竞赛（不包括数模竞赛等）的记录，里面的代码可以用在类似问题处理中
- 代码全部都是当时完成之后原封不动的（引用他人的代码文件头有相应的作者名称），部分代码经过整理放到了MachineLearningTools中作为工具
- 三个竞赛都是2017年7月暑假读研之前参加，之后读研集中于课题内相关的开发

# 天池大数据竞赛：数聚华夏 创享未来”中国数据创新行-智慧交通预测挑战赛
- 地址：[https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.10.3aee2784Y1VVQd&raceId=231598](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100150.711.10.3aee2784Y1VVQd&raceId=231598)
- 排名：提交时是8%，等到初赛结束为12%，10%进复赛，有点儿可惜（证书在最后）
- 使用ConvLSTM算法：将源数据按照时间归类，每一个时间有132个道路信息，合并起来，这样得到了每个时间下的132维度数据，将其视作“帧”处理，上面那只是一个“视频”，按照每个视频900帧，以及每个视频相当于上个视频向后滑动100帧来处理，数据量扩大8.9倍，得到了670多个900帧的视频，接着用前一个视频作为输入，后一个视频作为输出，建立ConvLSTM模型，预测最后900帧的视频


# 摩拜杯数据竞赛-MobikeCup
- 参加日期：2017年7月
- 摩拜杯数据竞赛：[https://biendata.com/competition/mobike/](https://biendata.com/competition/mobike/)
- 获奖ppt：[https://biendata.com/competition/mobike/winners/](https://biendata.com/competition/mobike/winners/)
- public分数为0.18左右，第一名分数为0.34，只提交了第一周的测评结果，第一周排名在前15%左右。
- ![](https://i.imgur.com/99U0nAD.png)
### 分析
- 数据集X是用户ID，车辆ID，用户类型，起始日期时间，起始区块，预测目的区块位置，区块位置是label，但是这个label种类相当多，因为是经纬度转化成的label，所以有近乎无穷个label。
- 提交时给出最可能的3个骑行目的区块位置
- 虽然label相当多，但是这个label是Geohash，与位置对应，所以可以将位置相近的label放在一起，减少label的数量，便于预测。同时也可以将起始时间粗粒化，或者去掉日期，只保留星期和每天的时段。
### 算法
- 当时先使用NN，进行经纬度数据回归，然后再转回经纬度label，但发现经纬度相差一点点儿，转回Geohash的label完全不同，最后提交上去只有小于1%的准确度
- 后来尝试采用朴素贝叶斯，即根据训练集上用户的车辆ID，起始时间和相应的地点，获取用户最大概率取得的地方，采用现成的模块，发现训练时间相当长
- 于是准备自己实现类似朴素贝叶斯的方法，对训练集进行字典记录，以用户ID作为第一个key，起始地点作为第二个key，时间作为第三个key，最终目的区块作为value。训练的时候就是不断地更新这些value，如果key完全一样，就把value变成list存储。最后预测的时候，如果能够从字典中找到一样的key，就把value中最常出现的label作为预测值，如果找不到，再通过省略用户ID，采用相近地点的方法来找到value。这样的方法简单粗暴，最后效果也还行，但提升空间不大。


# 工业大数据竞赛-IndustrialBigDataCompetition
- 参加日期： 2017年7月
- 风机叶片结冰预测大赛：[http://www.industrial-bigdata.com/competition/competitionAction!showDetail.action?competition.competitionId=1](http://www.industrial-bigdata.com/competition/competitionAction!showDetail.action?competition.competitionId=1)
- 在提交时名次在13%左右
- 二分类问题，给出一大堆运行参数如叶片速度、角度，设备温度等等
- 标签有特殊性，因为结冰是持续过程，所以有一段数据长期处于label=1，大部分数据长期处于label=0,因此数据shuffle和标签平衡非常重要
- 当时没有意思到这些问题，也没有特征工程的概念，直接上来就用DenseNN进行预测，预测出来的01结果比较离散，然后尝试把它们连成串。


- 天池数据竞赛证书：
![](https://i.imgur.com/LIvWwPE.png)