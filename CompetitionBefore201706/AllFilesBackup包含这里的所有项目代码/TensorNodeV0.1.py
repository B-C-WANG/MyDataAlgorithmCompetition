import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS
# 存储summary文件路径
flags.DEFINE_string('summaries_dir', '/tmp/save_graph_logs', 'Summaries directory')


#TODO：先确定了建立节点是先建立in link，后来是out link，但是道路是双向的，应该现将道路可视化处理！
class TensorNode:
    '''
    每个节点有一个输入、一个上游输入、一个输出和一个下游输出，节点总的就只有输入和输出，而
    节点内部的上游输入，下游输出等类似LSTM是内部结构，

    注意这里的输入输出都是placeholder，误差会在每个节点内部反向传播

    输入包含上游给的输入in_link_input 和 外部数据输入 input
    根据规则用in_link_input_to_input接收上游输入和原始数据输入，return 正式的输入input，如果没有上游
    直接就是input

    输出包含out_link_out 和 out，分别通过input_to_out_link input_to_out_put得到对下游节点的输出和正式输出

    正式的输出会和标签y求取loss，并反向训练，每个节点都有单独的loss训练函数，当然也可以加起来minimize

    如果每个node是上下游依次连接，那么很好说，就像RNN一样，但是这个模型是构建了树的，也就是不是
    链式传播，而是树状传播！



    2017年7月27日更新：

    每个节点有一个TI和TO，用于节点的输入输出，其中根节点没有TI，叶节点没有TO，除了TITO，就是NI和NO，指的是神经网络数据的IO

    这个类有一个总的placeholder，然后将矩阵切片之后，将placeholder得到的数据切分给各个节点，

    最终各个节点得到的数据再concat成输出


    '''
    # nodo记录
    nodes = []
    node_names = []

    # 信息常量记录
    node_shape = None
    node_number = None

    # 总IO记录
    TI = None
    TO = None
    split_TI = None


    # node变量记录，每添加一个节点，将其的index设为cls的index，并将index+1
    index = 0

    # 状态初始化记录，这表明在进行计算之前，需要按步骤进行初始化，完成一步初始化，就将相应的值置为1，某些步骤必须保证上一步骤为1，
    # 只有全部都为1之后，tf节点的运算操作才会执行
    init1 = 0
    init2 = 0
    init3 = 0


    '''
    操作步骤：
    1. 调用类的方法设置node_shape和node_number
    1.5 调用类的init1方法进行第一次初始化，之后才允许建立实例
    2. 为每个节点建立实例，每个实例在建立的时候会进行一次初始化
    3. 调用类的方法，连接节点，相当于第三次初始化
    4. 进行最后的初始化

    '''








    @classmethod
    def set_node_shape(cls,node_shape):
        if not isinstance(node_shape, int):
            raise ValueError("Only support int for shape.")
        cls.node_shape = node_shape

    @classmethod
    def set_node_number(cls,node_number):
        if not isinstance(node_number, int):
            raise ValueError("Only support int for nude number")
        cls.node_number = node_number


    # 建立总输出，并将输入切分，是第一次初始化的内容
    @classmethod
    def init1(cls):
        cls.TI = tf.placeholder(shape=(cls.node_number,cls.node_shape),dtype="float32",name="TI")
        cls.split_TI = tf.split(cls.TI,num_or_size_splits=cls.node_number)

    # 对每个实例连接节点，然后得到TO，是第三次初始化。里面会对每个实例调用其自己的初始化函数
    # 之后按照input同样的顺序回收output
    @classmethod
    def init3(cls):
        for i in cls.nodes:
            i.__init3()

        output = cls.nodes[0].NO

        for i in range(1,len(cls.nodes)):
            output = tf.concat(axis=0,values=[output,cls.nodes[i].NO])
        cls.TO = output

        cls.init3 = 1

    @classmethod
    def train(cls,x_train,save_weights=True,load_weights=True,epoch=200,predict=True,predict_length=20):
        '''
        这个train函数使用时需要根据实例修改，其中这里的node_shape相当于是时间节点时间输入的window_time
        这里的x_train作为输入，t+1时刻的x_train作为输出
        分别是在x_train是t,t+node_shape时刻每个节点的状态，y_train相应的就是t+1，t+1+node_shape时刻

        其中x_train输入，第一维为节点数目，第二维为节点时间序列，shape为(node_number,node_shape(time_window))

        :param x_train:
        :param y_train:
        :param node_shape:
        :param save_weights:
        :param load_weights:
        :return:

        '''

        # 这个node_shape决定了时间窗口的大小，一开始就需要确定
        node_shape = cls.__class__.node_shape

        # 标准答案
        y_ =  tf.placeholder(shape=(cls.node_number,cls.node_shape,),dtype="float32",name="TI")

        loss = tf.reduce_mean(tf.square(cls.TO-y_))

        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

        with tf.Session()  as sess:
            tf.initialize_all_variables()
            if load_weights:
                try:
                    saver = tf.train.Saver(tf.all_variables())
                    saver.restore(sess, "checkpoint.data")
                except Exception as e:
                    print("Error in loading",e)


        for e in range(epoch):
            for i in range(x_train.shape[1]-1-node_shape):
                batchX = x_train [:,    i   :  i*node_shape    ]
                batchY_ = x_train[:,    i+1 :  i*node_shape +1 ]

                loss = sess.run(loss,feed_dict={
                    cls.TI: batchX,
                    y_: batchY_

                })
                print("loss",loss)

                train_step.run(feed_dict={
                    cls.TI:batchX,
                    y_:batchY_

                })



        if save_weights:
            try:
                saver = tf.train.Saver(tf.all_variables())
                saver.save(sess, "checkpoint.data")
            except Exception as e:
                print("Error in saving",e)


        if predict:
            result = []
            # 接着最后的时间序列进行预测，这是第一次
            batchX = x_train[ : ,x_train.shape[1]-node_shape: ]
            for i in range(predict_length):
                predict = sess.run(cls.TO,feed_dict={
                    cls.TI:batchX
                })
                # 加上每一个node最后的一个值
                result.append(predict[ :, -1])

                # 后移一位，加上新预测的值
                batchX = batchX[ :, batchX.shape[1] - node_shape+1 : ]
                batchX = np.concatenate((batchX,result[-1]),axis=1)

            predict = np.array(predict)
            np.save("prediction.npy",predict)


    # 第二次初始化
    def __init__(self,name,father=None,child=None):
        '''
        node shape只能为整数，目前只支持一维
        name为唯一的变量
        in_node是上游节点，in_node_data是上游节点传过来的信息    (None, node_shape)
        out_node是下游节点，out_node_data是向下游节点传递的信息  (None, node_shape)
        input_data是真实数据集喂入的输入，shape为               (None, node_shape)
        output_data是用于喂入真实数据集进行误差计算的            (None, 1)
        '''
        # 初始化判断
        if self.__class__.node_shape is None:
            raise ValueError("Use set_node_shape to set an int shape.")

        if self.__class__.node_number is None:
            raise ValueError("Use set_node_number to set an int node number.")

        if self.__class__.init1 == 0:
            raise ValueError("Use cls.init1 to initialize before create nodes.")

        # 名字必须独一无二
        self.name = name
        if self.name in self.__class__.node_names:
            raise ValueError("name {} is exist".format(self.name))

        if (not isinstance(father, list)) or (not isinstance(child, list)):
            raise TypeError("Father and child must be a list.")

        self.father = father
        self.child = child



        # index设定
        self.index = self.__class__.index
        self.__class__.index += 1

        # 用于检验是否已经运算，避免节点形成圈的死循环
        self.done = 0

        # 检测第二次初始化是否完成
        if self.index == self.__class__.node_number:
            self.__class__.init2 = 1


    # 神经元的IO和节点的IO，全部都是单一值，多输入输出交给其他函数处理

    # 第三次初始化，这个函数只能够被类调用，接收I，计算后传入中心节点，然后输出至O
    # NI和TI原封不动输入，经过加权之后，得到center，center再进行神经网络算法，得到NO和TO
    # 这里的神经网络算法初步采用全连接神经网络，没有使用LSTM
    def __init3(self):
        if self.done == 1:
            return
        self.NI = self.return_NI()
        self.TI = self.return_TI()

        self.center = self.return_center()

        self.NO = self.neural_calc()
        self.TO = self.NO

        self.done = 1



    # NI是按照索引分配placeholder，之后节点怎么分配的，数据也要怎么分配
    def return_NI(self):
        return self.__class__.split_TI[self.index]

    # 直接搜集上一节点的TO，可能为空
    def return_TI(self):
        if self.father != None:
            _TI = []
            for i in self.father:
                _TI.append(i.TO)
            return _TI

    def return_center(self):
        now = self.NI
        if self.TI == None:
            return now
        else:
            for i in range(len(self.TI)):
                temp = self.set_d1_variable("mul_{}".format(i))
                # 将每一个TI与单变量相乘，得到的结果与now相加
                now = tf.add(tf.mul(self.TI[i],temp),now)

        return now


    # 最核心的算法在这里，输入已经为(node_shape, 1)的数据，要求输出为(node_shape, 1)
    def neural_calc(self):
        input = self.center
        input_shape = self.__class__.node_shape

        w1 = tf.matmul(self.set_mat_variable("w1",input_shape*2),input)
        b1 = tf.add(self.set_mat_bias("b1",input_shape*2),w1)
        r1 = tf.nn.dropout(b1, keep_prob=0.75)

        w2 = tf.matmul(self.set_mat_variable("w2", input_shape ), r1)
        b2 = tf.add(self.set_mat_bias("b2", input_shape ), w2)

        return  b2


    def set_d1_variable(self,variable_name):
        return tf.Variable(tf.random_normal(shape=[1])
                           , name="{}_{}".format(self.name, variable_name))

    def set_mat_variable(self,variable_name,hidden_size):
        return tf.Variable(tf.random_normal(shape=[self.__class__.node_shape,hidden_size],stddev=0.1)
                           ,name="{}_{}".format(self.name,variable_name))

    def set_mat_bias(self,variable_name,size):
        return tf.Variable(tf.random_normal(shape=size,stddev=0.1),
                           name="{}_{}".format(self.name,variable_name))


    # 检查节点有效性
    @staticmethod
    def check_if_valid_node(in_node):
        if not isinstance(in_node, list) or not isinstance(in_node[0], TensorNode):
            raise TypeError("Link node must be a list of TensorNode")




# tf的数据分割单元测试
def data_split_test():
    '''
    完成了一个用tf将input切片，索引，然后加上另一个矩阵。
    这表明tf的节点支持这些操作
    :return:
    '''
    input_ = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6]]
    input = tf.placeholder("float32")
    out = tf.split(input,num_or_size_splits=6,axis=0)[0]

    result = tf.add(out,[1,2,3])


    sess = tf.Session()
    out_ = sess.run(result,feed_dict={input:input_})
    print(out_)

#data_split_test()




