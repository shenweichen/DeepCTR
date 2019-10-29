# # 计算一个矩阵相乘
# import tensorflow as tf
#
# m1 = tf.constant([[1, 2]], dtype=tf.int32)  # 定义常量张量大小为 1x2, 类型为 tf.int32
# m2 = tf.constant([[1, 2]])  # 定义常量张量大小为 2x1， 类型 tf.int32
# m3 = tf.constant([[[1,2], [3,4]],[[3,2],[1,7]]])
#
#
# aaaaaaa = tf.expand_dims(m2, axis=5)
#
# # user_behavior_length = tf.reduce_sum(m3, axis=-1, keep_dims=True)
# # mask = tf.expand_dims(mask, axis=2)
#
# with tf.Session() as sess:
#     result = sess.run(aaaaaaa)  # 执行会话，得到运算的结果
#     print(result)
#
#     # result1 = sess.run(shapes)  # 执行会话，得到运算的结果
#     # print(result1)
#
# # [[[1]
# #   [2]]]

from collections import OrderedDict, namedtuple

class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):

        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)
    def __hash__(self):  # 哈希算法把形参转为一个数值
        return self.name.__hash__()  # 把形参 name 字符串,转换为数值

    def __eq__(self, other):  # 一般都跟随__hash__()出现，把转换过来的值，进行比较
        if self.name == other.name:
            return True
        return False

    def __repr__(self):
        return self.name + ";" + self.name


u1 = DenseFeat('user',3)
u2 = DenseFeat('user',3)
u3 = DenseFeat('usser',3)
u4 = DenseFeat('user',5)
u5 = DenseFeat('user',3)

u = set()  # 创建集合
print(u)

u.add(u1)  # 把u1对象加入集合中
print(u)  # 输出集合 u

u.add(u2)  # 把u2对象加入集合中
print(u)

u.add(u3)  # 把u3对象加入集合中
print(u)

a = {11:324,34:667}
print(list(a.values())+list(a.keys()),sep = '\n')