from keras import backend as K

class self_attention(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(self_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # 创建可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3, input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        super(self_attention, self).build(input_shape)  # 最后调用它

    def call(self, x):

        WQ = K.dot(x, self.kernel[0])  #由查询向量组成的矩阵
        WK = K.dot(x, self.kernel[1])  #由键向量组成的矩阵
        WV = K.dot(x, self.kernel[2])  #由值向量组成的矩阵

        print("WQ.shape", WQ.shape)

        print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))

        QK = QK / (64 ** 0.5)

        QK = K.softmax(QK)

        print("QK.shape", QK.shape)

        V = K.batch_dot(QK, WV)

        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)