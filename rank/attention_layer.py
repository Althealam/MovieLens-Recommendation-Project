import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# 注意力层
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[0][-1], input_shape[1][-1]),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1][1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # query: candidate item, key: history items
        query, keys = inputs
        # 计算注意力得分
        query = tf.expand_dims(query, 1)  # [B, 1, D]
        scores = tf.matmul(keys, tf.matmul(query, self.W) + self.b)  # [B, T, 1]
        scores = tf.nn.softmax(scores, axis=1)
        # 加权求和
        output = tf.reduce_sum(scores * keys, axis=1)  # [B, D]
        return output