import tensorflow as tf

slim = tf.contrib.slim
# https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_utils.py
# https://github.com/dalgu90/resnet-18-tensorflow/blob/master/utils.py

class Resnet18(object):
    # def conv2d_same(self, inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    #     if stride == 1:
    #         return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
    #                            padding='SAME', scope=scope)
    #
    #     else:
    #         kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    #         pad_total = kernel_size_effective - 1
    #         pad_beg = pad_total // 2
    #         pad_end = pad_total - pad_beg
    #         inputs = tf.pad(tensor=inputs,
    #                         paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    #         return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
    #                            rate=rate, padding='VALID', scope=scope)
    #
    # def res18(self, net):
    #     encodings = []
    #     with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
    #         net = self.conv2d_same(net, 64, 7, stride=2, scope='conv1')
    #     net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
    #
    #     return net, encodings

    def res18(self, net):
        print('Building resnet encoder')
        encodings = [[],]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]

        if 'k5' in self.param_dict['backbone']:
            kernels[0] = 5
        if 'k3' in self.param_dict['backbone']:
            kernels[0] = 3

        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
            # conv1
            x = slim.conv2d(net, filters[0], [kernels[0], kernels[0]], stride=2, scope='conv1')
            print('\tBuilt residual unit: {}, {}'.format('conv1_1', x.shape))
            encodings.append(x)
            x = slim.max_pool2d(x, [3, 3], stride=2, scope='pool1_1', padding='SAME')
            print('\t                     {}, {}'.format('pool1_1', x.shape))


            # conv2_x
            x = self._residual_block(x, name='conv2_1')
            x = self._residual_block(x, name='conv2_2')
            encodings.append(x)

            # conv3_x
            x = self._residual_block_first(x, filters[2], stride=2, name='conv3_1')
            x = self._residual_block(x, name='conv3_2')
            encodings.append(x)

            # conv4_x
            x = self._residual_block_first(x, filters[3], stride=2, name='conv4_1')
            x = self._residual_block(x, name='conv4_2')
            encodings.append(x)

            # conv5_x
            if 'l4' not in self.param_dict['backbone']:
                x = self._residual_block_first(x, filters[4], stride=2, name='conv5_1')
                x = self._residual_block(x, name='conv5_2')

        return x, encodings

    def _residual_block_first(self, x, out_channel, stride, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            # Shortcut connection
            if in_channel == out_channel:
                if stride == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
            else:
                shortcut = slim.conv2d(x, out_channel, [1, 1], stride=stride)

            # Residual
            x = self._residual_op(x, shortcut, out_channel, stride)
            print('\tBuilt residual unit: {}, {}'.format(scope.name, x.shape))
        return x

    def _residual_op(self,  x, shortcut, num_channel=None, stride=None):
        if num_channel is None:
            num_channel = x.get_shape().as_list()[-1]

        # conv 1
        if stride is None:
            x = slim.conv2d(x, num_channel, [3, 3])
        else:
            x = slim.conv2d(x, num_channel, [3, 3], stride=stride)
        x = slim.batch_norm(x)
        x = tf.nn.relu(x)

        # conv 2
        x = slim.conv2d(x, num_channel, [3, 3])
        x = slim.batch_norm(x)

        # shortcut
        x = x + shortcut
        x = tf.nn.relu(x)
        return x

    def _residual_block(self, x, name):
        with tf.variable_scope(name) as scope:
            x = self._residual_op(x, shortcut=x)
            print('\tBuilt residual unit: {}, {}'.format(scope.name, x.shape))
        return x

