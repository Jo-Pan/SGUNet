import os, shutil
import tensorflow as tf
from trainer import Trainer
from res18 import Resnet18

slim = tf.contrib.slim
import keras as K


class Model(Trainer, Resnet18):

    def __init__(self, sess, param_dict):
        # ---- Global Variables ----
        self.sess = sess
        self.param_dict = param_dict

        log_dir = os.path.join(self.param_dict['log_dir'], self.param_dict['model_name'])
        if self.param_dict['mode'] == 'train':
            if os.path.isdir(log_dir) and self.param_dict['mode'] == 'train':
                shutil.rmtree(log_dir)
                print("removed old log: ", log_dir)
            os.makedirs(log_dir)

            out_dir = os.path.join(self.param_dict['output_dir'], self.param_dict['model_name'])
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

        self.x = tf.placeholder(tf.float32, [None, ] + param_dict['im_size'] + [1], name='x')

        # ---- SEGMENTATION  ----
        if self.param_dict['task'] == 'segmentation':
            self.y = tf.placeholder(tf.float32, [None, ] + param_dict['im_size'], name='y')
            if self.param_dict['model_type'] == 'unet++':
                multi_outputs = self.u_net(self.x)
                self.loss = 0
                for i, logits in enumerate(multi_outputs):
                    self.loss += (1.0/len(multi_outputs)) * self.get_loss(target=self.y, logits=logits)

                logits = tf.reduce_mean(multi_outputs, axis=0)
                self.prob = tf.nn.softmax(logits)  # (?, 128, 128, 2) float32
                self.prob = self.prob[..., 1]  # (?, 128, 128)
                self.pred = K.backend.round(self.prob)  # (?, 128, 128) 0 or 1
                self.dice = self.dice_coef(target=self.y, prediction=self.pred)

            elif 'multi-output' not in self.param_dict['loss_type']:
                logits = self.u_net(self.x)  # (?, 128, 128, 2) float32
                if self.param_dict['num_class'] == 1:
                    self.prob = tf.math.sigmoid(logits)  # (?, 128, 128, 2) float32
                else:
                    self.prob = tf.nn.softmax(logits)  # (?, 128, 128, 2) float32
                self.prob = self.prob[..., -1]
                self.pred = K.backend.round(self.prob)  # (?, 128, 128) 0 or 1
                print('logits shape:', logits.get_shape)
                print('prob shape:', self.prob.get_shape)

                # ---- Loss ----
                self.dice = self.dice_coef(target=self.y, prediction=self.pred)
                self.loss = self.get_loss(target=self.y, logits=logits, prob=self.prob)

            else:
                multi_outputs = self.u_net(self.x)
                self.loss = 0
                p = 0
                for i, logits in enumerate(multi_outputs):
                    if i < len(multi_outputs) - 1:
                        self.loss += 0.1 * self.get_loss(target=self.y, logits=logits)
                        p += 0.1
                    else:
                        self.prob = tf.nn.softmax(logits)  # (?, 128, 128, 2) float32
                        self.prob = self.prob[..., 1]  # (?, 128, 128)
                        self.pred = K.backend.round(self.prob)  # (?, 128, 128) 0 or 1
                        self.dice = self.dice_coef(target=self.y, prediction=self.pred)
                        self.loss += (1 - p) * self.get_loss(target=self.y, logits=logits, prob=self.prob)

        # ---- CLASSIFICATION  ----
        elif self.param_dict['task'] == 'classification' and self.param_dict['loss_type'] == 'ce':
            self.y = tf.placeholder(tf.int64, [None, self.param_dict['numClass']], name='y')
            logits = self.cnn(self.x)  # (?, 3)
            self.prob = tf.nn.softmax(logits)  # (?, 3)
            self.pred = K.backend.round(self.prob)  # (?, 3) 0 or 3
            self.loss = tf.compat.v1.losses.softmax_cross_entropy(tf.cast(self.y, tf.int32), logits)

        # ---- maintenance ----
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=param_dict['lr']).minimize(self.loss)

        if param_dict['print_network']:
            slim.model_analyzer.analyze_vars(tf.trainable_variables(), print_info=True)
        
        self.saver = tf.train.Saver(list(set(tf.trainable_variables() + tf.get_collection_ref('bn_collections'))))

        self.train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'))
        self.test_writer = tf.summary.FileWriter(os.path.join(log_dir, 'test'))

        if self.param_dict['task'] == 'classification':
            self.feed_list = [self.optimizer, self.loss, self.prob]
        else:
            self.feed_list = [self.optimizer, self.loss, self.prob, self.pred, self.dice]

    def get_loss(self, target, logits, prob=None):
        if prob is None:
            prob = tf.nn.softmax(logits)  # (?, 128, 128, 2) float32
            prob = prob[..., -1]

        loss = 0
        n = 0

        current_shape = logits.get_shape().as_list()[1:3]
        sh = tf.TensorShape(current_shape)
        if target.get_shape().as_list()[1:3] != current_shape:
            target = tf.image.resize_images(tf.expand_dims(target, axis=-1), sh)
            target = tf.squeeze(target, axis=-1)
            target = K.backend.round(target)
            print('resized target for loss: ', target.get_shape())

        if 'dice' in self.param_dict['loss_type']:
            loss += self.dice_coef_loss(target=target, prediction=prob)
            n += 1.0
        if 'ce' in self.param_dict['loss_type'] and self.param_dict['loss_type'] != 'dice':
            # https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/sparse_softmax_cross_entropy
            # https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
            if self.param_dict['num_class'] == 2:
                loss += tf.compat.v1.losses.sparse_softmax_cross_entropy(tf.cast(target, tf.int32), logits)
            elif self.param_dict['num_class'] == 1:
                loss += tf.losses.log_loss(tf.cast(target, tf.int32), prob)

            n += 1.0
        if 'edge' in self.param_dict['loss_type']:
            # weight more loss on true boundary
            temp_y = tf.expand_dims(target, axis=-1)
            sobel = tf.abs(tf.image.sobel_edges(temp_y))  # [batch_size, h, w, 1, 2]
            sobel = tf.divide(sobel, tf.reduce_max(sobel))
            map = tf.reduce_sum(sobel, axis=-1)
            map = tf.squeeze(map, axis=-1)  # [batch_size, h, w]
            loss_map = tf.multiply(map, target - prob)
            loss += tf.reduce_sum(loss_map) / tf.cast(tf.math.count_nonzero(map), tf.float32)
            n += 1.0

        loss = loss / n
        if self.param_dict['loss_type'] == 'sorenson':
            loss = self.sorenson_dice_coef_loss(target=self.y, prediction=self.prob)

        return loss

    def dice_coef(self, target, prediction, axis=(1, 2), smooth=0.01):
        """
        Dice: prediction (0 or 1)
        Soft Dice: prediction (prob 0 to 1)
        https://github.com/IntelAI/unet/blob/master/2D/model.py

        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask
        """

        intersection = tf.reduce_sum(target * prediction, axis=axis)
        union = tf.reduce_sum(target + prediction, axis=axis)
        numerator = tf.constant(2.) * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator
        return tf.reduce_mean(coef)

    def dice_coef_loss(self, target, prediction, axis=(1, 2)):
        with tf.variable_scope("dice_loss", reuse=None):
            # https://www.jeremyjordan.me/semantic-segmentation/#loss
            numerator = 2.0 * tf.reduce_sum(prediction * target, axis=axis)
            denominator = tf.reduce_sum(tf.square(prediction) + tf.square(target), axis=axis)
            dice_loss = 1 - tf.reduce_mean(numerator / (denominator + 1e-6))
            return dice_loss

    def sorenson_dice_coef_loss(self, target, prediction, axis=(1, 2), smooth=1.0):
        """
        Sorenson (Soft) Dice loss
        Using -log(Dice) as the loss since it is better behaved.
        Also, the log allows avoidance of the division which
        can help prevent underflow when the numbers are very small.
        """
        intersection = tf.reduce_sum(prediction * target, axis=axis)
        p = tf.reduce_sum(prediction, axis=axis)
        t = tf.reduce_sum(target, axis=axis)
        numerator = tf.reduce_mean(intersection + smooth)
        denominator = tf.reduce_mean(t + p + smooth)
        dice_loss = -tf.log(2. * numerator) + tf.log(denominator)
        return dice_loss

    def basic_encoder_module(self, net, layer, scope="down"):
        f = self.param_dict['layer_fs'][layer]
        with tf.variable_scope(scope, reuse=None):
            # --- 1. Conv * 2 --- #
            net = slim.repeat(net, 2, slim.conv2d, f, [3, 3], scope='conv')

            connection = tf.identity(net)

            # --- 2. Pooling --- #
            if self.param_dict['avgpool']:
                net = slim.avg_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool')  # 1/2
            else:
                net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME', scope='pool')  # 1/2

            # --- 3. BN --- #
            # I like to think of batch normalization as being more important for the input of the next layer
            # than for the output of the current layer--i.e. ideally the input to any given layer has zero
            # mean and unit variance across a batch.
            net = slim.batch_norm(net, decay=0.9, scope="bn")
            print('down', layer, net.get_shape)
        return net, connection

    def u_net(self, image):
        """
        Segmentation Network
        https://github.com/lighttxu/slim-Unet/blob/master/slim-Unet.ipynb
        :param image: [batch, h, w, c]
        :return: logits
        """

        with tf.variable_scope("u_net", reuse=None):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                biases_initializer=tf.constant_initializer(0.0),
                                padding='SAME'):
                net = image
                encodings = []
                multi_outputs = []
                cc_dict = {}
                temp = []
                num_layers = self.param_dict['num_layers']
                num_class = self.param_dict['num_class']
                layer_fs = self.param_dict['layer_fs']

                # ------------ ENCODING ------------#
                if self.param_dict['backbone'] == "":
                    for layer in range(num_layers):
                        net, connection = self.basic_encoder_module(net, layer, scope="down{}".format(layer))
                        encodings.append(connection)
                        cc_dict['{}0'.format(layer)] = connection

                    net = slim.repeat(net, 2, slim.conv2d, layer_fs[-1],
                                      [3, 3], scope='conv_bottom')
                    print('bottom', net.get_shape)

                elif 'res18' in self.param_dict['backbone']:
                    net, encodings = self.res18(net)
                    if 'l4' not in self.param_dict['backbone']:
                        num_layers = 5

                else:
                    exit("Undefined backbone")

                if self.param_dict['model_type'] == "sgunet":
                    layer_pred = slim.conv2d(net, 1, [1, 1], activation_fn=None)
                    
                    multi_outputs.append(layer_pred)

                # ----------- UNet++ ----------------#
                if self.param_dict['model_type'] == "unet++":
                    for uplayer in range(num_layers-1): #range 3
                        layer = num_layers - 2 - uplayer #4-2-0 = 2
                        for cc_id in range(1, num_layers-layer):
                            with tf.variable_scope("pl{}{}".format(layer, cc_id), reuse=None):
                                out = cc_dict['{}{}'.format(layer+1, cc_id-1)]
                                layer_f = encodings[layer].shape[-1]
                                out = slim.conv2d_transpose(out, layer_f, [2, 2], stride=2, scope='conv_t')

                                cc_str = '{}{} = up{}{} '.format(layer, cc_id, layer+1, cc_id-1)
                                for prev_cc_id in range(cc_id):
                                    out = tf.concat([cc_dict['{}{}'.format(layer, prev_cc_id)], out], 3)
                                    cc_str += 'cc {}{} '.format(layer, prev_cc_id)
                                out = slim.stack(out, slim.conv2d, [(out.get_shape().as_list()[-1], [3, 3]), (layer_f, [3, 3])], scope='conv')
                                cc_dict['{}{}'.format(layer, cc_id)] = out
                                print(cc_str, out.get_shape())
                        encodings[layer] = out

                # ------------ DECODING ------------#
                # convert bottom to layer pred
                for uplayer in range(num_layers):
                    layer = num_layers - 1 - uplayer

                    try:
                        layer_f = encodings[layer].shape[-1]
                    except:
                        layer_f = 32 * 2 ** layer

                    with tf.variable_scope("up{}".format(layer), reuse=None):
                        low_res = tf.identity(net)

                        # --- 1. Deconv --- #
                        net = slim.conv2d_transpose(net, layer_f, [2, 2], stride=2, scope='conv_t')

                        # --- 2. Connection b/w encode & decode--- #
                        if encodings[layer] != []:
                            if self.param_dict['model_type'] == "unet++":
                                cc_str = 'up{} = lower decoder '.format(layer)
                                for prev_cc_id in range(uplayer+1):
                                    net = tf.concat([cc_dict['{}{}'.format(layer, prev_cc_id)], net], 3)
                                    cc_str += 'cc {}{}'.format(layer, prev_cc_id)
                                print(cc_str)
                            else:
                                net = tf.concat([net, encodings[layer]], 3)


                        print('up', layer,' after cc:   ', net.get_shape)


                        # --- 3. Conv * 2 + BN --- #
                        net = slim.stack(net, slim.conv2d, [(net.get_shape().as_list()[-1], [3, 3]), (layer_f, [3, 3])], scope='conv')
                        net = slim.batch_norm(net, decay=0.9, scope='bn')
                        print('up', layer,' after conv: ', net.get_shape)

                        current_shape = net.get_shape().as_list()[1:3]
                        sh = tf.TensorShape(current_shape)

                        # --- 4*. Residual Decoder --- #
                        if self.param_dict['model_type'] == "sgunet":
                            ##11b
                            residual = slim.conv2d(net, 1, [1, 1], activation_fn=None)

                            # upsize the old layer pred
                            layer_pred = tf.image.resize_images(layer_pred, sh)
        
                            # combine new layer pred with residuals
                            if layer > 0:
                                layer_pred = tf.concat([residual, layer_pred], axis=3)
                                layer_pred = slim.conv2d(layer_pred, 1, [1, 1], activation_fn=None)
                            else:
                                layer_pred = tf.concat([residual, layer_pred], axis=3)
                                layer_pred = slim.conv2d(layer_pred, num_class, [1, 1], activation_fn=None)

                            multi_outputs.append(layer_pred)
                            print('up', layer, ' layer_pred: ', layer_pred.get_shape)

                # ------------ Final Logit ------------#
                if self.param_dict['model_type'] == "sgunet":
                    self.multi_outputs = multi_outputs
                    if 'multi-output' in self.param_dict['loss_type']:
                        return multi_outputs
                    else:
                            return layer_pred

                elif self.param_dict['model_type'] == 'unet++':
                    multi_outputs = []
                    for j in range(1, num_layers):
                        multi_outputs.append(slim.conv2d(cc_dict['0{}'.format(j)],
                                                         num_class, [1, 1], activation_fn=None, normalizer_fn=None, scope='logit{}'.format(j)))
                    
                    multi_outputs.append(slim.conv2d(net, num_class, [1, 1], activation_fn=None, normalizer_fn=None, scope='logit'))
                    self.multi_outputs=multi_outputs
                    return multi_outputs
                
                net = slim.conv2d(net, num_class, [1, 1], activation_fn=None, normalizer_fn=None, scope='logit')
                self.multi_outputs = temp
            return net

    def cnn(self, net):
        # similar to unet
        with tf.variable_scope("u_net", reuse=None):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                biases_initializer=tf.constant_initializer(0.0),
                                padding='SAME'):
                connection = []
                num_layers = self.param_dict['num_layers'] + 1
                f = self.param_dict['initial_feature']

                with tf.variable_scope("down", reuse=None):
                    for layer in range(num_layers):
                        layer_f = f * 2 ** layer
                        if layer_f > 256:
                            layer_f = 256
                        net = slim.repeat(net, 2, slim.conv2d, layer_f, [3, 3], scope='conv{}'.format(layer))
                        connection.append(net)
                        net = slim.max_pool2d(net, [2, 2], stride=2, padding='SAME',
                                              scope='pool{}'.format(layer))  # 1/2 padding='SAME'
                        net = slim.batch_norm(net, decay=0.9, scope="bn{}".format(layer))
                        print('down', layer, net.get_shape)

                k = net.get_shape().as_list()[1:3]
                net = slim.conv2d(net, k[0] * layer_f, k, padding="VALID", scope='fc6')
                print('fc', net.get_shape)
                net = slim.conv2d(net, k[0] * layer_f, [1, 1], scope='fc7')
                print('fc', net.get_shape)
                net = slim.conv2d(net, self.param_dict['numClass'], [1, 1], activation_fn=None, normalizer_fn=None,
                                  scope='fc8')
                print('fc', net.get_shape)
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                print('fc', net.get_shape)
                return net

    def vgg16(self, inputs, n_class):
        # un-used
        f = self.param_dict['initial_feature']

        with tf.variable_scope("vgg_16"):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                                biases_initializer=tf.constant_initializer(0.0)):
                with slim.arg_scope([slim.conv2d], padding='SAME'):
                    net = slim.repeat(inputs, 1, slim.conv2d, 12, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 1, slim.conv2d, 24, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 1, slim.conv2d, 32, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 1, slim.conv2d, 48, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 1, slim.conv2d, 48, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    print(net.shape)

                net = slim.flatten(net)
                net = slim.fully_connected(net, n_class, activation_fn=None, normalizer_fn=None, scope='fc8')
        return net

