import pickle
import random
import re

import tensorflow as tf
import pandas as pd
import scipy
import os, shutil

import cv2
import numpy as np
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class Trainer(object):
    def train(self):
        self.sess.run(tf.global_variables_initializer())

        if not os.path.exists(self.param_dict['pred_seg_dir']):
            os.makedirs(self.param_dict['pred_seg_dir'])
        print(self.param_dict['model_name'])

        for epoch in range(self.param_dict['epoch']):
            # shuffle samples
            self.param_dict['batch_train_id'] = self.get_train_id()

            # run train
            x, y, _ = self.get_data('batch_train_id')

            if self.param_dict['task'] == 'classification':
                _, loss, prob = self.sess.run(self.feed_list, feed_dict={self.x: x, self.y: y})
                dice = None
                pred = None
            else:  # segmentation
                _, loss, prob, pred, dice = self.sess.run(self.feed_list, feed_dict={self.x: x, self.y: y})

            # eval
            print(self.eval(prob, y, epoch, loss, dice=dice))

            if np.mod(epoch, self.param_dict['valid_per_epoch']) == 0 or epoch == self.param_dict['epoch'] - 1:
                if self.param_dict['final']:
                    self.run_test('valid', self.feed_list, epoch, test_aug=self.param_dict['test_augmentation'])
                else:
                    for mode in ['valid', 'test']:
                        if len(self.param_dict['{}_ids'.format(mode)]) > 0:
                            self.run_test(mode, self.feed_list, epoch, test_aug=self.param_dict['test_augmentation'])
                print()

            # save checkpoint
            if epoch % 1000 == 0 or epoch == self.param_dict['epoch'] - 1:
                self.save(epoch)

    def test(self, mode):
        """
        Test wrapper
        :param mode: valid or test
        """
        # load
        if not self.load()[0]:
            raise Exception("No model is found, please train first")
        self.run_test(mode, self.feed_list, epoch=0, test_aug=self.param_dict['test_augmentation'])

    def run_test(self, mode, feed_list, epoch, test_aug=False):
        """
        Run test
        :param mode: valid or test
        :param feed_list: list of output from tf.run
        :param epoch: current epoch number
        :param test_aug: whether to use testing augmentation or not
        :return:
        """
        sum_prob_by_case = []
        prob_dict = {}
        all_loss = []
        small_dice = []
        true_dice = []
        gt = []
        num_cases = len(self.param_dict['{}_ids'.format(mode)])

        # load gt masks
        if self.param_dict['task'] == 'segmentation':
            for im_id in self.param_dict['{}_ids'.format(mode)]:
                mask = scipy.ndimage.imread(self.param_dict['data_dir'] + 'label/' + im_id, mode='L')
                mask = mask > 100
                gt.append(mask.astype(np.float32))
        
        out_dir = os.path.join(self.param_dict['output_dir'] , self.param_dict['model_name'])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        if mode == 'test': print('test_aug_num: ', self.param_dict['test_aug_num'])
            
        for test_aug_id in range(-1, self.param_dict['test_aug_num']):
            # run test
            x, y, aug_dicts = self.get_data('{}_ids'.format(mode), test_aug_id)
            if test_aug_id == -1:
                out = self.sess.run(feed_list[1:] +[self.multi_outputs], feed_dict={self.x: x, self.y: y})
                path = os.path.join(out_dir, mode + 'multi_outputs.pkl')
                with open(path, 'wb') as f:
                    pickle.dump(out[-1], f)
                out = out[:-1]
            else:
                out = self.sess.run(feed_list[1:], feed_dict={self.x: x, self.y: y})

            # reverse augmented prediction to original size
            for case, im_id in enumerate(self.param_dict['{}_ids'.format(mode)]):
                case_prob = out[1][case]
                if self.param_dict['task'] == 'segmentation':
                    case_prob = self.reverse_test_augment_im(mask=case_prob,
                                                             aug_dict=aug_dicts[case])
                if test_aug_id == -1:
                    #sum_prob_by_case.append(case_prob)
                    prob_dict[im_id] = [case_prob]
                else:
                    #sum_prob_by_case[case] += case_prob
                    prob_dict[im_id].append(case_prob)


            # store evaluation result
            all_loss.append(out[0])
            if self.param_dict['task'] == 'segmentation':
                small_dice.append(out[-1])

            if not test_aug:
                break

        if self.param_dict['test_aug_prob_out']:
            path = os.path.join(out_dir, mode+'probs.pkl')
            with open(path, 'wb') as f:
                pickle.dump(prob_dict, f)
            print("output probs to: " + dir)

        # calculate dice in original shape
        if self.param_dict['task'] == 'segmentation':
            case_probs = []
            for case, im_id in enumerate(self.param_dict['{}_ids'.format(mode)]):
                #case_prob = sum_prob_by_case[case]
                #case_prob = case_prob / self.param_dict['test_aug_num'] * 1.0
                case_prob = np.mean(prob_dict[im_id], axis=0)
                case_probs.append(case_prob)
                case_pred = np.around(case_prob)
                true_dice.append(self.eval_dice(gt[case], case_pred, axis=(0, 1)))
        else:
            case_probs = np.array(sum_prob_by_case) / self.param_dict['test_aug_num'] * 1.0
            gt = y.copy()

        # overall evaluation results
        print(self.eval(prob=case_probs, y=gt, epoch=epoch, loss=np.mean(all_loss),
                        dice=np.mean(small_dice), true_dice=np.mean(true_dice),
                        mode=mode))

        # save predicted masks
        if self.param_dict['write_output']:
            self.write_output(img_ids=self.param_dict['{}_ids'.format(mode)],
                              prob=case_probs, y=y, mode=mode)

    # ========================= UTILITY FUNCTIONS ======================== #
    def save(self, step, model_name='main'):
        """
        Saving checkpoint
        :param step: ~epoch
        :param model_name:
        """
        checkpoint_dir = os.path.join(self.param_dict['checkpoint_dir'], self.param_dict['model_name'])
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print('Created checkpoing_dir: {}'.format(checkpoint_dir))

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)
        print('** saved checkpoint: ' + str(os.path.join(checkpoint_dir, model_name)) + " step:" + str(step))

    def load(self, model_name='main'):
        """
        Load checkpoint
        :param model_name:
        :return: boolean load sucess, epoch
        """
        checkpoint_dir = os.path.join(self.param_dict['checkpoint_dir'], self.param_dict['model_name'])
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print('Loaded', os.path.join(checkpoint_dir, ckpt_name))

            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return True, counter
        else:
            print("Failed to find a checkpoint")
            return False, 0

    def get_train_id(self):
        """
        get a balanced set of train samples.
        :return: list of im_ids
        """
        ids = []
        df = self.param_dict['df']

        if self.param_dict['task'] == 'segmentation':
            _ids = df.loc[(df['fold'] == 'train')]['img_id'].tolist()
            if self.param_dict['final']:
                _ids += df.loc[(df['fold'] == 'test')]['img_id'].tolist()

            random.shuffle(_ids)
            ids += _ids[0:np.minimum(len(_ids), self.param_dict['train_num_per_class'])]

        else:
            for label in self.param_dict['labels']:
                _ids = df.loc[(df['fold'] == 'train') & (df[self.param_dict['column']] == label)]['img_id'].tolist()
                if self.param_dict['final']:
                    _ids += df.loc[(df['fold'] == 'test') & (df[self.param_dict['column']] == label)]['img_id'].tolist()
                random.shuffle(_ids)
                ids += _ids[0:np.minimum(len(_ids), self.param_dict['train_num_per_class'])]

        random.shuffle(ids)
        return ids

    def get_data(self, mode_id, test_aug_id=-1):
        """
        read and pre-process image
        :param mode_id:
        :return: images and labels in the right dimensions
        """

        ims = []
        labels = []
        aug_dicts = []

        for im_id in self.param_dict[mode_id]:
            # read im
            im = scipy.ndimage.imread(self.param_dict['data_dir'] + 'image/' + im_id, mode='L')
            mask = scipy.ndimage.imread(self.param_dict['data_dir'] + 'label/' + im_id, mode='L')

            # image augmentation
            if 'train' in mode_id:
                im, mask = self.augment_im(im, mask)
            else:
                im, mask, aug_dict = self.test_augment_im(im, mask, test_aug_id)
                aug_dicts.append(aug_dict)

                if self.param_dict['crop'] and self.param_dict['task'] == 'classification':
                    # crop based on mask
                    x0, x1, y0, y1 = self.get_crop_range(mask, fix_margin=10, random_range=None)
                    im = im[y0:y1, x0:x1]

            # resize im
            im = resize(im, self.param_dict['im_size'], mode='constant', preserve_range=True)

            # normalize im
            im = np.clip(im, 0, 255)
            im = im / 255.0
            # im = (im - np.min(im)) / np.float((np.max(im) - np.min(im)))  # (0-1)
            # im = (im - np.mean(im)) / np.std(im)  # 0-mean uni-variance

            # load label
            if self.param_dict['task'] == 'segmentation':
                _label = mask > 100
                label = resize(_label, self.param_dict['im_size'], mode='constant', preserve_range=True)
                # label = np.zeros(self.param_dict['im_size'] + [2])
                # label[..., 0] = _label >= 0.5
                # label[..., 1] = _label < 0.5
            else:
                label = self.load_label(self.param_dict['df'], im_id)

            ims.append(im)
            labels.append(label)

        ims = np.expand_dims(ims, -1).astype(np.float32)
        labels = np.array(labels).astype(np.float32)

        return ims, labels, aug_dicts

    def test_augment_im(self, im, mask, test_aug_id):
        """
        Testing augmentation on images
        :param im:
        :param mask:
        :param test_aug_id:
        :return: augmented image, augmented_gt, augmentation dict describing the augmentation done
        """
        aug_dict = {'test_aug_id': test_aug_id,
                    'original_shape': im.shape,
                    'shift': [False, 0, 0, 0, 0],
                    'flip': False}

        if test_aug_id == -1:
            return im, mask, aug_dict

        # max aug_id = 9
        _min, _max = np.min(im), np.max(im)

        if test_aug_id < 3:
            # contrast
            aug_min = [15, _min, 5]
            aug_max = [_max, int(0.98 * _max), int(0.96 * _max)]
            im = np.clip(im, a_min=aug_min[test_aug_id], a_max=aug_max[test_aug_id])

        elif test_aug_id < 6:
            # shift
            aug_pad = [[0, 4, 1, 0], [3, 0, 5, 7], [1, 2, 2, 1]]
            pad = aug_pad[test_aug_id - 3]
            h, w = np.shape(im)
            _im = im.copy()
            _mask = mask.copy()
            im = np.zeros([h + pad[0] + pad[1], w + pad[2] + pad[3]])
            mask = np.zeros([h + pad[0] + pad[1], w + pad[2] + pad[3]])
            im[pad[0]:h + pad[0], pad[2]: w + pad[2]] = _im
            mask[pad[0]:h + pad[0], pad[2]: w + pad[2]] = _mask
            aug_dict['shift'] = [True, ] + pad
        else:
            # noise
            aug_noise = [[-5, 10], [-8, 10], [-1, 1], [-8, 5]]
            rand = aug_noise[test_aug_id - 6]
            noise = np.random.randint(rand[0], rand[1], size=im.shape, dtype='int64')
            im = im + noise
            im = np.clip(im, _min, _max)  # clip values based on original min max

        if test_aug_id % 3 == 0:
            # flip
            im = im[:, ::-1]
            mask = mask[:, ::-1]
            aug_dict['flip'] = True

        return im, mask, aug_dict

    def reverse_test_augment_im(self, mask, aug_dict):
        '''
        reverse all testing augmentation. Thus, perform in reverse order of processing steps
        :param mask: [h, w]
        :param aug_dict: contains all the information about performed augmentation
        :return: mask in original shape
        '''

        if aug_dict['flip']:
            # flip
            mask = mask[:, ::-1]

        if aug_dict['shift'][0]:
            # shift
            pad = aug_dict['shift'][1:]
            h, w = aug_dict['original_shape']
            new_shape = [h + pad[0] + pad[1], w + pad[2] + pad[3]]
            mask = resize(mask, new_shape, mode='constant', preserve_range=True)
            mask = mask[pad[0]:h + pad[0], pad[2]: w + pad[2]]
        else:
            mask = resize(mask, aug_dict['original_shape'], mode='constant', preserve_range=True)
        return mask

    def augment_im(self, im, mask):
        '''
        randomly augment the input image with: contrast adjusting, flipping, rotation, noise
        :param im:
        :return: augmented image
        '''
        _min, _max = np.min(im), np.max(im)

        if self.param_dict['distort'] and np.random.uniform() > 0.5:
            im, mask = self.elastic_transform(im, mask, im.shape[1] * 2, im.shape[1] * 0.08)

        # contrast adjusting
        if np.random.uniform() > 0.5:
            im = np.clip(im, a_min=np.random.uniform(_min, _min + 10), a_max=np.random.uniform(0.95, 1) * _max)

        # flip left-right
        if np.random.uniform() > 0.5:
            im = im[:, ::-1]
            if mask is not None:
                mask = mask[:, ::-1]

        # shift
        if np.random.uniform() > 0.5:
            pad = [0, 0, 0, 0]
            for i in range(4):
                if np.random.uniform() > 0.2:
                    pad[i] = np.random.randint(1, 10)

            h, w = np.shape(im)
            _im = im.copy()
            _mask = mask.copy()
            im = np.zeros([h + pad[0] + pad[1], w + pad[2] + pad[3]])
            mask = np.zeros([h + pad[0] + pad[1], w + pad[2] + pad[3]])
            im[pad[0]:h + pad[0], pad[2]: w + pad[2]] = _im
            mask[pad[0]:h + pad[0], pad[2]: w + pad[2]] = _mask

        # rotate (max 20 degree)
        if np.random.uniform() > 0.5:
            rot_degree = np.random.uniform(-1, 1) * 20.0
            im = scipy.ndimage.rotate(im, rot_degree, reshape=False)
            if mask is not None:
                mask = scipy.ndimage.rotate(mask, rot_degree, reshape=False)

        # add noise
        if np.random.uniform() > 0.5:
            noise = np.random.randint(-np.random.randint(5, 10), np.random.randint(5, 10), size=im.shape, dtype='int64')
            im = im + noise
            im = np.clip(im, _min, _max)  # clip values based on original min max

        # Crop based on mask
        if self.param_dict['task'] == 'classification' and self.param_dict['crop']:
            x0, x1, y0, y1 = self.get_crop_range(mask, random_range=20)
            im = im[y0:y1, x0:x1]
        return im, mask

    def elastic_transform(self, im, mask, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
            .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
               Convolutional Neural Networks applied to Visual Document Analysis", in
               Proc. of the International Conference on Document Analysis and
               Recognition, 2003.
            """
        assert len(im.shape) == 2
        if random_state is None:
            random_state = np.random.RandomState(None)
        shape = im.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        return map_coordinates(im, indices, order=1).reshape(shape), \
               map_coordinates(mask, indices, order=1).reshape(shape)

    def get_crop_range(self, mask, fix_margin=10, random_range=None):
        """
        Crop image center at mask
        :param mask: segmentation mask [h, w]
        :param fix_margin: fix padding on all four edges of the mask
        :param random_range: random padding
        :return: 4 crop indexes
        """
        x0, x1 = np.min(np.nonzero(mask.any(axis=0))[0]), np.max(np.nonzero(mask.any(axis=0))[0])
        y0, y1 = np.min(np.nonzero(mask.any(axis=1))[0]), np.max(np.nonzero(mask.any(axis=1))[0])

        if random_range is not None and np.random.uniform() > 0.5:
            x0 = np.maximum(0, x0 - np.random.randint(4, random_range))
            y0 = np.maximum(0, y0 - np.random.randint(4, random_range))
            x1 = np.minimum(mask.shape[1], x1 + np.random.randint(4, random_range))
            y1 = np.minimum(mask.shape[0], y1 + np.random.randint(4, random_range))
        else:
            x0 = np.maximum(0, x0 - fix_margin)
            y0 = np.maximum(0, y0 - fix_margin)
            x1 = np.minimum(mask.shape[1], x1 + fix_margin)
            y1 = np.minimum(mask.shape[0], y1 + fix_margin)
        return x0, x1, y0, y1

    def load_label(self, df, im_id):
        '''
        convert label into right format
        :param df:
        :param im_id:
        :return: one-hot encoded label
        '''

        one_hot = [0, ] * self.param_dict['numClass']
        label = df.loc[df['img_id'] == im_id][self.param_dict['column']].get_values()[0]
        index = self.param_dict['labels'].index(label)
        one_hot[index] = 1
        return one_hot

    def eval(self, prob, y, epoch, loss, dice=None, true_dice=None, mode='Train'):
        # Classification
        if self.param_dict['task'] == 'classification':
            pred_class = np.argmax(prob, axis=1)
            gt_class = np.argmax(y, axis=1)
            cl_accuracy = []
            for cl in range(self.param_dict['numClass']):
                cl_ids = np.where(gt_class == cl)[0]
                cl_accuracy.append(np.mean(pred_class[cl_ids] == gt_class[cl_ids]))

            acc = np.mean(pred_class == gt_class)

            result = "{}| Epoch {}| loss={:.3f}  acc={:.3f}|".format(mode, epoch, loss, acc)
            for cl_id in range(self.param_dict["numClass"]):
                result += " cl{}={:.2f}".format(self.param_dict["labels"][cl_id], cl_accuracy[cl_id])

            if self.param_dict['tensorboard'] and self.param_dict['mode'] == 'train':
                self.write_tensorboard(epoch, mode, loss, acc=acc, cl_accuracy=cl_accuracy)

        # Segmentation
        else:
            result = "{}| Epoch {}| loss={:.3f}  dice={:.3f}  true_dice={}|".format(mode, epoch, loss, dice,
                                                                                    true_dice)

            if self.param_dict['tensorboard'] and self.param_dict['mode'] == 'train':
                self.write_tensorboard(epoch, mode, loss, dice=dice, true_dice=true_dice)

        with open(self.param_dict['output_dir'] + '{}_stat.txt'.format(self.param_dict['model_name']), "a") as fp:
            fp.write(result + " \n")

        return result

    def write_tensorboard(self, epoch, mode, loss, acc=None, dice=None, true_dice=None, cl_accuracy=None):
        if mode == 'Train':
            writer = self.train_writer
        elif mode in ['Valid', 'valid']:
            writer = self.valid_writer
        elif mode in ['Test ', 'test ', 'test']:
            writer = self.test_writer

        writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='loss', simple_value=loss)]), epoch)
        if self.param_dict['task'] == 'classification':
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='acc', simple_value=acc)]), epoch)
            for cl_id in range(self.param_dict["numClass"]):
                tag = self.param_dict['column'] + self.param_dict["labels"][cl_id]
                val = cl_accuracy[cl_id]
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)]), epoch)
        else:
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='dice', simple_value=dice)]), epoch)
            if true_dice is not None:
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='true_dice', simple_value=true_dice)]), epoch)

    def write_output(self, img_ids, prob, y, mode):
        # Classification
        if self.param_dict['task'] == 'classification':
            data = {'img_ids': img_ids,
                    'gt': np.argmax(y, axis=1),
                    'pred': np.argmax(prob, axis=1)}

            for i in range(self.param_dict['numClass']):
                data[self.param_dict['column'] + self.param_dict['labels'][i] + '_prob'] = prob[:, i]

            df = pd.DataFrame(data)
            df.to_csv(self.param_dict['output_dir']+"{}_{}_outputs.csv".format(self.param_dict['model_name'], mode), index=False)

        # segmentation
        else:
            # output dir maintenance
            out_dir = os.path.join(self.param_dict['output_dir'], self.param_dict['model_name'])

            for _id, mask in enumerate(prob):
                im_id = img_ids[_id]
                im = scipy.ndimage.imread(self.param_dict['data_dir'] + '/image/' + im_id, mode='L')
                if len(mask.shape) == 3:
                    mask = np.argmax(mask, -1)
                if mask.shape != im.shape:
                    mask = resize(mask, list(im.shape), mode='constant', preserve_range=True)
                mask *= 255
                # write probabilty map
                cv2.imwrite(out_dir + '/' + im_id, mask)

    def eval_dice(self, target, prediction, axis=(1, 2), smooth=0.01):
        """
        Dice: prediction (0 or 1)
        Soft Dice: prediction (prob 0 to 1)

        Sorenson Dice
        \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
        where T is ground truth mask and P is the prediction mask

        :param target: gt [batch, height, width]
        :param prediction: [batch, height, width]
        """

        intersection = np.sum(target * prediction, axis=axis)
        union = np.sum(target + prediction, axis=axis)
        numerator = 2. * intersection + smooth
        denominator = union + smooth
        coef = numerator / denominator
        return np.mean(coef)
