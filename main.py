import os, sys
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from model import Model
import warnings
warnings.filterwarnings('ignore')

flags = tf.app.flags
flags.DEFINE_string("model_type", "sgunet", "unet/unet++/sgunet")

flags.DEFINE_string("task", "segmentation", "Accept: segmentation, classification")
flags.DEFINE_string("mode", "train", "Accept: train, valid, test")
flags.DEFINE_string("GPU", "0", "define which GPU to use")
flags.DEFINE_string("loss", "cedice", "define loss type. acceptable value: cedice, ce, dice, sorenson")
flags.DEFINE_integer('initial_feature', 32, "define initial feature number")
flags.DEFINE_integer('num_layers', 4, "define number of layers used in model")
flags.DEFINE_integer('epoch', 3000, "define number of epoch used to train model")
flags.DEFINE_string("column", "maligns", "define the column for classification")
flags.DEFINE_bool("crop", False, "whether crop based on mask")
flags.DEFINE_bool("no_test_augmentation", True, "whether using testing augmentation. False (using test aug) is better.")
flags.DEFINE_bool("test_aug_prob_out", False, "whether output the probability maps during test augmentation")
flags.DEFINE_string("backbone", "", "1) unet is empty 2)res18 ")
flags.DEFINE_string("cv", "0", "which fold of cross-validation")




FLAGS = flags.FLAGS


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    if FLAGS.cv != "":
        df = pd.read_csv('../data/cv_{}.csv'.format(FLAGS.cv))
    else:
        df = pd.read_csv('../data/label.csv')

    param_dict = {
        'mode': FLAGS.mode,
        'task': FLAGS.task,
        'column': FLAGS.column,

        'train_ids': df.loc[df['fold'] == 'train']['img_id'].tolist(),
        'valid_ids': df.loc[df['fold'] == 'valid']['img_id'].tolist(),
        'test_ids': df.loc[df['fold'] == 'test']['img_id'].tolist(),
        'df': df,
        'print_network': False,

        'checkpoint_dir': './checkpoints/',
        'log_dir': './logs/',
        'output_dir': './output/no_testAug/' if FLAGS.no_test_augmentation else './output/',
        'pred_seg_dir': './pred_seg/',
        'data_dir': '../data/',

        'tensorboard': True,
        'write_output': True,
        'test_augmentation': not FLAGS.no_test_augmentation,
        'test_aug_num': 0 if FLAGS.no_test_augmentation else 10,
        'test_aug_prob_out': FLAGS.test_aug_prob_out,

        # Hyper-Parameters which can be tuned
        'epoch': FLAGS.epoch,
        'im_size': [128, 128],
        'valid_per_epoch': 100,
        'lr': 1e-4,
        'loss_type': FLAGS.loss,
        'initial_feature': FLAGS.initial_feature,
        'layer_fs': [32, 64, 128, 256],
        'layer_fs_res': [64, 128, 256, 512],
        'num_layers': FLAGS.num_layers,
        'num_class': 2,
        'crop': FLAGS.crop,
        "distort": True,
        'avgpool': True,
        'no_self_decn': False,
        'backbone': FLAGS.backbone,
        'cv': FLAGS.cv,
        'model_type': FLAGS.model_type,
        
        'train_num_per_class':50,# batchsize
    }

    param_dict['model_name'] = f"{param_dict['model_type']}_cv{FLAGS.cv}"

    if FLAGS.GPU != 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
        run_config = tf.ConfigProto(gpu_options=gpu_options)
    else:
        run_config = tf.ConfigProto()

    with tf.Session(config=run_config) as sess:
        model = Model(sess, param_dict)
        if FLAGS.mode == 'train':
            model.train()
        else:
            model.test(mode=FLAGS.mode)

    tf.reset_default_graph()


if __name__ == '__main__':
    tf.compat.v1.app.run()
