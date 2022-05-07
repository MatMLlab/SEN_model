"""
This is the SNE model running file for property prediction and material design.
The SNE-model consist of two part:
            1. Building the chemical environments of materials;
            2. Capsule transformer
"""
from __future__ import print_function
from __future__ import division
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import gzip
import json
import numpy as np

from models.layers.atom_config import CrystalPro
from pre_block import Forward_predictor
from models.che_env_block import MatCheCon
from models import cap_seq
from models.layers.atom_env import GaussianDistance
from Configs import *
import tensorflow as tf

# GPU SETTING
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict Tensorflow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=40240)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

def main(_):
    FLAGS = tf.compat.v1.app.flags.FLAGS
    config = FLAGS
    FLAGS.__dict__['config'] = config

    model_dict = cap_seq.get(FLAGS)
    cap_model = model_dict.model

    if FLAGS.use_gpu:
        with tf.compat.v1.Session() as sess:
            pre_model = Forward_predictor(sess, Cap_model=cap_model, MatCheCon_model=MatCheCon,
                                          cry_graph=crystal_pro, mode=FLAGS.mode,
                                          str_data=material_data, bg_data=band_gap,
                                          batch_size=FLAGS.batch_size, learning_rate=FLAGS.lr,
                                          n_epoch=FLAGS.training_epoch)
            if FLAGS.phase == 'train':
                pre_model.train_model()
            elif FLAGS.phase == 'test':
                pre_model.predict_model(FLAGS.job_dir)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.compat.v1.Session() as sess:
            pre_model = Forward_predictor(sess, Cap_model=cap_model, MatCheCon_model=MatCheCon,
                                          cry_graph=crystal_pro, mode=FLAGS.mode,
                                          str_data=material_data, bg_data=band_gap,
                                          batch_size=FLAGS.batch_size, learning_rate=FLAGS.lr,
                                          n_epoch=FLAGS.training_epoch)
            if FLAGS.phase == 'train':
                pre_model.train_model()
            elif FLAGS.phase == 'test':
                pre_model.predict_model(FLAGS.job_dir)
            else:
                print('[!]Unknown phase')
                exit(0)

# load input and label datasets
with open('/home/manager/data1/0-BandGap_Prediction_Case/0-Dataset/mp.2019.04.01.json', 'r') as d:
    data = json.load(d)
    material_data = {i['material_id']: i['structure'] for i in data}

with gzip.open('/home/manager/data1/0-BandGap_Prediction_Case/0-Dataset/data_no_structs.json.gz', 'rb') as d:
    band_gap = json.loads(d.read())

crystal_pro = CrystalPro(bond_converter = GaussianDistance(centers=np.linspace(0, 6, 100),
                                                           width=0.5), cutoff=5.0)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    tf.compat.v1.app.run()

