
import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.layers import Input, Embedding
import matplotlib as mpl
import matplotlib.pyplot as plt

from Loss import Pre_loss
from utils import CryMat_Gen
from utils import data_pro
from Loss import mse_metric, pred_loss_1
mpl.use('TkAgg')


class Forward_predictor(object):
    def __init__(self, sess, Cap_model, MatCheCon_model, cry_graph, str_data, bg_data, mode,
                 learning_rate=0.0005, batch_size = 48, job_dir='./Results/0705', n_epoch=100):
        self.sess = sess
        self.MatCheCon_model = MatCheCon_model
        self.Cap_model = Cap_model
        self.cry_graph = cry_graph
        self.mode = mode
        self.pred_loss = Pre_loss()
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.str_data = str_data
        self.bg_data = bg_data


    def load_datagenerator(self, data_mode = None):

        train_data, train_tar,test_data, test_tar = data_pro(self.bg_data, data_mode,
                                                             self.str_data, self.cry_graph)
        self.test_data = test_data
        self.test_tar = test_tar
        self.train_tar = train_tar

        self.train_gen, self.steps_per_train = CryMat_Gen(train_graphs = train_data,
                                                          train_targets = train_tar,
                                                          graph_converter = self.cry_graph,
                                                          batch_size = self.batch_size,
                                                          mode = 'train',).load_train_data()
        #self.train_generator, self.steps_per_train = self.train_gen.load_train_data()

        self.val_gen, self.steps_per_val = CryMat_Gen(val_graphs = test_data,
                                                      val_targets = test_tar,
                                                      graph_converter = self.cry_graph,
                                                      batch_size = self.batch_size,
                                                      mode = 'val').load_val_data()
        #self.val_generator, self.steps_per_val = self.val_gen.load_val_data()


    def forward_predictor(self, nbond = None, atom_emb = 108, emb_dim = 80, global_emb_dim = None):

        com_w = Input(shape=(None,), dtype=DataType.tf_float, name="composition_weights")
        ele_idx = Input(shape=(None,), dtype=DataType.tf_int, name="elements_index")
        atom_fea = Input(shape=(None, 103), dtype=DataType.tf_float, name="elements_fea")
        fea = Input(shape=(None,), dtype=DataType.tf_int, name="fea_list")
        nbr = Input(shape=(None,), dtype=DataType.tf_int, name="nvr_list")
        atom_vec = Input(shape=(None, ), dtype=DataType.tf_int, name="atom_input")
        atom_vec_ = Embedding(atom_emb, emb_dim, name="atom_emb")(atom_vec)

        bond_vec = Input(shape=(None, nbond), name="bond_feature_input")
        bond_vec_ = bond_vec

        state_vec = Input(shape=(None,), dtype=DataType.tf_int, name="state_bias")
        state_vec_ = Embedding(1, global_emb_dim, name="state_emb")(state_vec)

        atom_index = Input(shape=(None, ), dtype=DataType.tf_int, name="target_atom_index")
        nei_index = Input(shape=(None, ), dtype=DataType.tf_int, name="neighbor_index")
        atom_sou = Input(shape=(None, ), dtype=DataType.tf_int, name="atom_source")
        bond_sou = Input(shape=(None, ), dtype=DataType.tf_int, name="bond_source")
        real_bp = Input(shape=(None, ), dtype=DataType.tf_float, name="ground_truth_data")

        ele_inp = com_w, ele_idx, atom_fea, fea, nbr
        mat_inp = atom_vec_, bond_vec_, state_vec_
        index_inp = atom_index, nei_index, atom_sou, bond_sou

        # chemical environment process
        mat_rep_output = self.MatCheCon_model(ele_inp, mat_inp, index_inp, real_bp, n_loop = 3, alpha = 0.6)

        # capsule transformer and prediction
        final_vec, real_bg = mat_rep_output
        likelihood_loss, mse_loss, rec_mse_loss, \
        pace_pre = self.Cap_model(final_vec, real_bg)

        self.inputs = [com_w, ele_idx, atom_fea, fea, nbr, atom_vec, bond_vec, state_vec,
                       atom_index, nei_index, atom_sou, bond_sou, real_bp]

        outputs = [likelihood_loss, mse_loss, rec_mse_loss, pace_pre]

        #with tf.Session() as sess:
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
            #session.run(tf.tables_initializer())
        self.model = tf.keras.Model(inputs = self.inputs, outputs = outputs)

        print("--- Models initialize 100% ---")
            #plot_model(self.model, to_file='./Results/0705/model_structure.png', show_shapes=True)

        self.optimizer = tf.keras.optimizers.RMSprop(lr = self.learning_rate)

        self.losses = [{'tf_op_layer_material_autoencoder/mse_loss': self.Cap_model._mse_loss}]


    def train_model(self):
        # load data and build model
        # build data generator for training process
        self.load_datagenerator(data_mode=self.mode)

        #self.load_datagen(data_mode=self.mode)
        self.forward_predictor(nbond= 100, global_emb_dim = 64)

        #with tf.Session() as sess:
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.local_variables_initializer())
        for epoch in range(self.n_epoch):
            print("\nStart of epoch %d" % (epoch,))

            for l, loss_function in enumerate(self.losses):

                self.model.compile(optimizer =  self.optimizer,
                                        loss =       loss_function,
                                        metrics =    [mse_metric])


                self.model.fit_generator(generator = self.train_gen,
                                         steps_per_epoch = self.steps_per_train,
                                         validation_data = self.val_gen,
                                         validation_steps = self.steps_per_val,
                                         use_multiprocessing = False,
                                         epochs = 2000,
                                         #callbacks = callbacks,
                                         shuffle=False,
                                         #initial_epoch = init_epoch,
                                         )

                layer_2_out = self.model.get_layer('activation').output
                inter_2 = tf.keras.Model(inputs=self.model.input, outputs=layer_2_out)
                layer_2_out = inter_2.predict(self.train_gen)
                np.savetxt('./Results/0902/re_data_1/hf_1_data/layer_1_out_%s_%d.csv' % (epoch, l), layer_2_out,
                           delimiter=',')

                train_targets = self.train_gen.targets
                pred_target = self.model.predict_generator(self.train_gen)

                a = np.array(pred_target[-1]).reshape(-1)
                c = np.array(self.train_tar).reshape(-1)

                np.savetxt('./Results/0902/re_data_1/pre_data/pre_deta_%s-%d.csv' % (epoch, l), a, delimiter=',')
                np.savetxt('./Results/0902/re_data_1/pre_data/real_deta_%s-%d.csv' % (epoch, l), c, delimiter=',')
                l1_loss = np.mean(np.abs(a - c))
                print("prediction L1 loss - %d: " % l, l1_loss)

                fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(18, 6))
                style = dict(size=32, color='gray')
                axs.plot(range(200), a[200:400], 'b')
                axs.plot(range(200), c[200:400], 'r')
                fig.suptitle('Prediction & Ground truth data - %s-%d' % (epoch, l))
                    # plt.yticks([0, 0.5, 1])
                    # plt.xticks([0, 0.5, 1])
                plt.savefig('./Results/0902/re_data_1/figure/plot_gen & real_%d_%s' % (epoch, l))
                plt.close()
                    # plt.show()

                fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 6))
                style = dict(size=32, color='gray')
                axs.scatter(c, a, c='b', alpha=0.5)
                fig.suptitle('Prediction & Ground truth data %s-%d' % (epoch, l))
                plt.yticks([0, 0.5, 1])
                plt.xticks([0, 0.5, 1])
                plt.savefig('./Results/0902/re_data_1/figure/scatter_gen & real_%d_%s' % (epoch, l))
                plt.close()
                # plt.show()

                # validation and test
                pred_val_target = self.model.predict_generator(self.val_gen)

                b = np.array(pred_val_target[-1]).reshape(-1)
                d = np.array(self.test_tar).reshape(-1)
                np.savetxt('./Results/0902/re_data_1/val_pre_data/val_data_%s-%d.csv' % (epoch, l), b, delimiter=',')
                np.savetxt('./Results/0902/re_data_1/val_pre_data/real_val_data_%s-%d.csv' % (epoch, l), d,
                               delimiter=',')
                l1_loss = np.mean(np.abs(b - d))
                print("prediction L1 loss of Val - %d: " % l, l1_loss)

                fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 6))
                style = dict(size=32, color='gray')
                axs.scatter(d, b, c='b', alpha=0.5)
                fig.suptitle('Val Prediction & Ground truth data %s-%d' % (epoch, l))
                plt.yticks([0, 0.5, 1])
                plt.xticks([0, 0.5, 1])
                plt.savefig('./Results/0902/re_data_1/val_figure/val_scatter_gen & real_%d_%s' % (epoch, l))
                plt.close()


            if epoch % 5  == 0 and epoch != 0:
                atom_inp = self.model.get_layer('atom_embedding').output[0]
                preblock_atom_0 = self.model.get_layer('preblock_atom_0').output[0]
                preblock_atom_1 = self.model.get_layer('preblock_atom_1').output[0]
                block_add_atom_0 = self.model.get_layer('block_add_atom_0').output[0]
                block_add_atom_1 = self.model.get_layer('block_add_atom_1').output[0]
                block_add_atom_2 = self.model.get_layer('block_add_atom_2').output[0]
                set2set_atom = self.model.get_layer('set2set_atom').output[0]
                multiply = self.model.get_layer('multiply').output[0]
                set2set_bond = self.model.get_layer('set2set_bond').output[0]
                lstm = self.model.get_layer('lstm').output[0]
                dense_out = self.model.get_layer('tf_op_layer_material_autoencoder/dense_out').output



                atom_inp_0 = tf.keras.Model(inputs=self.model.input, outputs=atom_inp)
                inter_layer_0 = tf.keras.Model(inputs=self.model.input, outputs=preblock_atom_0)
                inter_layer_1 = tf.keras.Model(inputs = self.model.input, outputs = preblock_atom_1)
                inter_layer_2 = tf.keras.Model(inputs=self.model.input, outputs=block_add_atom_0)
                inter_layer_3 = tf.keras.Model(inputs=self.model.input, outputs=block_add_atom_1)
                inter_layer_4 = tf.keras.Model(inputs=self.model.input, outputs=block_add_atom_2)
                set_atom_layer = tf.keras.Model(inputs=self.model.input, outputs=set2set_atom)
                multiply_layer = tf.keras.Model(inputs=self.model.input, outputs=multiply)
                set_bond_layer = tf.keras.Model(inputs=self.model.input, outputs=set2set_bond)
                lstm_layer = tf.keras.Model(inputs=self.model.input, outputs=lstm)
                dense_layer = tf.keras.Model(inputs=self.model.input, outputs=dense_out)

                atom_inp_out = atom_inp_0.predict(self.train_gen)
                atom_0_out = inter_layer_0.predict(self.train_gen)
                atom_1_out = inter_layer_1.predict(self.train_gen)
                atom_2_out = inter_layer_2.predict(self.train_gen)
                atom_3_out = inter_layer_3.predict(self.train_gen)
                atom_4_out = inter_layer_4.predict(self.train_gen)
                set_atom_out = set_atom_layer.predict(self.train_gen)
                multiply_out = multiply_layer.predict(self.train_gen)
                set_bond_out = set_bond_layer.predict(self.train_gen)
                lstm_out = lstm_layer.predict(self.train_gen)
                dense_out = dense_layer.predict(self.train_gen)

                np.savetxt('./Results/0902/mid_out/atom_inp_%s_%d.csv' % (epoch, l), atom_inp_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/atom_0_out_%s_%d.csv' % (epoch, l), atom_0_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/atom_1_out_%s_%d.csv' % (epoch, l), atom_1_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/atom_2_out_%s_%d.csv' % (epoch, l), atom_2_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/atom_3_out_%s_%d.csv' % (epoch, l), atom_3_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/atom_4_out_%s_%d.csv' % (epoch, l), atom_4_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/set_atom_%s_%d.csv' % (epoch, l), set_atom_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/multiply_%s_%d.csv' % (epoch, l), multiply_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/set_bond_%s_%d.csv' % (epoch, l), set_bond_out, delimiter=',')
                np.savetxt('./Results/0902/mid_out/lstm_%s_%d.csv' % (epoch, l), lstm_out, delimiter=',')

                for i in range(dense_out.shape[0]):
                    a = dense_out[i]
                    np.savetxt('./Results/0902/mid_out/dense/dense_%s_%d_%s.csv' % (epoch, l, i), a, delimiter=',')


    def predict_model(self, epoch, l):

        pred_val_target = self.model.predict_generator(self.val_gen)

        b = np.array(pred_val_target[-1]).reshape(-1)
        d = np.array(self.test_tar).reshape(-1)
        np.savetxt('./Results/0902/re_data_1/val_pre_data/val_data_%s-%d.csv' % (epoch, l), b, delimiter=',')
        np.savetxt('./Results/0902/re_data_1/val_pre_data/real_val_data_%s-%d.csv' % (epoch, l), d,
                   delimiter=',')
        l1_loss = np.mean(np.abs(b - d))
        print("prediction L1 loss of Val - %d: " % l, l1_loss)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        style = dict(size=32, color='gray')
        axs.scatter(d, b, c='b', alpha=0.5)
        fig.suptitle('Val Prediction & Ground truth data %s-%d' % (epoch, l))
        plt.yticks([0, 0.5, 1])
        plt.xticks([0, 0.5, 1])
        plt.savefig('./Results/0902/re_data_1/val_figure/val_scatter_gen & real_%d_%s' % (epoch, l))
        plt.close()

def reorder_predictions(raw_predictions, num_data, tau):
    """
    Reorder raw prediction array

    Parameters
    ----------
    raw_predictions: shape (num_data * (F - tau), num_atom, 2 * n_classes)
    predictions: shape (num_data, F, num_atom, n_classes)
    """
    if (raw_predictions.shape[0] % num_data != 0 or
            len(raw_predictions.shape) != 3 or
            raw_predictions.shape[2] % 2 != 0):
        raise ValueError('Bad format!')
    n_classes = raw_predictions.shape[2] // 2
    num_atom = raw_predictions.shape[1]
    raw_predictions = raw_predictions.reshape(num_data, -1, num_atom,
                                              n_classes * 2)
    assert np.allclose(raw_predictions[:, tau:, :, :n_classes],
                       raw_predictions[:, :-tau, :, n_classes:])
    predictions = np.concatenate([raw_predictions[:, :, :, :n_classes],
                                  raw_predictions[:, -tau:, :, n_classes:]],
                                 axis=1)
    return predictions


DTYPES = {
    "float32": {"numpy": np.float32, "tf": tf.float32},
    "float16": {"numpy": np.float16, "tf": tf.float16},
    "int32": {"numpy": np.int32, "tf": tf.int32},
    "int16": {"numpy": np.int16, "tf": tf.int16},
}


class DataType:

    np_float = np.float32
    np_int = np.int32
    tf_float = tf.float32
    tf_int = tf.int32

    @classmethod
    def set_dtype(cls, data_type: str) -> None:
        """
        Class method to set the data types
        Args:
            data_type (str): '16' or '32'
        """
        if data_type.endswith("32"):
            float_key = "float32"
            int_key = "int32"
        elif data_type.endswith("16"):
            float_key = "float16"
            int_key = "int16"
        else:
            raise ValueError("Data type not known, choose '16' or '32'")

        cls.np_float = DTYPES[float_key]["numpy"]
        cls.tf_float = DTYPES[float_key]["tf"]
        cls.np_int = DTYPES[int_key]["numpy"]
        cls.tf_int = DTYPES[int_key]["tf"]


