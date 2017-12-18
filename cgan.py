from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from scipy.io import savemat
from scipy.misc import imsave
from ops import *


class FaceAging(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=64,  # size the input images
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=100,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=3,  # number of channels of input images
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_z_channels=50,  # number of channels of the layer z (noise or code)
                 num_categories=10,  # number of categories (age segments) in the training dataset
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 ):

        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        # self.dataset_name = dataset_name

        # ************************************* input to graph ********************************************************
        self.input_image = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_images'
        )
        self.type = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_categories],
            name='type_labels'
        )
        self.z_prior = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_z_channels],
            name='z_prior'
        )
        # ************************************* build the graph *******************************************************
        print '\n\tBuilding graph ...'

        # encoder: input image --> z
        self.z = self.encoder(
            image=self.input_image,
        )

        # generator: z + label --> generated image
        self.G = self.generator(
            z=self.z,
            y=self.type,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
        )

        # discriminator on z
        self.D_z, self.D_z_logits = self.discriminator_z(
            z=self.z,
            is_training=self.is_training,
        )

        # discriminator on G
        self.D_G, self.D_G_logits = self.discriminator_img(
            image=self.G,
            y=self.type,
            is_training=self.is_training,
        )

        # discriminator on z_prior
        self.D_z_prior, self.D_z_prior_logits = self.discriminator_z(
            z=self.z_prior,
            is_training=self.is_training,
            reuse_variables=True
        )

        # discriminator on input image
        self.D_input, self.D_input_logits = self.discriminator_img(
            image=self.input_image,
            y=self.type,
            is_training=self.is_training,
            reuse_variables=True
        )

        # def euclidean_distance(x, y):
        #     a = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x, y)), reduction_indices=1))
        #     print ("euclidean tensor: ", a)
        #     return a

        # ************************************* loss functions *******************************************************
        # loss function of encoder + generator
        # taking euclidean distance between orignal and generated image followed by l2 loss estimation
        # self.eucl_dist_diff = euclidean_distance(self.input_image, self.G)
        # self.EG_loss = tf.nn.l2_loss(self.input_image - self.G) / self.size_batch  # L2 loss
        self.EG_loss = tf.reduce_mean(tf.abs(self.input_image - self.G))  # L1 loss

        # loss function of discriminator on z
        self.D_z_loss_prior = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_prior_logits, tf.ones_like(self.D_z_prior_logits))
        )
        self.D_z_loss_z = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_logits, tf.zeros_like(self.D_z_logits))
        )
        self.E_z_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_z_logits, tf.ones_like(self.D_z_logits))
        )
        # loss function of discriminator on image
        self.D_img_loss_input = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_input_logits, tf.ones_like(self.D_input_logits))
        )
        self.D_img_loss_G = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.zeros_like(self.D_G_logits))
        )
        self.G_img_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_G_logits, tf.ones_like(self.D_G_logits))
        )

        # total variation to smooth the generated image
        tv_y_size = self.size_image
        tv_x_size = self.size_image
        self.tv_loss = (
                           (tf.nn.l2_loss(self.G[:, 1:, :, :] - self.G[:, :self.size_image - 1, :, :]) / tv_y_size) +
                           (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.size_image - 1,
                                                                :]) / tv_x_size)) / self.size_batch

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        # variables of encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of generator
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on z
        self.D_z_variables = [var for var in trainable_variables if 'D_z_' in var.name]
        # variables of discriminator on image
        self.D_img_variables = [var for var in trainable_variables if 'D_img_' in var.name]

        # ************************************* collect the summary ***************************************
        self.z_summary = tf.summary.histogram('z', self.z)
        self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.D_z_loss_z_summary = tf.summary.scalar('D_z_loss_z', self.D_z_loss_z)
        self.D_z_loss_prior_summary = tf.summary.scalar('D_z_loss_prior', self.D_z_loss_prior)
        self.E_z_loss_summary = tf.summary.scalar('E_z_loss', self.E_z_loss)
        self.D_z_logits_summary = tf.summary.histogram('D_z_logits', self.D_z_logits)
        self.D_z_prior_logits_summary = tf.summary.histogram('D_z_prior_logits', self.D_z_prior_logits)
        self.D_img_loss_input_summary = tf.summary.scalar('D_img_loss_input', self.D_img_loss_input)
        self.D_img_loss_G_summary = tf.summary.scalar('D_img_loss_G', self.D_img_loss_G)
        self.G_img_loss_summary = tf.summary.scalar('G_img_loss', self.G_img_loss)
        self.D_G_logits_summary = tf.summary.histogram('D_G_logits', self.D_G_logits)
        self.D_input_logits_summary = tf.summary.histogram('D_input_logits', self.D_input_logits)
        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=10)

    def train(self,
              file_names,
              image_labels,
              num_epochs=200,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # used the saved checkpoint to initialize the model
              G_img_learning=0.000,
              E_z_learning=0.000,
              tv_learning=0.000
              ):

        # *************************** load file names of images ******************************************************
        # file_names = glob(os.path.join('./data', self.dataset_name, '*.jpg'))
        size_data = len(file_names)
        np.random.seed(seed=2017)
        if enable_shuffle:
            np.random.shuffle(file_names)

        # *********************************** optimizer **************************************************************
        # over all, there are three loss functions, weights may differ from the paper because of different datasets
        self.loss_EG = self.EG_loss + G_img_learning * self.G_img_loss + \
                       E_z_learning * self.E_z_loss + \
                       tv_learning * self.tv_loss  # slightly increase the params
        self.loss_Dz = self.D_z_loss_prior + self.D_z_loss_z
        self.loss_Di = self.D_img_loss_input + self.D_img_loss_G

        # set learning rate decay
        self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
        EG_learning_rate = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.EG_global_step,
            decay_steps=size_data / self.size_batch * 2,
            decay_rate=decay_rate,
            staircase=True
        )

        # optimizer for encoder + generator
        self.EG_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_EG,
            global_step=self.EG_global_step,
            var_list=self.E_variables + self.G_variables
        )

        # optimizer for discriminator on z
        self.D_z_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Dz,
            var_list=self.D_z_variables
        )

        # optimizer for discriminator on image
        self.D_img_optimizer = tf.train.AdamOptimizer(
            learning_rate=EG_learning_rate,
            beta1=beta1
        ).minimize(
            loss=self.loss_Di,
            var_list=self.D_img_variables
        )

        # *********************************** tensorboard *************************************************************
        # for visualization (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.z_summary, self.z_prior_summary,
            self.D_z_loss_z_summary, self.D_z_loss_prior_summary,
            self.D_z_logits_summary, self.D_z_prior_logits_summary,
            self.EG_loss_summary, self.E_z_loss_summary,
            self.D_img_loss_input_summary, self.D_img_loss_G_summary,
            self.G_img_loss_summary, self.EG_learning_rate_summary,
            self.D_G_logits_summary, self.D_input_logits_summary
        ])
        self.writer = tf.summary.FileWriter(os.path.join(self.save_dir, 'summary'), self.session.graph)

        # ************* get some random samples as testing data to visualize the learning process *********************
        sample_files = file_names[0:self.size_batch]
        # file_names[0:self.size_batch] = []
        sample = [fa_load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        sample_label_type = image_labels[0:self.size_batch]

        # ******************************************* training *******************************************************
        print '\n\tPreparing for training ...'

        # initialize the graph
        tf.global_variables_initializer().run()

        # load check point
        if use_trained_model:
            if self.load_checkpoint():
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        # epoch iteration
        num_batches = len(file_names) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(file_names)
            for ind_batch in range(num_batches):
                start_time = time.time()
                # read batch images and labels
                batch_files = file_names[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch = [fa_load_image(
                    image_path=batch_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for batch_file in batch_files]
                if self.num_input_channels == 1:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)
                batch_label_type = image_labels[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)

                # update
                labels = self.session.run(tf.one_hot(indices=batch_label_type, depth=self.num_categories))

                _, _, _, EG_err, Ez_err, Dz_err, Dzp_err, Gi_err, DiG_err, Di_err, TV = self.session.run(
                    fetches=[
                        self.EG_optimizer,
                        self.D_z_optimizer,
                        self.D_img_optimizer,
                        self.EG_loss,
                        self.E_z_loss,
                        self.D_z_loss_z,
                        self.D_z_loss_prior,
                        self.G_img_loss,
                        self.D_img_loss_G,
                        self.D_img_loss_input,
                        self.tv_loss
                    ],
                    feed_dict={
                        self.input_image: batch_images,
                        self.type: labels,
                        self.z_prior: batch_z_prior
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tTV=%.4f" %
                      (epoch + 1, num_epochs, ind_batch + 1, num_batches, EG_err, TV))
                print("\tEz=%.4f\tDz=%.4f\tDzp=%.4f" % (Ez_err, Dz_err, Dzp_err))
                print("\tGi=%.4f\tDi=%.4f\tDiG=%.4f" % (Gi_err, Di_err, DiG_err))

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # add to summary
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image: batch_images,
                        self.type: batch_label_type,
                        self.z_prior: batch_z_prior
                    }
                )
                self.writer.add_summary(summary, self.EG_global_step.eval())

            # save sample images for each epoch
            name = '{:02d}.png'.format(epoch + 1)
            self.sample(sample_images, sample_label_type, name)
            # self.test(sample_images, sample_label_gender, name)

            # save checkpoint for each 10 epoch
            if np.mod(epoch, 10) == 9:
                self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()

    def encoder(self, image, reuse_variables=False):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'E_conv' + str(i)
            current = fa_conv2d(
                input_map=current,
                num_output_channels=self.num_encoder_channels * (2 ** i),
                size_kernel=self.size_kernel,
                name=name
            )
            current = tf.nn.relu(current)

        # fully connection layer
        name = 'E_fc'
        current = fa_fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=self.num_z_channels,
            name=name
        )

        # output
        return tf.nn.tanh(current)

    def generator(self, z, y, reuse_variables=False, enable_tile_label=True, tile_ratio=1.0):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / self.num_categories)
        else:
            duplicate = 1
        z = fa_concat_label(z, y, duplicate=duplicate)
        if enable_tile_label:
            duplicate = int(self.num_z_channels * tile_ratio / 2)
        else:
            duplicate = 1
        size_mini_map = int(self.size_image / 2 ** num_layers)
        # fc layer
        name = 'G_fc'
        current = fa_fc(
            input_vector=z,
            num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
            name=name
        )
        print "Current: ", current
        # reshape to cube for deconv
        current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
        print "Current: ", current
        current = tf.nn.relu(current)
        print "Current: ", current
        # deconv layers with stride 2
        for i in range(num_layers):
            name = 'G_deconv' + str(i)
            current = fa_deconv2d(
                input_map=current,
                output_shape=[self.size_batch,
                              size_mini_map * 2 ** (i + 1),
                              size_mini_map * 2 ** (i + 1),
                              int(self.num_gen_channels / 2 ** (i + 1))],
                size_kernel=self.size_kernel,
                name=name
            )
            print "Current: ", current
            current = tf.nn.relu(current)
        name = 'G_deconv' + str(i + 1)
        current = fa_deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          int(self.num_gen_channels / 2 ** (i + 2))],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )
        print "Current: ", current
        current = tf.nn.relu(current)
        name = 'G_deconv' + str(i + 2)
        current = fa_deconv2d(
            input_map=current,
            output_shape=[self.size_batch,
                          self.size_image,
                          self.size_image,
                          self.num_input_channels],
            size_kernel=self.size_kernel,
            stride=1,
            name=name
        )
        print "Current: ", current

        # output
        return tf.nn.tanh(current)

    def discriminator_z(self, z, is_training=True, reuse_variables=False, num_hidden_layer_channels=(64, 32, 16),
                        enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        current = z
        print ("desc: ", current)
        # fully connection layer
        for i in range(len(num_hidden_layer_channels)):
            name = 'D_z_fc' + str(i)
            current = fa_fc(
                input_vector=current,
                num_output_length=num_hidden_layer_channels[i],
                name=name
            )
            print ("desc: ", current)
            if enable_bn:
                name = 'D_z_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
                print ("desc: ", current)
            current = tf.nn.relu(current)
            print ("desc: ", current)
        # output layer
        name = 'D_z_fc' + str(i + 1)
        current = fa_fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        print ("desc: ", current)
        return tf.nn.sigmoid(current), current

    def discriminator_img(self, image, y, is_training=True, reuse_variables=False,
                          num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        num_layers = len(num_hidden_layer_channels)
        current = image
        # conv layers with stride 2
        for i in range(num_layers):
            name = 'D_img_conv' + str(i)
            current = fa_conv2d(
                input_map=current,
                num_output_channels=num_hidden_layer_channels[i],
                size_kernel=self.size_kernel,
                name=name
            )
            print ("desc img: ", current)
            if enable_bn:
                name = 'D_img_bn' + str(i)
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse_variables
                )
                print ("desc img: ", current)
            current = tf.nn.relu(current)
            print ("desc img: ", current)
            if i == 0:
                current = fa_concat_label(current, y)
        # fully connection layer
        name = 'D_img_fc1'
        current = fa_fc(
            input_vector=tf.reshape(current, [self.size_batch, -1]),
            num_output_length=1024,
            name=name
        )
        print ("desc img: ", current)
        current = lrelu(current)
        print ("desc img: ", current)
        name = 'D_img_fc2'
        current = fa_fc(
            input_vector=current,
            num_output_length=1,
            name=name
        )
        print ("desc img: ", current)
        # output
        return tf.nn.sigmoid(current), current

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
            return True
        else:
            return False

    def sample(self, images, labels, name):
        sample_dir = os.path.join(self.save_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        z, G = self.session.run(
            [self.z, self.G],
            feed_dict={
                self.input_image: images,
                self.type: labels
            }
        )
        size_frame = int(np.sqrt(self.size_batch))
        fa_save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )