import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from scipy.misc import imsave, imread, imshow, imresize
import numpy as np

#the implements of leakyRelu
def lrelu(x , alpha = 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):

    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        shape = tf.shape(conv)
        shape = tf.stack(shape)
        conv = tf.reshape(tf.nn.bias_add(conv, biases), shape)

        return conv

def de_conv(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        shape = tf.stack(output_shape)
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:

            deconv = tf.nn.deconv2d(input_, w, output_shape=shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), shape)

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def fully_connect(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()
  with tf.variable_scope(scope or "Linear"):

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))

    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:

      return tf.matmul(input_, matrix) + bias

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse=reuse , updates_collections=None)

def instance_norm(x):

    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def residual(x, output_dims, kernel, strides, name_1, name_2):

    with tf.variable_scope('residual') as scope:

        conv1 = conv2d(x, output_dims, k_h=kernel, k_w=kernel, d_h=strides, d_w=strides, name=name_1)
        conv2 = conv2d(tf.nn.relu(conv1), output_dims, k_h=kernel, k_w=kernel, d_h=strides, d_w=strides, name=name_2)
        resi = x + conv2

        return resi

def deresidual(x, output_shape, kernel, strides, name_1, name_2):

    with tf.variable_scope('residual_un') as scope:

        deconv1 = de_conv(x, output_shape=output_shape, k_h=kernel, k_w=kernel, d_h=strides, d_w=strides, name=name_1)
        deconv2 = de_conv(tf.nn.relu(deconv1), output_shape=output_shape, k_h=kernel, k_w=kernel, d_h=strides, d_w=strides, name=name_2)
        resi = x + deconv2

        return resi

def fa_conv2d(input_map, num_output_channels, size_kernel=5, stride=2, name='conv2d'):
    with tf.variable_scope(name):
        stddev = np.sqrt(2.0 / (np.sqrt(input_map.get_shape()[-1].value * num_output_channels) * size_kernel ** 2))
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, input_map.get_shape()[-1], num_output_channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[num_output_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(input_map, kernel, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(conv, biases)


def fa_fc(input_vector, num_output_length, name='fc'):
    with tf.variable_scope(name):
        stddev = np.sqrt(1.0 / (np.sqrt(input_vector.get_shape()[-1].value * num_output_length)))
        w = tf.get_variable(
            name='w',
            shape=[input_vector.get_shape()[1], num_output_length],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            name='b',
            shape=[num_output_length],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input_vector, w) + b


def fa_deconv2d(input_map, output_shape, size_kernel=5, stride=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        stddev = np.sqrt(1.0 / (np.sqrt(input_map.get_shape()[-1].value * output_shape[-1]) * size_kernel ** 2))
        # filter : [height, width, output_channels, in_channels]
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, output_shape[-1], input_map.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[output_shape[-1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        deconv = tf.nn.conv2d_transpose(input_map, kernel, strides=[1, stride, stride, 1], output_shape=output_shape)
        return tf.nn.bias_add(deconv, biases)


def fa_lrelu(logits, leak=0.2):
    return tf.maximum(logits, leak * logits)


def fa_concat_label(x, label, duplicate=1):
    x_shape = x.get_shape().as_list()
    if duplicate < 1:
        return x
    # duplicate the label to enhance its effect, does it really affect the result?
    label = tf.tile(label, [1, duplicate])
    label_shape = label.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat(1, [x, label])
    elif len(x_shape) == 4:
        x_shape = tf.shape(x)
        # print "x_shape: ", x_shape
        label = tf.reshape(label, [x_shape[0], 1, 1, label_shape[-1]])
        # print "labels: ", label
        return tf.concat(3, [x, label * tf.ones([x_shape[0], x_shape[1], x_shape[2], label_shape[-1]])])


def fa_load_image(
        image_path,  # path of a image
        image_size=64,  # expected size of the image
        image_value_range=(-1, 1),  # expected pixel value range of the image
        is_gray=False,  # gray scale or color image
        is_read=True  # want to read from file or array
):
    if is_read:
        if is_gray:
            image = imread(image_path, flatten=True).astype(np.float32)
        else:
            image = imread(image_path, mode="RGB").astype(np.float32)
    else:
        image = image_path
    image = imresize(image, [image_size, image_size])
    image = image.astype(np.float32) * (image_value_range[-1] - image_value_range[0]) / 255.0 + image_value_range[0]
    return image


def fa_save_batch_images(
        batch_images,  # a batch of images
        save_path,  # path to save the images
        image_value_range=(-1, 1),  # value range of the input batch images
        size_frame=None,  # size of the image matrix, number of images in each row and column
        save=False
):
    # transform the pixcel value to 0~1
    images = (batch_images - image_value_range[0]) / (image_value_range[-1] - image_value_range[0])
    if size_frame is None:
        auto_size = int(np.ceil(np.sqrt(images.shape[0])))
        size_frame = [auto_size, auto_size]
    img_h, img_w = batch_images.shape[1], batch_images.shape[2]
    frame = np.zeros([img_h * size_frame[0], img_w * size_frame[1], 3])
    for ind, image in enumerate(images):
        ind_col = ind % size_frame[1]
        ind_row = ind // size_frame[1]
        if (ind_row * img_h + img_h) > frame.shape[0]:
            break
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = image
    if save:
        imsave(save_path, frame)
    return frame

