import tensorflow as tf

from utils import mkdir_p
from vaegan import vaegan
from utils import *
import os
flags = tf.app.flags

flags.DEFINE_integer("batch_size" , 64, "batch size")
flags.DEFINE_integer("max_iters" , 60000, "the maxmization epoch")
flags.DEFINE_integer("latent_dim" , 128, "the dim of latent code")
flags.DEFINE_integer("learn_rate_init" , 0.0003, "the init of learn rate")
#Please set this num of repeat by the size of your datasets.
# flags.DEFINE_integer("repeat", 10000, "the numbers of repeat for your datasets")
flags.DEFINE_string("path", './data/',
"for example, '/home/ubuntu/workspace/shaivi/VAEGAN_Tensorflow/train_lfw' is the directory of your celebA data")
flags.DEFINE_integer("op", 0, "Training or Test")

FLAGS = flags.FLAGS
if __name__ == "__main__":
    root_log_dir = "./vaeganlogs/logs/lfwa_test"
    vaegan_checkpoint_dir = "./model_vaegan/model.ckpt"
    sample_path = "./vaeganSample/sample"

    mkdir_p(root_log_dir)
    mkdir_p(vaegan_checkpoint_dir)
    mkdir_p(sample_path)

    model_path = vaegan_checkpoint_dir

    batch_size = FLAGS.batch_size
    max_iters = FLAGS.max_iters
    latent_dim = FLAGS.latent_dim

    learn_rate_init = FLAGS.learn_rate_init
    cb_ob = load_data(FLAGS.path, shape=(64, 64), need=6000)
    print ("Loaded train data with shape: ", cb_ob.shape)
    print ("Model path: ", model_path)
    # cb_ob = CelebA(FLAGS.path)

    vaeGan = vaegan(batch_size= batch_size, max_iters= max_iters,
                      model_path= model_path, data_ob=cb_ob, latent_dim= latent_dim,
                      sample_path= sample_path , log_dir= root_log_dir , learnrate_init= learn_rate_init)

    if FLAGS.op == 0:
        vaeGan.build_model_vaegan()
        vaeGan.train()

    else:
        vaeGan.build_model_vaegan()
        test_arr = load_data(FLAGS.path, shape=(64, 64), need=500)
        print ("Loaded test data with shape: ", test_arr.shape)
        vaeGan.test()