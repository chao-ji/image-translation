r"""Executable for performing image translation using CycleGAN. 

For help, run
  python run_inferencer.py --help
"""

import glob
import os

import tensorflow as tf
import imageio

from dataset import CycleGANInferencerDataset
from model_runners import CycleGANInferencer
from prediction_model import CycleGANPredictionModel


flags = tf.app.flags

flags.DEFINE_string(
    'domain_x_dir', None, 'Path to the directory holding domain x images.')
flags.DEFINE_string(
    'domain_y_dir', None, 'Path to the directory holding domain x images.')
flags.DEFINE_string('ckpt_filename', None, 'Path to the checkpoint file name.')
flags.DEFINE_string('output_dir', '.', 'Path to the output directory.')
flags.DEFINE_string('extension', 'jpg', 'Extension of input images files.')
flags.DEFINE_integer(
    'ngf', 32, 'Num of output channels of the first Conv2D in generator.')
flags.DEFINE_integer(
    'ndf', 64, 'Num of output channels of the first Conv2D in discriminator.')

FLAGS = flags.FLAGS


def main(_):

  domain_x_filenames = glob.glob(os.path.join(FLAGS.domain_x_dir, '*.' + FLAGS.extension))
  domain_y_filenames = glob.glob(os.path.join(FLAGS.domain_y_dir, '*.' + FLAGS.extension))


  dataset = CycleGANInferencerDataset()
  cyclegan_model = CycleGANPredictionModel(
      ngf=FLAGS.ngf, ndf=FLAGS.ndf)

  inferencer = CycleGANInferencer(cyclegan_model)

  filename, real_images, fake_images, cycle_images = inferencer.infer(
      domain_x_filenames, domain_y_filenames, dataset)

  saver = tf.train.Saver()
  sess = tf.InteractiveSession()

  saver.restore(sess, FLAGS.ckpt_filename)


  while True:
    try:
      n, fake = sess.run([filename, fake_images])
    except tf.errors.OutOfRangeError:
      break

    n = n.decode('utf-8')
    ext = n.split('.')[-1]
    n = os.path.join(FLAGS.output_dir, os.path.basename(n))

    fake = (fake[0] * 127.5 + 127.5).astype('uint8')

    imageio.imwrite(n + '.fake.' + ext , fake)



if __name__ == '__main__':
  tf.flags.mark_flag_as_required('domain_x_dir')
  tf.flags.mark_flag_as_required('domain_y_dir')
  tf.flags.mark_flag_as_required('ckpt_filename')
  tf.app.run()
