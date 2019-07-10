r"""Executable for training CycleGAN for unpaired image translation.

For help, run
  python run_trainer.py --help
"""
import tensorflow as tf

import utils
from dataset import CycleGANTrainerDataset
from prediction_model import CycleGANPredictionModel
from model_runners import CycleGANTrainer


#domain_x_filenames, domain_y_filenames = ['x.tfrecord'], ['y.tfrecord']
#init_learning_rate = 0.0002
#num_minibatches_epoch = 1334
#num_epochs = 200

#max_to_keep = 5
#ngf = 32
#ndf = 64
#weights_init_stddev = 0.02 

#domain_x_pool_size = 50
#domain_y_pool_size = 50
#load_ckpt_path = None 

flags = tf.app.flags
flags.DEFINE_integer('num_minibatches_epoch', None, 
    'Num of minibatches per epoch.')
flags.DEFINE_string('load_ckpt_path', None, 'Path to the checkpoint file to be '
    'loaded. If provided, weights will be restored from the provided checkpoint.')
flags.DEFINE_multi_string(
    'domain_x_filenames', None, 'List of domain x TFRecord filenames.')
flags.DEFINE_multi_string(
    'domain_y_filenames', None, 'List of domain y TFRecord filenames.')

flags.DEFINE_float('init_learning_rate', 0.0002, 'Initial learning rate.')
flags.DEFINE_float('weights_init_stddev', 0.02, 
    'Standard deviation bound for truncated normal weight initializer.')
flags.DEFINE_integer('num_epochs', 200, 'Num of epochs.')
flags.DEFINE_integer('max_to_keep', 5, 'Maximum num of ckpt files to keep.')
flags.DEFINE_integer(
    'ngf', 32, 'Num of output channels of the first Conv2D in generator.')
flags.DEFINE_integer(
    'ndf', 64, 'Num of output channels of the first Conv2D in discriminator.') 
flags.DEFINE_integer('batch_size', 1, 'Batch size')
flags.DEFINE_integer(
    'domain_x_pool_size', 50, 'Pool size of fake images of domain x.')
flags.DEFINE_integer(
    'domain_y_pool_size', 50, 'Pool size of fake images of domain y.')

FLAGS = flags.FLAGS


def main(_):
  
  fake_x_pool = utils.FakeImagePool(FLAGS.domain_x_pool_size)
  fake_y_pool = utils.FakeImagePool(FLAGS.domain_y_pool_size)

  dataset = CycleGANTrainerDataset(
      FLAGS.num_minibatches_epoch, batch_size=FLAGS.batch_size)
  cyclegan_model = CycleGANPredictionModel(
      ngf=FLAGS.ngf, ndf=FLAGS.ndf, 
      weights_init_stddev=FLAGS.weights_init_stddev)

  trainer = CycleGANTrainer(cyclegan_model, batch_size=FLAGS.batch_size)

  lr, global_step = utils.get_learning_rate(
      FLAGS.init_learning_rate, FLAGS.num_epochs)

  optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
  to_be_run_dict = trainer.train(
      FLAGS.domain_x_filenames, FLAGS.domain_y_filenames, 
      dataset, optimizer)

  saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
  sess = tf.InteractiveSession()

  if FLAGS.load_ckpt_path:
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.load_ckpt_path)
    print('\n\n\nRestoring from checkpoint %s...' % latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
  else:
    sess.run(tf.global_variables_initializer())
    print('\n\n\nTrain from scratch...')
  writer = tf.summary.FileWriter('.')

  start = global_step.eval()
  print('\n\n\nStart training from epoch %d' % start)

  num_minibatches_epoch = FLAGS.num_minibatches_epoch
  num_epochs = FLAGS.num_epochs

  for i in range(start, num_epochs):
    print('Epoch:', i, 'Learning rate:', lr.eval())
    sess.run(tf.assign(global_step, i))
    saver.save(sess, 'model', global_step=i)

    for j in range(num_minibatches_epoch):
      # train g x2y
      _, fake_y_images, g_x2y_loss, g_x2y_loss_summary = sess.run([
          to_be_run_dict['g_x2y_update_ops'], 
          to_be_run_dict['fake_y_images'], 
          to_be_run_dict['g_x2y_loss'],
          to_be_run_dict['g_x2y_loss_summary']])
      writer.add_summary(g_x2y_loss_summary, i * num_minibatches_epoch + j)
      fake_y_images = fake_y_pool.get(fake_y_images) 
    
      # train d y1x0
      _, d_y1x0_loss, d_y1x0_loss_summary = sess.run([
          to_be_run_dict['d_y1x0_update_ops'],
          to_be_run_dict['d_y1x0_loss'],
          to_be_run_dict['d_y1x0_loss_summary']],
          {trainer.fake_y_images_discriminator: fake_y_images})
      writer.add_summary(d_y1x0_loss_summary, i * num_minibatches_epoch + j)
    
      # train g y2x
      _, fake_x_images, g_y2x_loss, g_y2x_loss_summary = sess.run([
          to_be_run_dict['g_y2x_update_ops'],
          to_be_run_dict['fake_x_images'],
          to_be_run_dict['g_y2x_loss'],
          to_be_run_dict['g_y2x_loss_summary']])
      writer.add_summary(g_y2x_loss_summary, i * num_minibatches_epoch + j)
      fake_x_images = fake_x_pool.get(fake_x_images)

      # train d x1y0
      _, d_x1y0_loss, d_x1y0_loss_summary = sess.run([
          to_be_run_dict['d_x1y0_update_ops'],
          to_be_run_dict['d_x1y0_loss'],
          to_be_run_dict['d_x1y0_loss_summary']],
          {trainer.fake_x_images_discriminator: fake_x_images})
      writer.add_summary(d_x1y0_loss_summary, i * num_minibatches_epoch + j) 

      print('epoch: %d, iteration: %d/%d, '
          'g_x2y_loss: %f, g_y2x_loss: %f, d_y1x0_loss: %f, d_x1y0_loss: %f' % (
          i, j, num_minibatches_epoch, 
          g_x2y_loss, g_y2x_loss, d_y1x0_loss, d_x1y0_loss))

    ( real_x_images_summary, real_y_images_summary, 
      fake_x_images_summary, fake_y_images_summary, 
      cycle_x_images_summary, cycle_y_images_summary) = sess.run([
          to_be_run_dict['real_x_images_summary'], 
          to_be_run_dict['real_y_images_summary'], 
          to_be_run_dict['fake_x_images_summary'], 
          to_be_run_dict['fake_y_images_summary'], 
          to_be_run_dict['cycle_x_images_summary'], 
          to_be_run_dict['cycle_y_images_summary']])
    writer.add_summary(real_x_images_summary, i)
    writer.add_summary(real_y_images_summary, i)
    writer.add_summary(fake_x_images_summary, i)
    writer.add_summary(fake_y_images_summary, i)
    writer.add_summary(cycle_x_images_summary, i)
    writer.add_summary(cycle_y_images_summary, i)

  saver.save(sess, 'model', global_step=num_epochs)

  writer.close()
  sess.close()


if __name__== '__main__':
  tf.flags.mark_flag_as_required('num_minibatches_epoch')
  tf.flags.mark_flag_as_required('domain_x_filenames')
  tf.flags.mark_flag_as_required('domain_y_filenames')
  tf.app.run()
