import os
import glob

import tensorflow as tf

flags = tf.app.flags

IMAGE_FILE_EXTENSIONS = 'jpg', 'png'

flags.DEFINE_string('X_images_dir', None, 'Directory holding X domain images.')
flags.DEFINE_string('Y_images_dir', None, 'Directory holding Y domain images.')
flags.DEFINE_string('X_tfrecord_file', 'x.tfrecord', 'Filename of tfrecord file for X domain images.')
flags.DEFINE_string('Y_tfrecord_file', 'y.tfrecord', 'Filename of tfrecord file for Y domain images.')

FLAGS = flags.FLAGS


def _bytes_feature(value):
  if isinstance(value, bytes):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def create_tfrecord_file(images_dir, tfrecord_file):
  files = []
  for e in IMAGE_FILE_EXTENSIONS:
    files.extend(glob.glob(os.path.join(images_dir, '*.%s' % e)))

  writer = tf.python_io.TFRecordWriter(tfrecord_file)

  for fn in files:
    with open(fn, 'rb') as fid:
      image_encoded = fid.read()
    feature = {'image/encoded': _bytes_feature(image_encoded)}
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(example.SerializeToString()) 
  writer.close()


def main(_):
  create_tfrecord_file(FLAGS.X_images_dir, FLAGS.X_tfrecord_file)
  create_tfrecord_file(FLAGS.Y_images_dir, FLAGS.Y_tfrecord_file)


if __name__ == '__main__':
  tf.flags.mark_flag_as_required('X_images_dir')
  tf.flags.mark_flag_as_required('Y_images_dir')
  tf.app.run()
