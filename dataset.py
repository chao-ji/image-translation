import tensorflow as tf


class CycleGANTrainerDataset(object):
  """Decode TFRecord files into tensors and perform preprocessing and data 
  augmentation.
  """
  def __init__(self, 
               shuffle_buffer_size, 
               batch_size=1, 
               resize_height=286, 
               resize_width=286, 
               crop_height=256, 
               crop_width=256, 
               horizontal_flip=True):
    """Constructor.

    Args:
      shuffle_buffer_size: int scalar, buffer size for shuffling the images.
      batch_size: int scalar, batch size.
      resize_height: int scalar, height of resized image.
      resize_width: int scalar, width of resized image.  
      crop_height: int scalar, height of cropped image.
      crop_width: int scalar, width of cropped image.
      horizontal_flip: bool scalar, whether to randomly horizontally flip 
        the image 
    """  
    self._shuffle_buffer_size = shuffle_buffer_size
    self._batch_size = 1
    self._resize_height = 286
    self._resize_width = 286
    self._crop_height = 256
    self._crop_width = 256
    self._horizontal_flip = True


  def get_tensor_dict(self, domain_x_filenames, domain_y_filenames):
    """Generates tensor dict for training.

    The images will be first resized to `[resize_height, resize_width]`, and be
    randomly cropped to size `[crop_height, crop_width]`.

    Args:
      domain_x_filenames: list of strs, the list of TFRecord filenames.
      domain_y_filenames: list of strs, the list of TFRecord filenames.
    """
    dataset_x = tf.data.TFRecordDataset(domain_x_filenames)
    dataset_y = tf.data.TFRecordDataset(domain_y_filenames)

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))

    dataset = dataset.shuffle(self._shuffle_buffer_size).repeat()

    def _decode_raw_protobuf_string(protobuf_string):
      keys_to_features = {
          'image/encoded': tf.FixedLenFeature((), tf.string, default_value='')}
      tensor_dict = tf.io.parse_single_example(
          protobuf_string, keys_to_features)
      image = tf.cast(tf.image.decode_image(
          tensor_dict['image/encoded'], channels=3), 'float32')
      image.set_shape([None, None, 3])
      return image

    dataset = dataset.map(lambda encoded_x, encoded_y: 
        (_decode_raw_protobuf_string(encoded_x),
         _decode_raw_protobuf_string(encoded_y)))

    def _preprocess(image):
      image = tf.image.resize(image, [self._resize_height, self._resize_width])
      if self._horizontal_flip:
        image = tf.image.random_flip_left_right(image)
      
      image = tf.random_crop(image, [self._crop_height, self._crop_width, 3])
      return (image - 127.5) / 127.5


    dataset = dataset.map(lambda image_x, image_y:
        (_preprocess(image_x), _preprocess(image_y)))

    dataset = dataset.batch(self._batch_size, drop_remainder=True)

    tensor_dict = dataset.make_one_shot_iterator().get_next()
    tensor_dict = { 'real_x_images': tensor_dict[0], 
                    'real_y_images': tensor_dict[1]}
    return tensor_dict


class CycleGANInferencerDataset(object):
  """Perform image preprocessing for inference making using CycleGAN."""
  def __init__(self):
    pass    

  def get_tensor_dict(self, domain_x_filenames=None, domain_y_filenames=None):
    """Generates tensor dict for inference.

    Args:
      domain_x_filenames: list of strs, the list of image filenames.
      domain_y_filenames: list of strs, the list of image filenames.
    """
    if domain_x_filenames is None and domain_y_filenames is None:
      raise ValueError('`domain_x_filenames` and `domain_y_filenames` can\'t be'
          ' both empty.')

    def _get_and_preprocess_image(filename):
      image = tf.expand_dims(tf.cast(tf.image.decode_image(
          tf.read_file(filename), channels=3), 'float32'), axis=0)
      return (image - 127.5) / 127.5

    if domain_x_filenames:
      dataset_x = tf.data.Dataset.from_tensor_slices(domain_x_filenames)
      dataset_x = dataset_x.map(lambda filename: (
          _get_and_preprocess_image(filename), filename, 'x'))

    if domain_y_filenames:
      dataset_y = tf.data.Dataset.from_tensor_slices(domain_y_filenames)
      dataset_y = dataset_y.map(lambda filename: (
          _get_and_preprocess_image(filename), filename, 'y'))

    if domain_x_filenames:
      dataset = dataset_x

    if domain_y_filenames:
      if domain_x_filenames:
        dataset = dataset.concatenate(dataset_y)
      else:
        dataset = dataset_y

    iterator = dataset.make_one_shot_iterator()

    tensor_dict = iterator.get_next()
    tensor_dict[0].set_shape([1, None, None, 3])
    tensor_dict = {'real_images': tensor_dict[0], 
                   'filename': tensor_dict[1], 
                   'domain': tensor_dict[2]}
    return tensor_dict
