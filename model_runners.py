import tensorflow as tf

import utils


class CycleGANTrainer(object):
  """Trainer for CycleGAN."""
  def __init__(self, 
               prediction_model, 
               batch_size=1, 
               crop_height=256, 
               crop_width=256, 
               lambda_x=10., 
               lambda_y=10.):
    """Constructor.

    Args:
      prediction_model: a CycleGANPredictionModel instance.
      batch_size: int scalar, batch size
      crop_height: int scalar, height of cropped image.
      crop_width: int scalar, width of cropped image.
      lambda_x: float scalar, weighting factor for consistency loss of domain x.
      lambda_y: float scalar, weighting factor for consistency loss of domain y.
    """
    self._fake_x_images_discriminator = None
    self._fake_y_images_discriminator = None
    self._prediction_model = prediction_model
    self._batch_size = 1
    self._crop_height = 256
    self._crop_width = 256
    self._lambda_x = 10.
    self._lambda_y = 10.

  @property
  def fake_x_images_discriminator(self):
    return self._fake_x_images_discriminator

  @property
  def fake_y_images_discriminator(self):
    return self._fake_y_images_discriminator

  def train(self, 
            domain_x_filenames, 
            domain_y_filenames, 
            dataset, 
            optimizer):
    """Adds training related ops to the graph.

    Args:
      domain_x_filenames: list of strs, the list of TFRecord filenames.
      domain_y_filenames: list of strs, the list of TFRecord filenames.
      dataset: a CycleGANTrainerDataset instance.
      optimizer: an Optimizer instance.
    """
    if not self._fake_x_images_discriminator:
      self._fake_x_images_discriminator = tf.placeholder(
          'float32', [self._batch_size, self._crop_height, self._crop_width, 3])

    if not self._fake_y_images_discriminator:
      self._fake_y_images_discriminator = tf.placeholder(
          'float32', [self._batch_size, self._crop_height, self._crop_width, 3])

    tensor_dict = dataset.get_tensor_dict(
        domain_x_filenames, domain_y_filenames)
    
    real_x_images = tensor_dict['real_x_images']
    real_y_images = tensor_dict['real_y_images']

    # generated images
    fake_y_images = self._prediction_model.predict_generator_output(
        real_x_images, 'GeneratorX2Y')
    fake_x_images = self._prediction_model.predict_generator_output(
        real_y_images, 'GeneratorY2X')

    cycle_x_images = self._prediction_model.predict_generator_output(
        fake_y_images, 'GeneratorY2X')
    cycle_y_images = self._prediction_model.predict_generator_output(
        fake_x_images, 'GeneratorX2Y')

    # discriminator logits
    real_x_logits = self._prediction_model.predict_discriminator_output(
        real_x_images, 'DiscriminatorX1Y0')
    real_y_logits = self._prediction_model.predict_discriminator_output(
        real_y_images, 'DiscriminatorY1X0')

    fake_x_discriminator_logits = (
        self._prediction_model.predict_discriminator_output(
            self._fake_x_images_discriminator, 'DiscriminatorX1Y0'))
    fake_y_discriminator_logits = (
        self._prediction_model.predict_discriminator_output(
            self._fake_y_images_discriminator, 'DiscriminatorY1X0'))

    # generator logits
    fake_x_logits = self._prediction_model.predict_discriminator_output(
        fake_x_images, 'DiscriminatorX1Y0')
    fake_y_logits = self._prediction_model.predict_discriminator_output(
        fake_y_images, 'DiscriminatorY1X0')

    cycle_consistency_loss_x = self._lambda_x * utils.cycle_consistency_loss(
        real_x_images, cycle_x_images)
    cycle_consistency_loss_y = self._lambda_y * utils.cycle_consistency_loss(
        real_y_images, cycle_y_images)

    # losses
    generator_x2y_loss = utils.least_square_generator_loss(
        fake_y_logits) + cycle_consistency_loss_x + cycle_consistency_loss_y
    generator_y2x_loss = utils.least_square_generator_loss(
        fake_x_logits) + cycle_consistency_loss_x + cycle_consistency_loss_y

    discriminator_x1y0_loss = utils.least_square_discriminator_loss(
        real_x_logits, fake_x_discriminator_logits)
    discriminator_y1x0_loss = utils.least_square_discriminator_loss(
        real_y_logits, fake_y_discriminator_logits)
   
    # update ops
    ( generator_x2y_update_ops, generator_y2x_update_ops, 
      discriminator_x1y0_update_ops, discriminator_y1x0_update_ops
        ) = self._get_update_ops(optimizer, 
            generator_x2y_loss, generator_y2x_loss, 
            discriminator_x1y0_loss, discriminator_y1x0_loss) 

    g_x2y_loss_summary = tf.summary.scalar('g_x2y_loss', generator_x2y_loss)
    g_y2x_loss_summary = tf.summary.scalar('g_y2x_loss', generator_y2x_loss)
    d_x1y0_loss_summary = tf.summary.scalar(
        'd_x1y0_loss', discriminator_x1y0_loss)
    d_y1x0_loss_summary = tf.summary.scalar(
        'd_y1x0_loss', discriminator_y1x0_loss)
 
    real_x_images_summary = tf.summary.image(
        'real_x_images_summary', real_x_images * 127.5 + 127.5)
    real_y_images_summary = tf.summary.image(
        'real_y_images_summary', real_y_images * 127.5 + 127.5)
    fake_x_images_summary = tf.summary.image(
        'fake_x_images_summary', fake_x_images * 127.5 + 127.5)
    fake_y_images_summary = tf.summary.image(
        'fake_y_images_summary', fake_y_images * 127.5 + 127.5)
    cycle_x_images_summary = tf.summary.image(
        'cycle_x_images_summary', cycle_x_images * 127.5 + 127.5)
    cycle_y_images_summary = tf.summary.image(
        'cycle_y_images_summary', cycle_y_images * 127.5 + 127.5)

    to_be_run_dict = {
        'g_x2y_update_ops': generator_x2y_update_ops,
        'g_y2x_update_ops': generator_y2x_update_ops,
        'd_x1y0_update_ops': discriminator_x1y0_update_ops,
        'd_y1x0_update_ops': discriminator_y1x0_update_ops,
        'g_x2y_loss': generator_x2y_loss,
        'g_y2x_loss': generator_y2x_loss,
        'd_x1y0_loss': discriminator_x1y0_loss,
        'd_y1x0_loss': discriminator_y1x0_loss,
        'fake_x_images': fake_x_images,
        'fake_y_images': fake_y_images,
        'g_x2y_loss_summary': g_x2y_loss_summary,
        'g_y2x_loss_summary': g_y2x_loss_summary,
        'd_x1y0_loss_summary': d_x1y0_loss_summary,
        'd_y1x0_loss_summary': d_y1x0_loss_summary,
        'real_x_images_summary': real_x_images_summary,
        'real_y_images_summary': real_y_images_summary,
        'fake_x_images_summary': fake_x_images_summary,
        'fake_y_images_summary': fake_y_images_summary,
        'cycle_x_images_summary': cycle_x_images_summary,
        'cycle_y_images_summary': cycle_y_images_summary}
    return to_be_run_dict

  def _get_update_ops(self, 
                      optimizer, 
                      generator_x2y_loss, 
                      generator_y2x_loss, 
                      discriminator_x1y0_loss, 
                      discriminator_y1x0_loss):
    """Build the variable update operations for the Discriminators and 
    Generators.
    """
    var_list = tf.trainable_variables()
    generator_x2y_var_list = [v for v in var_list 
        if v.name.startswith('GeneratorX2Y')]
    generator_y2x_var_list = [v for v in var_list 
        if v.name.startswith('GeneratorY2X')]
    discriminator_x1y0_var_list = [v for v in var_list 
        if v.name.startswith('DiscriminatorX1Y0')]
    discriminator_y1x0_var_list = [v for v in var_list 
        if v.name.startswith('DiscriminatorY1X0')]

    generator_x2y_update_ops = optimizer.minimize(
        generator_x2y_loss, var_list=generator_x2y_var_list)
    generator_y2x_update_ops = optimizer.minimize(
        generator_y2x_loss, var_list=generator_y2x_var_list)
    discriminator_x1y0_update_ops = optimizer.minimize(
        discriminator_x1y0_loss, var_list=discriminator_x1y0_var_list)
    discriminator_y1x0_update_ops = optimizer.minimize(
        discriminator_y1x0_loss, var_list=discriminator_y1x0_var_list)
     
    return (generator_x2y_update_ops, generator_y2x_update_ops, 
        discriminator_x1y0_update_ops, discriminator_y1x0_update_ops)


class CycleGANInferencer(object):
  """Make inference using CycleGAN."""
  def __init__(self, prediction_model):
    """Constructor.

    Args:
      prediction_model: a CycleGANPredictionModel instance.
    """
    self._prediction_model = prediction_model

  def infer(self, domain_x_filenames, domain_y_filenames, dataset):
    """Adds inference related ops to the graph.

    Args:
      domain_x_filenames: list of strs, the list of image filenames.
      domain_y_filenames: list of strs, the list of image filenames.
      dataset: a CycleGANInferencerDataset instance.
    """
    tensor_dict = dataset.get_tensor_dict(
        domain_x_filenames, domain_y_filenames)

    def x2y2x(real_x_images):
      fake_y_images = self._prediction_model.predict_generator_output(
          real_x_images, 'GeneratorX2Y')
      cycle_x_images = self._prediction_model.predict_generator_output(
          fake_y_images, 'GeneratorY2X')
      return fake_y_images, cycle_x_images

    def y2x2y(real_y_images):
      fake_x_images = self._prediction_model.predict_generator_output(
          real_y_images, 'GeneratorY2X')
      cycle_y_images = self._prediction_model.predict_generator_output(
          fake_x_images, 'GeneratorX2Y')
      return fake_x_images, cycle_y_images

    fake_images, cycle_images = tf.cond(
        tf.equal(tensor_dict['domain'], tf.constant('x')), 
        lambda: x2y2x(tensor_dict['real_images']), 
        lambda: y2x2y(tensor_dict['real_images']))

    return (tensor_dict['filename'], tensor_dict['real_images'], 
        fake_images, cycle_images)
