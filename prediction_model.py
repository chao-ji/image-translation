import functools
import tensorflow as tf

slim = tf.contrib.slim


class CycleGANPredictionModel(object):
  """Builds the Discriminators (X1Y0, Y1X0) and Generators (X2Y, Y2X) for 
  CycleGAN. 
  """
  def __init__(self, 
               ngf=32, 
               ndf=64, 
               leaky_relu_alpha=0.2, 
               instance_norm_epsilon=1e-5, 
               weights_init_stddev=0.02):
    """Constructor.

    Args:
      ngf: int scalar, num of output channels of the first Conv2D in generator.
        Defaults to 32.
      ndf: int scalar, num of output channels of the first Conv2D in 
        discriminator. Defaults to 64.
      leaky_relu_alpha: float scalar, slope of leaky relu.
      instance_norm_epsilon: float scalar, epsilon of instance norm.
      weights_init_stddev: float scalar, standard deviation of truncated normal
        initializer.
    """
    self._ngf = 32
    self._ndf = 64
    self._leaky_relu_alpha = leaky_relu_alpha
    self._instance_norm_epsilon = instance_norm_epsilon
    self._weights_init_stddev = weights_init_stddev

  def predict_generator_output(self, images, scope='Generator'):
    """Builds the Generaotrs (X2Y, Y2X). 

    NOTE: the variables will be reused if being called the second time with the 
      same `scope`.

    Args:
      images: 4-D tensor of shape [batch_size, height, width, depth], input 
        real image for domain X or Y.
      scope: str scalar, name of the variable scope for generator.
    """
    images = tf.pad(images, [[0, 0], [3, 3], [3, 3], [0, 0]], 'constant')
    with tf.variable_scope(scope, 'Generator', [images], reuse=tf.AUTO_REUSE):

      with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
          weights_initializer=tf.truncated_normal_initializer(
              stddev=self._weights_init_stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.instance_norm,
          normalizer_params={'epsilon': self._instance_norm_epsilon}):

        # one `c7s1-k` layer
        conv = slim.conv2d(images,
                           num_outputs=self._ngf,
                           kernel_size=7,
                           stride=1,
                           padding='VALID')

        # two `dk` layers
        for i in range(2):
          conv = slim.conv2d(conv, 
            num_outputs=self._ngf * 2 if i == 0 else self._ngf * 4,
            kernel_size=3,
            stride=2,
            padding='SAME')

        for i in range(9):
          conv = self._resnet_unit(conv, self._ngf * 4, 'ResUnit%d' % i)

        tconv = conv
        for i in range(2):
          tconv = slim.conv2d_transpose(tconv,
              num_outputs=self._ngf if i == 1 else self._ngf * 2,
              kernel_size=3,
              stride=2,
              padding='SAME')

        fake_images = slim.conv2d(tconv,
                                  num_outputs=3,
                                  kernel_size=7,
                                  stride=1,
                                  padding='SAME',
                                  activation_fn=tf.nn.tanh,                                       
                                  normalizer_fn=None,                                                  
                                  normalizer_params=None)                                              
    return fake_images

  def predict_discriminator_output(self, inputs, scope='Discriminator'):
    """Builds the Discriminator (X1Y0 or Y1X0). 

    NOTE: the variables will be reused if being called the second time with the
      same `scope`.

    Args:
      inputs: 4-D tensor of shape [batch_size, height, width, depth], input 
        real image for domain X or Y.
      scope: str scalar, name of the variable scope for discriminator.
    """
    leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    with tf.variable_scope(
        scope, 'Discriminator', [inputs], reuse=tf.AUTO_REUSE):
      with slim.arg_scope([slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(
              stddev=self._weights_init_stddev),
          activation_fn=leaky_relu):

        for i in range(4):
          inputs = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], 'constant')
          conv = slim.conv2d(
              inputs,
              num_outputs=self._ndf * 2 ** i,
              kernel_size=4,
              stride=2 if i < 3 else 1,
              padding='VALID',
              normalizer_fn=None if i == 0 else slim.instance_norm,
              normalizer_params=None if i == 0 else 
                  {'epsilon': self._instance_norm_epsilon})
          inputs = conv

        inputs = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], 'constant')
        logits = slim.conv2d(
            inputs,
            num_outputs=1,
            kernel_size=4,
            stride=1,
            padding='VALID',
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params=None)
      return logits

  def _resnet_unit(self, inputs, num_outputs, name):
    """Builds the residual connection unit.

    Args:
      inputs: input feature map of shape [batch_size, height, width, depth].
      num_outputs: int scalar, num of output channels.
      name: str scalar, scope name.
    """
    with tf.variable_scope(name, values=[inputs]):
      conv = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
      conv = slim.conv2d(
          conv,
          num_outputs=num_outputs,
          kernel_size=3,
          stride=1,
          padding='VALID')
      conv = tf.pad(conv, [[0, 0], [1, 1], [1, 1], [0, 0]], 'constant')
      conv = slim.conv2d(conv,
                         num_outputs=num_outputs,
                         kernel_size=3,
                         stride=1,
                         padding='VALID',
                         activation_fn=None)
      return tf.nn.relu(conv + inputs)      
