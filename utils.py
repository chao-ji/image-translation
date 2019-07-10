import random
import tensorflow as tf 


def get_learning_rate(init_learning_rate, num_epochs):
  """Get the learning rate. Learning rate stays at `init_learning_rate` for 
  the first half of `num_epochs`, and linearly decays to zero in the second 
  half.

  Args:
    init_learning_rate: float scalar, initial learning rate.
    num_epochs: int scalar, num of epochs.

  Returns:
    learning_rate: float scalar tensor, learning rate.
    global_step: int scalar tensor, a mutable variable holding the epoch index.
  """
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.train.polynomial_decay(
      init_learning_rate, 
      global_step - num_epochs // 2, num_epochs // 2, 
      end_learning_rate=0, 
      power=1)

  learning_rate = tf.cond(global_step <= num_epochs // 2, 
      lambda: init_learning_rate, 
      lambda: learning_rate)

  return learning_rate, global_step


def cycle_consistency_loss(real_images, fake_images):
  """Computes cycle consistency loss -- L1 norm of `real_images - fake_images`.

  Args:
    real_images: float tensor of shape [batch_size, height, width, 3].
    fake_images: float tensor of shape [batch_size, height, width, 3].

  Returns:
    float scalar tensor: the cycle consistency loss.
  """
  return tf.reduce_mean(tf.abs(real_images - fake_images)) 


def least_square_generator_loss(logits):
  """Computes the generator loss (squared difference).

  Args:
    logits: float tensor of shape [batch_size, height, width, 1].

  Returns:
    float scalar tensor: the generator loss.
  """
  return tf.reduce_mean((logits - 1) ** 2)

def least_square_discriminator_loss(real_image_logits, fake_image_logits):
  """Computes the discriminator loss (squared difference).
  Scale it by 0.5 to make the discriminators train slower than generators. 

  Args:
    real_image_logits: float tensor of shape [batch_size, height, width, 1].
    fake_image_logits: float tensor of shape [batch_size, height, width, 1].
  """
  return (tf.reduce_mean((real_image_logits - 1) ** 2) + 
      tf.reduce_mean((fake_image_logits - 0) ** 2)) * 0.5


class FakeImagePool(object):
  """Keep track of a pool of history fake images, and provies API to sample 
  from this pool.
  """
  def __init__(self, pool_size=50):
    """Constructor.

    Args:
      pool_size: int scalar, size of pool.
    """
    self._pool_size = pool_size
    self._images = []

  def get(self, image):
    """Returns the input image, or sample from the history.

    Args:
      image: a numpy array of shape [batch_size, height, width, depth].

    Returns:
      the original image or the sampled image.
    """
    if self._pool_size == 0:
      return image

    if len(self._images) < self._pool_size:
      self._images.append(image)
      return image
    else:
      p = random.random()

      if p >= 0.5:
        rand_index = random.randint(0, self._pool_size - 1) 
        sampled = self._images[rand_index]
        self._images[rand_index] = image
        return sampled
      else:
        return image      
