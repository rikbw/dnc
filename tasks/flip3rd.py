import sonnet as snt
import tensorflow as tf

import collections

DatasetTensors = collections.namedtuple('DatasetTensors', ('observations', 'target'))

# Flip the third vector of a sequence
class Flip3rd(snt.AbstractModule):

  def __init__(self, nb_vecs, vec_length, batch_size, every_third=False):
    super(Flip3rd, self).__init__(name='Flip3rd')
    self._nb_vecs = nb_vecs
    self._vec_length = vec_length
    self._batch_size = batch_size
    self._every_third = every_third
    self.target_size = vec_length

  def _build(self):
    obs_tensors = []
    target_tensors = []

    for batch_index in range(0, self._batch_size):
      obs_pattern = tf.cast(
        tf.random_uniform([self._nb_vecs, self._vec_length], minval=0, maxval=2, dtype=tf.int32),
        tf.float32
      )

      obs_padded = tf.concat([
        obs_pattern,
        tf.zeros([self._nb_vecs, self._vec_length], dtype=tf.float32)
      ], 0)

      target_padding = tf.zeros([self._nb_vecs, self._vec_length], dtype=tf.float32)

      # Flip 3rd bit
      obs_pattern_flipped = tf.concat([
        obs_pattern[0:2],
        tf.expand_dims(tf.ones(self._vec_length, dtype=tf.float32) - (obs_pattern[2]), 0),
        obs_pattern[3:]
      ], 0)

      target = tf.concat([target_padding, obs_pattern_flipped], 0)

      obs_tensors.append(obs_padded)
      target_tensors.append(target)

    return DatasetTensors(tf.convert_to_tensor(obs_tensors), tf.convert_to_tensor(target_tensors))

  def time_major(self):
    return False

  def cost(self, logits, targ):
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=targ, logits=logits)

    # Sum across the vectors
    loss_batch_time = tf.reduce_sum(xent, axis=2)

    # Sum away time
    loss_batch = tf.reduce_sum(loss_batch_time, axis=1)

    # Batch major
    batch_size = tf.cast(tf.shape(logits)[0], dtype=loss_batch_time.dtype)
    loss = tf.reduce_sum(loss_batch) / batch_size

    return loss

  def error(self, logits, targ):
    output = tf.round(tf.sigmoid(logits))
    error = tf.subtract(output, targ)
    error = tf.square(error)

    # Sum across the vectors
    error_batch_time = tf.reduce_sum(error, axis=2)

    # Sum away time
    error_batch = tf.reduce_sum(error_batch_time, axis=1)

    # Batch major
    batch_size = tf.cast(tf.shape(logits)[0], dtype=error.dtype)
    error = tf.reduce_sum(error_batch) / batch_size

    return error

  def to_human_readable(self, data, model_output=None, whole_batch=False):
    return bitstring_readable(data, self._batch_size, model_output, whole_batch)

def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
  """Produce a human readable representation of the sequences in data.

  Args:
    data: data to be visualised
    batch_size: size of batch
    model_output: optional model output tensor to visualize alongside data.
    whole_batch: whether to visualise the whole batch. Only the first sample
        will be visualized if False

  Returns:
    A string used to visualise the data batch. X axis is time, Y axis are vectors
  """

  def _readable(datum):
    return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

  # Transform input: shape is (batch_size, max_time, nb_bits)
  # This must become (batch_size, nb_bits, max_time)

  obs_batch = data.observations.transpose((0, 2, 1))
  targ_batch = data.target.transpose((0, 2, 1))

  iterate_over = range(batch_size) if whole_batch else range(1)

  batch_strings = []

  for batch_index in iterate_over:
    obs = obs_batch[batch_index, :, :]
    targ = targ_batch[batch_index, :, :]

    readable_obs = 'Observations:\n' + '\n'.join([_readable(obs_vector) for obs_vector in obs])
    readable_targ = 'Targets:\n' + '\n'.join([_readable(targ_vector) for targ_vector in targ])
    strings = [readable_obs, readable_targ]

    if model_output is not None:
      output = model_output.transpose((0, 2, 1))[batch_index, :, :]
      strings.append('Model Output:\n' + '\n'.join([_readable(output_vec) for output_vec in output]))

    batch_strings.append('\n\n'.join(strings))

  return '\n' + '\n\n\n\n'.join(batch_strings)