# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dnc import dnc
from tasks import repeat_sequence

import numpy as np
np.set_printoptions(threshold=np.inf)

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 96, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 1, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-2, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 15, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer(
    "max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 1500,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 1000,
                        "Checkpointing step interval.")
tf.flags.DEFINE_string("summary_dir", "./summaries", "Summary directoyu")

# Writing error to file
tf.flags.DEFINE_string("error_file_name", "error_file_default", "The file where the error will be written")
tf.flags.DEFINE_integer("error_interval", 10, "The interval over which the error is averaged")

# Outputting tensorboard images
tf.flags.DEFINE_boolean("output_images", False, "Whether or not we output images for tensorboard")

def run_model(input_sequence, output_size, return_weights, time_major=False):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value, return_weights=return_weights)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=time_major,
      initial_state=initial_state
  )

  return output_sequence


def train(num_training_iterations, report_interval):
  """Trains the DNC and periodically reports the loss."""

  dataset = repeat_sequence.RepeatSequence(5, 5, 7, 4, FLAGS.batch_size)

  dataset_tensors = dataset()

  output_concat = run_model(dataset_tensors.observations, dataset.target_size, FLAGS.output_images, time_major=dataset.time_major())

  rom_weighting_size = 12

  output_logits = output_concat[:, :, 0:dataset.target_size]
  if FLAGS.output_images:
    output_read_weightings = output_concat[:, :, dataset.target_size:(dataset.target_size+FLAGS.memory_size)]
    output_write_weightings = output_concat[:, :, (dataset.target_size+FLAGS.memory_size):(dataset.target_size+2*FLAGS.memory_size)]
    output_mu = output_concat[:, :, (dataset.target_size+2*FLAGS.memory_size):(dataset.target_size+2*FLAGS.memory_size+1)]
    output_rom_weight = output_concat[:, :, (dataset.target_size+2*FLAGS.memory_size+1):(dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size)]
    output_rom_mode = output_concat[:, :, (dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size):(dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2)]
    output_read_mode = output_concat[:, :, (dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2):(dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2+3)]
    output_rom_key = output_concat[:, :, (dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2+3):(dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2+3+2)]
    output_original_read_weights = output_concat[:, :, (dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2+3+2):(dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2+3+2+FLAGS.memory_size)]
    output_forward_weights = output_concat[:, :, (dataset.target_size+2*FLAGS.memory_size+1+rom_weighting_size+2+3+2+FLAGS.memory_size):]

    # Rescaling first adds a row of ones so that max is always 255 in the rescaling
    output_read_weightings = get_concat_with_ones(output_read_weightings)
    output_write_weightings = get_concat_with_ones(output_write_weightings)
    output_mu = get_concat_with_ones(output_mu)
    output_rom_weight = get_concat_with_ones(output_rom_weight)
    output_rom_mode = get_concat_with_ones(output_rom_mode)
    output_read_mode = get_concat_with_ones(output_read_mode)
    output_rom_key = get_concat_with_ones(output_rom_key)
    output_original_read_weights = get_concat_with_ones((output_original_read_weights))
    output_forward_weights = get_concat_with_ones(output_forward_weights)

    tf.summary.image('Input', tf.expand_dims(dataset_tensors.observations, 3))
    tf.summary.image('Target', tf.expand_dims(dataset_tensors.target, 3))
    tf.summary.image('Output', tf.expand_dims(output_logits, 3))
    tf.summary.image('Read_weightings', tf.expand_dims(output_read_weightings, 3))
    tf.summary.image('Write_weightings', tf.expand_dims(output_write_weightings, 3))
    tf.summary.image('Mu', tf.expand_dims(output_mu, 3))
    tf.summary.image('Rom_weight', tf.expand_dims(output_rom_weight, 3))
    tf.summary.image('rom_mode', tf.expand_dims(output_rom_mode, 3))
    tf.summary.image('read_mode', tf.expand_dims(output_read_mode, 3))
    tf.summary.image('rom_key', tf.expand_dims(output_rom_key, 3))
    tf.summary.image('Non mixed read weights', tf.expand_dims(output_original_read_weights, 3))
    tf.summary.image('Forward read weights', tf.expand_dims(output_forward_weights, 3))

  # Used for visualization.
  output = tf.round(tf.sigmoid(output_logits))
  train_loss = dataset.cost(output_logits, dataset_tensors.target)
  train_error = dataset.error(output_logits, dataset_tensors.target)

  tf.summary.scalar('Loss', train_loss)
  merged = tf.summary.merge_all()

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  optimizer = tf.train.RMSPropOptimizer(
      FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step)

  saver = tf.train.Saver()

  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []

  error_file = open('error_files/' + FLAGS.error_file_name + '.csv', 'a')
  error_file.write('error\n')

  # Train.
  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    train_writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

    start_iteration = sess.run(global_step)
    total_loss = 0
    total_error = 0

    for train_iteration in range(start_iteration, num_training_iterations):
      _, loss, summary, error = sess.run([train_step, train_loss, merged, train_error])
      total_loss += loss
      total_error += error

      if (train_iteration + 1) % FLAGS.error_interval == 0:
        avg_error = total_error / FLAGS.error_interval
        error_file.write(str(avg_error) + "\n")
        total_error = 0

      if (train_iteration + 1) % report_interval == 0:
        dataset_tensors_np, output_np = sess.run([dataset_tensors, output])
        dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                   output_np)

        tf.logging.info("%d: Avg training loss %f.\n%s",
                        train_iteration, total_loss / report_interval,
                        dataset_string)
        total_loss = 0

        train_writer.add_summary(summary, train_iteration)

  error_file.close()

def to_batch_major(tensor):
  return tf.transpose(tensor, [1, 0, 2])

def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)

def get_concat_with_ones(tensor):
  shape = tf.shape(tensor)
  return tf.concat([tf.ones([shape[0], 1, shape[2]]), tensor], 1)

def my_tf_round(x, decimals=0):
  multiplier = tf.constant(10 ** decimals, dtype=x.dtype)
  return tf.round(x * multiplier) / multiplier

if __name__ == "__main__":
  tf.app.run()
