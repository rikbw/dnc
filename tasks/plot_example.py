import tensorflow as tf
from tasks import repeat_sequence
import matplotlib.pyplot as plt
import numpy as np

with tf.Session() as sess:
  examples = sess.run(repeat_sequence.RepeatSequence(5, 5, 7, 4, 2)())

for i in range(0, 2):
  fig = plt.figure()
  fig.add_subplot(1, 2, 1)
  plt.xlabel('Time')
  plt.ylabel('Vector index')
  plt.yticks([0,2,4,6,8])
  plt.imshow(np.transpose(examples.observations[i]), cmap='gray', vmin=0, vmax=1)
  fig.add_subplot(1, 2, 2)
  plt.xlabel('Time')
  plt.ylabel('Vector index')
  plt.yticks([0,2,4,6,8])
  plt.imshow(np.transpose(examples.target[i]), cmap='gray', vmin=0, vmax=1)
  # plt.show()
  plt.savefig('input_output_example' + str(i+1) + '.png')
