import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_examples=100
x=np.linspace(10, 1000, 100)

def read_result_files(filename):
  res = []
  for i in range(1,11):
    content = np.squeeze(pd.read_csv('error_files/' + filename + str(i) + '.csv'))
    res.append(content)
  return np.array(res)

def map_results_with(results, f):
  median_array = []
  for i in range(0, num_examples):
    nums = results[:, i]
    median_array.append(f(nums))
  return median_array

def results(filename):
  file = read_result_files(filename)
  return (
    map_results_with(file, np.amin),
    map_results_with(file, np.median),
    map_results_with(file, np.amax)
  )

def plot_results(results, label, color):
  plt.plot(x, results[1], c=color, label=label, linewidth=4)
  plt.plot(x, results[0], c=color, linewidth=0.5)
  plt.plot(x, results[2], c=color, linewidth=0.5)

plot_results(results('error_rom_rounded'), color='g', label='With ROM')
plot_results(results('error_no_rom_rounded'), color='r', label='Without ROM')
plot_results(results('error_weird_rom_rounded'), color='b', label='With random ROM')

leg = plt.legend()
plt.ylabel('MSE')
plt.xlabel('Number of training iterations')
plt.xlim(left=10)
# plt.show()
plt.savefig('results_plot.png', transparent=True)
