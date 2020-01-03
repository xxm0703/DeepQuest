import time

import matplotlib.pyplot as plt
import numpy as np

#
#
# i = 0
# x = list()
# y = list()
#
# while i < 100:
#     temp_y = np.random.random()
#     x.append(i)
#     y.append(temp_y)
#     plt.plot(i, i, '.-')
#     i += 1
#     plt.pause(0.0001)  # Note this correction
# plt.show()


x1 = np.linspace(0.0, 10.0)
x2 = np.linspace(0.0, 2.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
# plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
plt.pause(0.0001)  # Note this correction

time.sleep(2)
