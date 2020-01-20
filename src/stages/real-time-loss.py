from time import sleep

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


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')
for i in range(len(x1)):
    plt.plot(x1[i], i, 'o-')
    plt.pause(0.001)  # Note this correction
    sleep(3)
plt.show()
# plt.draw()
#
# sleep(2)
