import time, board, busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import cv2
import matplotlib as mpl


# fig = plt.figure(figsize=(8, 6))  # start figure
# fig.show()
# fig.canvas.manager.set_window_title('Hệ thống giám sát nhiệt độ ra/vào')
# fig.canvas.toolbar_visible = False
# ax = fig.add_subplot(1, 2, 1)  # add subplot
# ax2 = fig.add_subplot(1, 2, 2)
# fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of unnecessary padding

x = np.arange(0, 25, 0.1)

fig, axis = plt.subplots(1,2, figsize=(15,5))

plt.ylabel('sin(x)')
plt.xlabel('x')
axis[0].plot(np.sin(x))
axis[1].plot(np.cos(x))
