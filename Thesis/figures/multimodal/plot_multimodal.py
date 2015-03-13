#! /usr/bin/python
from math import exp
import numpy as np
import pypmc
import matplotlib.pyplot as plt
from multimodal import LogTarget

# need only two dimension to plot the essential part
log_target = LogTarget(2)
target = lambda x: exp(log_target(x))

# define the grid to plot the banana on
N_x_points = 2000
N_y_points = 2000
x_begin = -6.
x_end = 6.
y_begin = -6.
y_end = 6.

x_delta = (x_end - x_begin) / N_x_points
y_delta = (y_end - y_begin) / N_y_points
grid = np.mgrid[x_begin : x_end:x_delta, y_begin:y_end:y_delta]

# evaluate the target all over the grid
values = np.empty(list(grid.shape)[1:])
assert  values.shape == (grid.shape[1], grid.shape[2])
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        values[i,j] = target(grid[:,i,j])

# plot
plt.figure()
plt.contourf(grid[0], grid[1], values, cmap='gray_r')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('multimodal.svg')
