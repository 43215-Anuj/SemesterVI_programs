#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:31:37 2020

@author: baljeetkaur
"""

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

def func(x, y):
    return (x ** 2 + y ** 2)

x = np.linspace(-12, 12, 300)
y = np.linspace(-12, 12, 300)

X, Y = np.meshgrid(x, y)
Z = func(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

fig,ax=plt.subplots(1,1)
cp = ax.contour(X, Y, Z)

ax.set_title(' Contour Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

