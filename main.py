#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sympy as sym

#
from gradient_descent_visualization import GradientDescent

#
gd = GradientDescent()
gd.fit()

#
#gd.plot2d()
gd.plot3d()
