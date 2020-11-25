#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sympy as sym

#
class GraphicsTwoDim():

    def plot2d(self):

        plt.plot(self.x_local_min_list[0], self.y_local_min_list[0], marker='o', markersize=12, color='violet', alpha=0.8)

        plt.plot(self.x_local_min_list[-1], self.y_local_min_list[-1], marker='x', markersize=12, color='violet', alpha=0.8)

        plt.plot(self.x_local_min_list, self.y_local_min_list, 'k')

        plt.imshow(self.wave_equation(self.x, self.y), extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]], vmin=-5, vmax=5, cmap='magma')
        plt.xlabel('X-Axis')
        plt.ylabel('Y-Axis')
        plt.title('Gradient Descent - Local Minimum Path', fontweight="bold")
        plt.legend(['Random Start Coordinate','Local Mininum'])
        plt.colorbar()
        plt.show()

#
class GraphicsThreeDim():

    def plot3d(self):

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        x_mesh, y_mesh = np.meshgrid(self.x, self.y)
        ax.plot_surface(x_mesh, y_mesh, self.wave_equation(self.x, self.y), cmap='magma', alpha=0.6)

        sc1 = ax.scatter(self.x_local_min_list[0], self.y_local_min_list[0],
                self.wave_equation(self.x_local_min_list[0], self.y_local_min_list[0]), color='violet', marker='o', s=180, alpha=0.8)

        sc2 = ax.scatter(self.x_local_min_list[-1], self.y_local_min_list[-1],
                self.wave_equation(float(self.x_local_min_list[-1]), float(self.y_local_min_list[-1])), color='violet', marker='x', s=180, alpha=0.8)

        z_path = [float(self.wave_equation(float(self.x_local_min_list[i]), float(self.y_local_min_list[i]))) for i in range(len(self.x_local_min_list))]

        ax.plot3D(self.x_local_min_list, self.y_local_min_list, z_path, 'k')
        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')
        ax.set_title('Gradient Descent - Local Minimum Path', fontweight="bold")
        ax.legend([sc1, sc2], ['Random Start Coordinate', 'Local Minimum'])
        plt.show()


#
class GradientDescent(GraphicsTwoDim, GraphicsThreeDim):

    def __init__(self, learning_rate = 0.1, epochs = 100):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.x_local_min_list = []
        self.y_local_min_list = []
        self.x = np.linspace(-3, 3, 100)
        self.y = np.linspace(-3, 3, 100)
        

    def wave_equation(self, x, y):

        x, y = np.meshgrid(x,y)

        z = 3 * (1-x)**2 * np.exp(-(x**2) - (y+1)**2) \
        - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2-y**2) \
        - 1/3 * np.exp(-(x+1)**2 - y**2)

        return z


    def derivative(self):

        sx, sy = sym.symbols('sx, sy')

        sZ = 3 * (1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
        - 10 * (sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
        - 1/3 * sym.exp(-(sx+1)**2 - sy**2)

        df_x = sym.lambdify((sx, sy), sym.diff(sZ, sx), 'sympy')
        df_y = sym.lambdify((sx, sy), sym.diff(sZ, sy), 'sympy')

        return df_x, df_y


    def fit(self):

        x = np.linspace(-2, 2, 100)

        local_min_x = np.random.choice(x)
        local_min_y = np.random.choice(x)

        self.x_local_min_list.append(local_min_x)
        self.y_local_min_list.append(local_min_y)

        df_x, df_y = self.derivative()

        for _ in range(self.epochs):
            #print('test')
    
            x_deriv = df_x(local_min_x, local_min_y).evalf()
            y_deriv = df_y(local_min_x, local_min_y).evalf()
            
            local_min_x -= x_deriv * self.learning_rate
            local_min_y -= y_deriv * self.learning_rate
            
            self.x_local_min_list.append(local_min_x)
            self.y_local_min_list.append(local_min_y)
            #print(x_local_min_list)

        return self.x_local_min_list, self.y_local_min_list





gd = GradientDescent()
gd.fit()
gd.plot3d()








