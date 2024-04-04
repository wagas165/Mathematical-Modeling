import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# 定义Lorentz系统的函数
def lorentz(t, y, theta):
    sigma, rho, beta = theta
    return [sigma * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - beta * y[2]]


# 定义目标函数，也就是我们想要最小化的损失函数
def objective(theta, t, y_obs):
    sol = solve_ivp(lorentz, [t[0], t[-1]], y_obs[:, 0], args=(theta,), t_eval=t)
    return np.mean((y_obs - sol.y) ** 2)


# 时间范围
t = np.linspace(0, 1, 100)

# 初始猜测的参数值
theta_init = [5, 20, 1]

# 洛伦兹系统的真实参数值
real_thetas = [[10, 28, 8 / 3], [15, 36, 10 / 3], [20, 45, 12 / 3]]

for real_theta in real_thetas:
    # 储存每一步优化过程中的损失函数值和估计参数
    objective_history = []
    params_history = []


    def callback(x):
        params_history.append(x)
        objective_history.append(objective(x, t, y_obs))


    # 使用真实的参数值生成模拟数据
    sol = solve_ivp(lorentz, [t[0], t[-1]], [1, 1, 1], args=(real_theta,), t_eval=t)
    y_obs = sol.y + np.random.randn(3, 100) * 0.1  # 加入一些噪声

    # 使用scipy的minimize函数来求解优化问题
    res = minimize(objective, theta_init, args=(t, y_obs), method='BFGS', callback=callback)

    # 打印出真实的参数值和估计的参数值
    print('Real parameters: ', real_theta)
    print('Estimated parameters: ', res.x)
    print('----------------------')

    # 绘制误差趋势图
    plt.figure(figsize=(8, 6))
    plt.plot(objective_history)
    plt.title(f"Prediction Error Trend for Real Parameters {real_theta}")
    plt.xlabel("Iteration")
    plt.ylabel("Prediction Error")
    plt.savefig(f'/Users/zhangyichi/Desktop/lorentz/{time.time()}.png')

    # 从参数历史记录中提取σ, ρ, β
    sigma_values, rho_values, beta_values = zip(*params_history)

    # 创建3D散点图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制估计参数的散点
    ax.scatter(sigma_values, rho_values, beta_values, color='blue', label='Estimated Parameters')

    # 绘制真实参数的散点
    ax.scatter(*real_theta, color='red', label='Real Parameters')

    ax.set_xlabel("σ")
    ax.set_ylabel("ρ")
    ax.set_zlabel("β")
    ax.legend()
    ax.set_title(f"Parameter Estimation for Real Parameters {real_theta}")
    plt.savefig(f'/Users/zhangyichi/Desktop/lorentz/{time.time()}.png')




