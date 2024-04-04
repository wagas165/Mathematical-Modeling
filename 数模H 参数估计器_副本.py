import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# 定义Lorenz系统的微分方程
def lorenz(t, x, p):
    sigma, rho, beta = p
    dxdt = [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
    return dxdt

# 定义损失函数
def loss(p, t, x0, x_obs):
    sol = solve_ivp(lambda t, x: lorenz(t, x, p), [t[0], t[-1]], x0, t_eval=t)
    return np.mean((sol.y[0,:] - x_obs[0,:])**2 + (sol.y[1,:] - x_obs[1,:])**2 + (sol.y[2,:] - x_obs[2,:])**2)

# 生成观测数据
t = np.linspace(0, 20, 2000)
x0 = [0, 1, 1.05]
p_true = [10, 28, 8/3]
x_obs = solve_ivp(lambda t, x: lorenz(t, x, p_true), [t[0], t[-1]], x0, t_eval=t).y

# 使用最小化算法进行参数估计
res = minimize(loss, [8, 20, 1], args=(t, x0, x_obs),method='Nelder-Mead')
res2= minimize(loss, [8, 20, 1], args=(t, x0, x_obs),method='CG')
res3= minimize(loss, [8, 20, 1], args=(t, x0, x_obs),method='BFGS')
# 输出估计的参数
print(res.x)
print(res2.x)
print(res3.x)

