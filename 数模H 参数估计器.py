import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# 定义Lorentz系统的微分方程
def lorenz(t, x, p):
    sigma, rho, beta = p
    dxdt = [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
    return dxdt


# 定义损失函数
def loss(p, t, x0, x_obs):
    sol = solve_ivp(lambda t, x: lorenz(t, x, p), [t[0], t[-1]], x0, t_eval=t)
    return np.mean(
        (sol.y[0, :] - x_obs[0, :]) ** 2 + (sol.y[1, :] - x_obs[1, :]) ** 2 + (sol.y[2, :] - x_obs[2, :]) ** 2)


# 使用多种最小化算法进行参数估计
dic={0:['Nelder-Mead'],1:['CG'],2:['BFGS']} #分别存储Nelder_Mead，CG，BFGS的误差值
loop=100

for j in range(3):
    alog=dic[j]
    for i in range(loop):
        print(f'循环第{j+1}个算法第{i+1}次')
        # 生成观测数据
        t = np.linspace(0, 20, 2000)
        x0 = [0, 1, 1.05]
        p_true = [10, 28, 8 / 3]
        x_obs = solve_ivp(lambda t, x: lorenz(t, x, p_true), [t[0], t[-1]], x0, t_eval=t).y
        rand=[1,2,3]
        print(rand)
        res_N = minimize(loss, x0=rand, args=(t, x0, x_obs), method=str(alog[0]))
        print(res_N.x)
        alog.append(1 / 3 * np.sum(np.square(rand - np.array(res_N.x))))
