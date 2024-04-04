import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# 定义SIR模型的微分方程组
def SIR(y,t,N, k, h):
    S, I, R = y
    dSdt = -k * S * I
    dIdt = k * S * I  - h * I
    dRdt = h * I
    return dSdt, dIdt, dRdt


# 设置初始值和参数
I0, R0 = 0.9, 0
S0 = 1 - I0 - R0
k=2
h=0.4

# 定义时间轴
t = np.linspace(0, 30)

# 求解微分方程组
y0 = S0, I0, R0
sol = odeint(SIR, y0, t, args=(1,k,h))
S, I, R = sol.T

# 绘制图像
fig = plt.figure()
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of people')
plt.legend()
plt.show()
plt.savefig('SIR.png')