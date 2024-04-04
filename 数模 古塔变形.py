import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import fsolve

# 获得数据
df = pd.read_excel('/Users/zhangyichi/Desktop/古塔变形分析data.xls')
lx = np.array(df.iloc[98:106, 2], dtype=float)
lx = lx[~np.isnan(lx)]

ly = np.array(df.iloc[98:106, 3], dtype=float)
ly = ly[~np.isnan(ly)]

lz = np.array(df.iloc[98:106, 4], dtype=float)
lz = lz[~np.isnan(lz)]


# 构造设计矩阵
X = sm.add_constant(np.column_stack((lx, ly)))

# 构造OLS模型并进行拟合
model = sm.OLS(lz, X)
result = model.fit()

# 获取拟合结果
intercept, slope_x, slope_y = result.params
print(f"拟合平面方程：z = {slope_x:.4f}x + {slope_y:.4f}y + {intercept:.4f}")

def plane_func(params, z):
    a, b, c, d = params
    return a * z**3 + b * z**2 + c * z + d

def error_func(params, z, x, y):
    return np.sqrt((plane_func(params, z) - x)**2 + (plane_func(params, z) - y)**2)

known_x = lx  # 已知点的x坐标
known_y = ly  # 已知点的y坐标
known_z = lz  # 已知点的z坐标

# 获取第i层平面方程的系数
a = slope_x
b = slope_y
c = -1
d = intercept

def solve_z(params, z_prev, x_prev, y_prev):
    def equations(z):
        return error_func(params, z, x_prev, y_prev)

    z_sol = fsolve(equations, z_prev)
    return z_sol

# 缺失点的索引j
j = 4

# 获取缺失点的前后已知点坐标及z坐标
x_prev = known_x[j-1]
y_prev = known_y[j-1]
z_prev = known_z[j-1]

x_next = known_x[j+1]
y_next = known_y[j+1]
z_next = known_z[j+1]

# 求解缺失点的z坐标
z_missing = solve_z([a, b, c, d], z_prev, x_prev, y_prev)

# 计算缺失点的x坐标和y坐标
x_missing = plane_func([a, b, c, d], z_missing)
y_missing = plane_func([a, b, c, d], z_missing)
# 输出结果
print("缺失点的x坐标：", x_missing)
print("缺失点的y坐标：", y_missing)
print("缺失点的z坐标：", z_missing)

#代码说明：通过numpy的column_stack函数和statsmodels的add_constant函数创建一个设计矩阵X，这个矩阵将用于最小二乘法模型。
#利用statsmodels的OLS函数创建一个最小二乘法模型，并用fit函数拟合这个模型。然后从模型结果中提取拟合平面的参数。
#定义一个描述3D平面的函数和一个计算误差的函数。
#获取已知点的x、y、z坐标。
#根据最小二乘法模型的结果，获取平面的系数。
#定义一个求解函数，它使用fsolve来求解非线性方程，以获取缺失点的z坐标。
#获取缺失点的前后已知点的坐标及z坐标。
#使用求解函数来求解缺失点的z坐标。
#使用3D平面函数来计算缺失点的x和y坐标，并将结果打印出来。



