def lagrange_interpolation(x, y, xi):
    """
    计算拉格朗日插值多项式在给定点 xi 处的值

    参数：
    x: x 坐标列表
    y: y 坐标列表
    xi: 给定点的 x 坐标

    返回值：
    插值多项式在给定点 xi 处的值
    """
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (xi - x[j]) / (x[i] - x[j])
        result += term
    return result

# 示例用法
x = [0.4,0.5, 0.6]
y = [-0.916291,-0.693147,-0.510826]
xi = 0.54
interpolated_value = lagrange_interpolation(x, y, xi)
print(f"The interpolated value at xi = {xi} is: {interpolated_value}")
