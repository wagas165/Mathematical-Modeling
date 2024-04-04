import numpy as np

buy_price=[13,14,16,19,24]
repair_price=[8,10,13,18,27]

class Machine():
    def __init__(self):
        self.price=0
        self.use_years=0
        self.years=0
        pass

    def buy(self):
        self.price+=buy_price[self.years]
        self.use_years=0
        self.years+=1

    def repair(self):
        self.price+=repair_price[self.use_years]
        self.use_years+=1
        self.years+=1


min_price = float('inf')  # 初始化为无穷大
best_list = []  # 初始化为空列表
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                for m in range(2):
                    a=[i, j, k, l, m]
                    M = Machine()
                    for num in a:
                        if num == 0:
                            M.repair()
                        else:
                            M.buy()
                    if M.price < min_price:
                        min_price = M.price  # 更新最小价格
                        best_list = a  # 更新最优方案

print(min_price, best_list)


