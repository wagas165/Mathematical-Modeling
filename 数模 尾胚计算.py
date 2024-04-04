import pulp

prob = pulp.LpProblem("Cutting Problem", pulp.LpMinimize)

length=62.7
# define variables
x=pulp.LpVariable.dicts('x',range(6),lowBound=9,upBound=10)
y=pulp.LpVariable.dicts('y',range(0),lowBound=10)
z=pulp.LpVariable.dicts('z',range(1),upBound=9)

# define the objective function
prob += (pulp.lpSum(y)+pulp.lpSum(z))

# constraint for total length of tails being equal to the total steel length
prob += (pulp.lpSum(x)+pulp.lpSum(y)+pulp.lpSum(z))==length

# solve the problem
prob.solve()

print(f"Maximum tail length = {pulp.value(prob.objective)}")


