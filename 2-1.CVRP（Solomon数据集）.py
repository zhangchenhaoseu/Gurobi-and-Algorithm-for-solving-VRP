# 靡不有初，鲜克有终
# 开发时间：2023/10/8 9:49
'''CVRP without time window, there is one depot and a single type of vehicles with the same capacity '''
'''data type: R101 or C101 from solomon dataset'''
import numpy as np
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt
import random
import time
import re
from itertools import product


'''定义数据类。建立用于存放和调用输入data的数据结构框架'''
class Data:
    def __init__(self):  # 建立类属性
        self.customerNum = 0  # 客户点数量
        self.nodeNum = 0  # 节点总数量（客户点、起点车场、终点车场）
        self.vehicleNum = 0  # 车辆数量
        self.capacity = 0  # 车辆容量
        self.nodeId = []  # 节点的id，不含虚拟结点，从0开始
        self.customerId = []  # 客户点的id
        self.vehicleId = None  # 车辆的id
        self.corX = []  # 节点横坐标
        self.corY = []  # 节点纵坐标
        self.demand = []  # 节点需求
        self.readyTime = []  # 节点时间窗的最早时间
        self.dueTime = []  # 节点时间窗的最晚时间
        self.serviceTime = []  # 节点服务时长
        self.distanceMatrix = None  # 节点与结点之间的距离


'''读取数据。将对应的输入数据存放至对应的数据结构框架中。函数参数包括数据路径、客户点数量（客户点在1-100之间）、车辆数量、车辆容量'''
def readData(path, customerNum, vehicleNum, capacity):
    data = Data()
    data.customerNum = customerNum
    data.vehicleNum = vehicleNum
    data.vehicleId = [i for i in range(0,vehicleNum)]
    data.capacity = capacity
    data_df = pd.read_csv(path)
    # 将1个起始车场（文本数据中的第一个）+customerNum个客户点的信息存放在对应数据结构中
    for i in range(0, data.customerNum+1):
        data.nodeId.append(data_df.loc[i, 'CUST NO']-1)  # 从0开始的所有实节点,不含虚拟结点
        data.corX.append(data_df.loc[i, 'XCOORD'])
        data.corY.append(data_df.loc[i, 'YCOORD'])
        data.demand.append(data_df.loc[i, 'DEMAND'])
        data.readyTime.append(data_df.loc[i, 'READY TIME'])
        data.dueTime.append(data_df.loc[i, 'DUE TIME'])
        data.serviceTime.append(data_df.loc[i, 'SERVICE TIME'])
    # 再增加一个虚拟终点车场，并添加对应的信息
    data.corX.append(data_df.loc[0, 'XCOORD'])
    data.corY.append(data_df.loc[0, 'YCOORD'])
    data.demand.append(data_df.loc[0, 'DEMAND'])
    data.readyTime.append(data_df.loc[0, 'READY TIME'])
    data.dueTime.append(data_df.loc[0, 'DUE TIME'])
    data.serviceTime.append(data_df.loc[0, 'SERVICE TIME'])
    data.customerId = data.nodeId.copy()
    data.customerId.remove(0)
    # 节点总数为：1个起点车场+customerNum个客户点+1个终点车场
    data.nodeNum = customerNum + 2
    # 填补距离矩阵
    data. distanceMatrix = np.zeros((data.nodeNum, data.nodeNum))
    for i in range(0, data.nodeNum):
        for j in range(0, data.nodeNum):
            if i != j:
                data.distanceMatrix[i][j] = ((data.corX[i]-data.corX[j])**2+(data.corY[i]-data.corY[j])**2)**0.5
            else:
                pass
    # print("distanceMatrix:")
    # print(data.distanceMatrix)
    return data


'''定义解类。建立求解结果的数据结构框架，并建立将Gurobi求解结果存放在数据结构中的连接'''
class Solution:
    ObjVal = 0  # 目标函数值
    X = None  # 决策变量X_ijk
    routes = None  # 存放路由所经过的节点序列
    routeNum = 0  # 路由的数量

    def __init__(self, data, model):  # 建立类属性
        self.ObjVal = model.ObjVal
        self.X = [[[0 for k in range(0, data.vehicleNum)] for j in range(0, data.nodeNum)] for i in range(0, data.nodeNum)]
        self.routes = []


def getSolution(data, model):  # 定义类方法，从Gurobi输出的模型中获得目标函数值、决策变量、继而得到路由的节点序列
    solution = Solution(data, model)
    # 在三维矩阵结构中存储自变量的取值，拆字段，检验是否是x，然后保存
    for v in model.getVars():
        split_arr = re.split(r'[,\[\]]', v.VarName)  # 将gurobi形式的变量进行拆解，便于利用数据结构实现实现存储
        if split_arr[0] == 'x' and v.x!=0 :
            # print(v)
            solution.X[int(split_arr[1])][int(split_arr[2])][int(split_arr[3])] = v.x  # X_ijk
        elif split_arr[0] == 'u':
            pass
            # print(v)
    # print("solution.X:",solution.X)
    # 在二维矩阵结构中存储车辆访问的结点序列
    for k in range(0, data.vehicleNum):
        i = 0  # 起点车场
        route_of_k = [i]  # 第k辆车路由的节点序列，起点从i=0开始
        loop_state = True
        while loop_state:
            for j in range(0, data.nodeNum):  # 含虚拟结点
                if solution.X[i][j][k] != 0:  # 对于第k辆车，若从节点i到j的变量不为0，说明k车在i选择前往j，则需要将j节点的id计入路由序列
                    route_of_k.append(j)
                    # print('route_of_k:',route_of_k)
                    i = j
                    if j == data.nodeNum - 1:  # j已经来到了虚拟结点
                        loop_state = False  # 遍历完即结束第k辆车当前的路由循环
        print('车辆行驶的路线:', route_of_k)
        if len(route_of_k) >= 3:  # route_of_k中只有两个点的情况，其实就是车辆停在在车场没有出库，所以要排除
            route_of_k[-1] = 0  # 修正，将倒数第一个虚拟结点的id改为0
            # print('最终route_of_k:', route_of_k)
            solution.routes.append(route_of_k)  # 将第k辆车的路由序列添入routes
            solution.routeNum += 1
            # print('最终solution.routes:', solution.routes)
    return solution


'''绘图。以图像展示CVRP求解结果'''
def plotSolution(data, solution):
    # 绘制画布
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{data.customerNum} Customers,{data.vehicleNum} Vehicles with the capacity of {data.capacity}")
    # 绘制节点
    plt.scatter(data.corX[0], data.corY[0], c='blue', alpha=1, marker=',', linewidths=2, label='depot')  # 起终点
    plt.scatter(data.corX[1:-1], data.corY[1:-1], c='black', alpha=1, marker='o', linewidths=1, label='customer')  # 客户点
    # 绘制节点的id
    for i in range(0,data.customerNum+1):
        x_ = data.corX[i]
        y_ = data.corY[i]
        label = data.nodeId[i]
        plt.text(x_, y_, str(label), family='serif', style='italic', fontsize=10, verticalalignment="bottom", ha='left', color='k')
    # 绘制路径
    for k in range(0,solution.routeNum):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        for i in range(0,len(solution.routes[k]) - 1):
            a = solution.routes[k][i]
            b = solution.routes[k][i + 1]
            x = [data.corX[a], data.corX[b]]
            y = [data.corY[a], data.corY[b]]
            plt.plot(x, y, color=(red/255, green/255, blue/255), linewidth=1)
    plt.grid(False)
    plt.legend(loc='best')
    plt.show()
    return 0


'''展示计算结果。以文字展示CVRP求解结果'''
def printSolution(data,solution):
    for index, route_of_k in enumerate(solution.routes):
        distance = 0
        load = 0
        for i in range(len(route_of_k) - 1):
            distance += data.distanceMatrix[route_of_k[i]][route_of_k[i + 1]]
            load += data.demand[route_of_k[i]]
        print(f"Route-{index + 1} : {route_of_k} , distance: {distance} , load: {load}")


'''建模和求解。使用Gurobi对问题进行建模'''
def modelingAndSolve(data):
    # 建立模型
    m = Model('CVRP')
    # 模型设置：由于存在函数printSolution，因此关闭输出;以及容许误差
    m.setParam('MIPGap', 0.05)
    # m.setParam('OutputFlag', 0)
    # 定义变量：
    # Step1.建立合适的数据结构建立变量的索引，CVRP仅有一个决策变量X,一个MTZ辅助变量U
    X_set = []
    k_set = data.vehicleId
    i_set = [i for i in range(0,data.nodeNum-1)]  # 不含虚拟终点车场的其他节点 ”0“+n
    j_set = [j for j in range(0,data.nodeNum)]  # 含虚拟终点车场的所有节点 ”0“+n+”n+1“
    for k in k_set:
        for i in i_set:
            for j in j_set:
                if i!=j:
                    X_set.append((i, j, k))
    U_set = []  # u集合
    for k in k_set:
        for i in range(0, data.nodeNum):
            U_set.append((i, k))

    MTZ_X_set = []  # 在MTZ约束中X的下标范围
    for k in k_set:
        for i in i_set:
            for j in range(1, data.nodeNum):
                if i != j:
                    MTZ_X_set.append((i, j, k))

    X_set_tplst = tuplelist(X_set)
    U_set_tplst = tuplelist(U_set)
    MTZ_X_set_tplst = tuplelist(MTZ_X_set)
    # Step2.根据索引，为模型建立变量
    x = m.addVars(X_set_tplst, vtype=GRB.BINARY, name='x')
    u = m.addVars(U_set_tplst, vtype=GRB.CONTINUOUS, lb=0.0,  name='u')  # 非负连续变量
    m.update()
    # 定义目标函数
    m.setObjective(quicksum(x[i, j, k] * data.distanceMatrix[i, j] for i, j, k in X_set_tplst), sense=GRB.MINIMIZE)

    # 定义约束条件:
    # 1.客户点服务一次约束
    m.addConstrs((quicksum(x[i, j, k] for i, j, k in X_set_tplst.select(I, '*', '*')) == 1 for I in data.customerId),'-')
    # 2.起点流出约束
    m.addConstrs((quicksum(x[i, j, k] for i, j, k in X_set_tplst.select(0, '*', K)) == 1 for K in data.vehicleId),'-')
    # 3.终点流入约束
    m.addConstrs((quicksum(x[i, j, k] for i, j, k in X_set_tplst.select('*', data.customerNum+1, K)) == 1 for K in data.vehicleId),'-')
    # 4.流平衡约束
    m.addConstrs((quicksum(x[i, h, k] for i, h, k in X_set_tplst.select('*',H, K))-quicksum(x[h, j, k] for h, j, k in X_set_tplst.select(H, '*', K)) == 0 for H,K in product(data.customerId, data.vehicleId)),'-')
    # 5.破子圈约束
    m.addConstrs((u[i, k] - u[j, k] + data.nodeNum * x[i, j, k] <= data.nodeNum - 1 for i, j, k in MTZ_X_set_tplst),'-')  # MTZ约束
    # 6. 容量约束
    m.addConstrs((quicksum(data.demand[i] * x[i, j, k] for i, j, k in X_set_tplst.select('*', '*', K)) <= data.capacity for K in data.vehicleId), '-')

    # 记录求解开始时间
    start_time = time.time()
    # 求解
    m.optimize()
    m.write('CVRP.lp')
    if m.status == GRB.OPTIMAL:
        print("-" * 20, "求解成功", '-' * 20)
        # 输出求解总用时
        print(f"求解时间: {time.time() - start_time} s")
        print(f"总行驶距离: {m.ObjVal}")
        solution = getSolution(data,m)
        # print(solution.X)
        # print(solution.routes)
        plotSolution(data,solution)
        printSolution(data,solution)
    else:
        print("无解")
    return m


'''主函数，调用函数实现问题的求解'''
if __name__ =="__main__":
    # 数据集路径
    data_path = r'C:\Users\张晨皓\Desktop\张晨皓的汇报内容\50.几种常见的VRP及Gurobi实现\程序代码\data\R101network.txt'  # 这里是节点文件
    customerNum = 15
    vehicleNum = 4
    capacity = 100
    # 读取数据
    data = readData(data_path, customerNum, vehicleNum, capacity)
    # 输出相关数据
    print("-" * 20, "Problem Information", '-' * 20)
    print(f'节点总数: {data.nodeNum}')
    print(f'客户点总数: {data.customerNum}')
    print(f'车辆总数: {data.vehicleNum}')
    print(f'车容量: {data.capacity}')
    # 求解
    modelingAndSolve(data)
