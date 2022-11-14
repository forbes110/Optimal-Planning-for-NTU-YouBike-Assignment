# In[] Import packages
from unittest import result
from gurobipy import*
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from pandas import ExcelWriter
import os
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# 請記得自己修改資料集路徑，有必要也可以修改Gurobi執行時間限制秒數
fontManager.addfont('./TaipeiSansTCBeta-Regular.ttf')
mpl.rc('font', family='Taipei Sans TC Beta')
data_path = 'spaceInfo_6.3_10.xlsx'
time_limit = 40
lambda_val = 0.3
cluster_start = 1 
"""
Contents:
    line 31~316 : Pre-defined: Gurobi & Obtaining objective values
    line 317~448: Pre-defined: Ratio Visualization
    line 449~487: 01. K-Means Clustering
    line 488~518: 02. Split matrix 
    line 519~534: 03. Gurobi solver
    line 535~696: 04. Map and routes visualization
    line 697~733: 05. Decide best k by objective values graphs
"""
# In[] Main program
def each_run(labels,cluster_i,A,S_H,S_L,L,lambda_value,van_cost,oil_cost):
    file_name = "./Kmeans_clustered_dist_matrix/Clustered_dist_matrix_labels"+str(labels)+".xlsx"
    with pd.ExcelFile(file_name, engine='openpyxl') as reader:
        cluster0 = pd.read_excel(reader, sheet_name='Cluster '+str(cluster_i),header=None)
    
    # 記錄站點名稱
    name_list = [0]
    for i in range(1,cluster0[0].size):
        name_list.append(int(cluster0[0][i]))
    name_list.append(0)

    # 準備矩陣（丟掉第一欄、第一列）
    cluster = cluster0.drop(columns=cluster0.columns[[0]], axis=1)
    cluster = cluster.drop(index=[0], axis=0)

    # 製作加入虛擬點的成本矩陣
    c = []
    I = cluster.columns.size
    print(I)
    c0 = [0] * (I+2)
    c.append(c0)
    for i in range(I):
        c.append([0])
        for j in range(I):
            c[i+1].append(cluster[i+1][j+1]) 
            # print(cluster[i+1][j+1])
        c[i+1].append(0)
    c.append(c0)

    # 記錄站點容量與已停數量
    data = pd.read_excel(data_path)
    data.drop(['Unnamed: 0'], axis = 1)

    # initialization
    station_id = [0]
    Parked = [0]
    Capacity = [0]
    # print("name_list",name_list)

    for i in range(1,len(name_list)-1):
        id = name_list[i]
        # print(id)
        station_id.append(id)
        Capacity.append(int(data["space_total"][id]))
        Parked.append(int(data["space_occupied"][id]))

    station_id.append(10000)
    Parked.append(0)
    Capacity.append(10000)

    # print("station_id",station_id)
    # print("Parked",Parked)
    # print("Capacity",Capacity)

    model = Model('OR Final Project')

    # I = 5
    # A = 16               # 一台調度車容量上限
    # S_H = 0.7            # 停車率的安全範圍上限比率
    # S_L = 0.3            # 停車率的安全範圍下限比率
    # L = 1               # 調度車在單位時間內移動的距離（或時間）上限
    # lambda_value = 500   #有改過 比較合理

    # 為站點 i 的車柱總數
    # Capacity = [0,200,100,150,20,100,10000]

    # 站點 i 的停車數量
    # Parked = [Capacity[i]*S_H if i!= 0 and i!=I+1 else 0 for i in range(I+2)] # 測試剛好最高
    # Parked = [Capacity[i]*S_L if i!= 0 and i!=I+1 else 0 for i in range(I+2)] # 測試剛好最低
    # Parked = [0,55,80,40,10,50,0]  # 自己key



    """ set cost """
    # 調度車輛從站點 i 到站點 j 的成本 (c_ij 要等於 c_ji)
    # 
    real_cost = 1
    virtual_cost = 0
    # oil_cost = 3.136
    # van_cost = 250
    # c = [[real_cost if i!=I+1 and i!=0 and i!=j else virtual_cost for i in range(I+2)] if j!=I+1 and j!=0 else [virtual_cost for k in range(I+2)]for j in range(I+2)]

    # 不用做事（如全部的車站都符合標準）的時候，讓虛擬起點到虛擬終點的成本是負的，迫使solver讓調度車不經過真實站點
    virtual_start_and_end = -(van_cost/oil_cost)
    c[0][0]     = 0
    c[0][I+1]   = virtual_start_and_end
    c[I+1][0]   = 0
    c[I+1][I+1] = 0

    # print("Cost:",*c, sep='\n') # print cost to check

    """ declare variables """
    a=[]    # 調度車從i站拿走的量
    b=[]    # 調度車放入i站的量
    p=[]    # 進i站時調度車上的腳踏車量
    q=[]    # 出i站時調度車上的腳踏車量
    x=[]    # =1代表有經過i站
    y=[]    # =1代表從i走到j
    w_L=[]  # 各點的懲罰比率
    w_H=[]  # 各點的懲罰比率
    u=[]    # 車站的經過「順序」
    v=[]    # =1代表u[i]>u[j]

    a =   model.addVars(I+2,      lb = 0, vtype = GRB.INTEGER, name = 'a')
    b =   model.addVars(I+2,      lb = 0, vtype = GRB.INTEGER, name = 'b')
    p =   model.addVars(I+2,      lb = 0, vtype = GRB.INTEGER, name = 'p')
    q =   model.addVars(I+2,      lb = 0, vtype = GRB.INTEGER, name = 'q')

    x =   model.addVars(I+2,      lb = 0, vtype = GRB.BINARY,  name = 'x')
    y =   model.addVars(I+2, I+2, lb = 0, vtype = GRB.BINARY,  name = 'y') 
    w_L = model.addVars(I+2,      lb = 0, vtype = GRB.CONTINUOUS, name = 'w_L')
    w_H = model.addVars(I+2,      lb = 0, vtype = GRB.CONTINUOUS, name = 'w_H')

    u =   model.addVars(I+2,      lb = 0, vtype = GRB.INTEGER, name = 'u')
    v =   model.addVars(I+2, I+2, lb = 0, vtype = GRB.BINARY , name = 'v') # 決定每個u的關聯

    """ set objective and constraints """
    # 虛擬點為0是起點,n+1為終點
    # 真正的點為1~n
    model.update()
    model.setObjective( oil_cost * quicksum(y[i,j]*c[i][j] for i in range(I+2) for j in range(I+2)) + lambda_value * (100 * quicksum(w_L[i]+w_H[i] for i in range(1,I+1))) ** 2 + van_cost, GRB.MINIMIZE)
        
    M=sum(Parked)
    """路線限制!!!"""

    # 兩個點都走到才有機會連在一起->有y必有x
    model.addConstrs((x[i] + x[j]  >= 2*(y[i, j]+y[j, i]) for i in range(I+2) for j in range(I+2)), "C1")

    # 對於起終點以外，進的點=出的點(也就是都只有一個點)
    model.addConstrs(   (quicksum(y[i, k] for i in range(0,I+1))==quicksum(y[k, j] for j in range(1,I+2)) for k in range(1,I+1)   ),"C2")

    # 不能到自己
    model.addConstrs(   (y[i, i]==0 for i in range(I+2))   ,"C3") # 這邊的寫法跟pdf上不一樣有小改ij to ii

    # 不能來回走
    model.addConstrs(   (y[i, j]+y[j, i] <= 1 for i in range(I+2) for j in range(I+2))   ,"C4")

    # 不能跑超過一小時能跑的距離(L)
    model.addConstr(    (quicksum(y[i, j]*c[i][j] for i in range(I+2) for j in range(I+2)) <= L)   ,"C5")

    # 一站只能進或出一次
    model.addConstrs(   (quicksum(y[i, j] for i in range(I+2))<=1 for j in range(I+2))   ,"C6-1")
    model.addConstrs(   (quicksum(y[i, j] for j in range(I+2))<=1 for i in range(I+2))   ,"C6-2")

    # x不為0時y要有東西
    # 這邊的寫法跟pdf上不一樣有小改範圍
    model.addConstrs(   ((quicksum(y[i, j] for j in range(I+2))+quicksum(y[j, i] for j in range(I+2))) >= x[i] for i in range(I+2))   ,"C7")


    """ 處理subtour """
    model.addConstr(  (u[0] == 1), "")
    model.addConstrs( (u[i] <= (I+2) for i in range(1,I+2)) , "")
    model.addConstrs( (u[i] >= 2 for i in range(1,I+2)) , "")
    model.addConstrs( (u[i] - u[j] + 1 <= (M)*(1-y[i, j]) for i in range(I+2) for j in range(I+2) if i != j) , "")

    # v_ij=1 if ui > uj else 0
    e = 0.01
    model.addConstrs( (u[i]-u[j] <= M*v[i, j]-e     for i in range(I+2) for j in range(I+2) if i != j),"每個u不一樣")
    model.addConstrs( (u[i]-u[j] >= e-M*(1-v[i, j]) for i in range(I+2) for j in range(I+2) if i != j),"每個u不一樣")



    """站點bike限制!!!"""
    # 只能拿或放，不能同時
    model.addConstrs(   (a[i]*b[i] == 0 for i in range(I+2)) , "C8")

    # 有經過才拿或放
    model.addConstrs( (M*x[i] >= a[i] for i in range(I+2)), "C9-1")
    model.addConstrs( (M*x[i] >= b[i] for i in range(I+2)), "C9-2")

    # 懲罰
    model.addConstrs((w_L[i] >= S_L - (Parked[i] - a[i] * x[i] + b[i] * x[i])/Capacity[i]    for i in range(1,I+1)), "C10-1-1")
    model.addConstrs((w_L[i] >= 0 for i in range(1,I)), "")

    model.addConstrs((w_H[i] >= (Parked[i] - a[i] * x[i] + b[i] * x[i])/Capacity[i] - S_H    for i in range(1,I+1)), "C10-2-1")
    model.addConstrs((w_H[i] >= 0 for i in range(1,I)), "")

    # 所有從各站拿走總合等於放回 卡車最後沒東西都放回虛擬點
    model.addConstr(  (quicksum(a[i]*x[i] for i in range(I+2)) == quicksum(b[i]*x[i] for i in range(I+2))), "C11")

    # 合理的放回車站
    model.addConstrs( (b[i] +  Parked[i] <= Capacity[i] for i in range(1,I+2)), "C12-1")
    # 合理地拿出車站
    model.addConstrs( (a[i] <= Parked[i]                for i in range(1,I+1)), "C12-2")


    """ trucks constraints """
    # 進+拿-放=出
    model.addConstrs( (p[i] + a[i] - b[i] == q[i] for i in range(I+2)), "C13")
    # 從此站出的腳踏車量等於到下站時的腳踏車量
    model.addConstrs( (q[i]*y[i,j] == p[j]*y[i,j] for i in range(I+2) for j in range(I+2) if i != j), "C14")
    # 車上總車數不能超過容量
    model.addConstrs( (p[i] <= A for i in range(I+2)), "C15-1")
    model.addConstrs( (q[i] <= A for i in range(I+2)), "C15-2")
    # 不能放下超過貨車上進場時的腳踏車量
    model.addConstrs( (b[i] <= p[i] for i in range(I+2)), "")


    """ Virtual points constraints """
    # 起點的a,b為0
    model.addConstr(  (a[0]   == 0), "")
    model.addConstr(  (b[0]   == 0), "")
    # 終點的a為0,b都可
    model.addConstr(  (a[I+1] == 0), "")
    model.addConstr(  (b[I+1] >= 0), "終點可以放車（真實的站點最後一站沒補完）")
    # 起終點必經
    model.addConstr(  (x[0]   == 1), "")
    model.addConstr(  (x[I+1] == 1), "")
    # 起點不進，終點不出
    model.addConstrs( (y[i,   0] == 0 for i in range(1, I+2))  , "不能進入虛擬起點")# 範圍小改
    model.addConstrs( (y[I+1, i] == 0 for i in range(I+1))     , "虛擬終點不能出去")   # 範圍小改
    # 在起點時沒車，終點時即便原本有車也要放到沒
    model.addConstr(  (p[0]   == 0), "")
    model.addConstr(  (q[0]   == 0), "")
    model.addConstr(  (p[I+1] >= 0), "進入終點時可以有車（真實的站點最後一站沒補完）")
    model.addConstr(  (q[I+1] == 0), "")
    # 起終點不懲罰
    model.addConstr(  (w_L[0]   == 0), "")
    model.addConstr(  (w_H[0]   == 0), "")
    model.addConstr(  (w_L[I+1] == 0), "")
    model.addConstr(  (w_H[I+1] == 0), "")

    model.Params.timeLimit = time_limit

    model.optimize()

    r_a = []
    r_b = []
    r_p = []
    r_q = []
    r_x = []
    r_w_L = []
    r_w_H = []
    r_u = []

    r_y = []
    r_v = []
    
    if model.Status != GRB.INFEASIBLE and model.Status != GRB.UNBOUNDED:
        try:
            for var in model.getVars():
                if var.varName[0] == "a": r_a.append(var.x)
                if var.varName[0] == "b": r_b.append(var.x)
                if var.varName[0] == "p": r_p.append(var.x)
                if var.varName[0] == "q": r_q.append(var.x)
                if var.varName[0] == "x": r_x.append(var.x)
                if var.varName[0] == "u": r_u.append(var.x)
                if var.varName[2] == "L": r_w_L.append(var.x)
                if var.varName[2] == "H": r_w_H.append(var.x)
                if var.varName[0] == "y": r_y.append(var.x)
                step = I+2
                new_r_y = [r_y[i:i+step] for i in range(0, len(r_y), step)]

            only_station_ID = name_list[1:-1]
            return model.objVal, r_a, r_b, r_p, r_q, r_x, r_w_H, r_w_L, r_u, new_r_y, only_station_ID, Parked, Capacity
        except:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    
def return_detail(labels, A, S_H, S_L, L, lambda_value,van_cost,oil_cost):
    detail = {}
    
    for i in range(labels):
        print("============ Cluster : %d ==============="%(labels))
        objVal, r_a, r_b, r_p, r_q, r_x, r_w_H, r_w_L, r_u, r_y, name_list, Parked, Capacity = each_run(labels,i,A,S_H,S_L,L,lambda_value,van_cost,oil_cost)
        if objVal == 0:
            detail[i] = "infeasible"
            continue
        detail[i] = {}
        detail[i]["ID"] = i
        detail[i]["name_list"] = name_list
        detail[i]["r_a"] = r_a
        detail[i]["r_b"] = r_b
        detail[i]["r_p"] = r_p
        detail[i]["r_q"] = r_q
        detail[i]["r_x"] = r_x
        detail[i]["r_w_H"] = r_w_H
        detail[i]["r_w_L"] = r_w_L
        detail[i]["r_u"] = r_u
        detail[i]["r_y"] = r_y
        detail[i]["objVal"] = objVal
        detail[i]["Parked"] = Parked
        detail[i]["Capacity"] = Capacity
    return detail

def draw_after_ratio(groups):
    print(groups)
    LABEL_COLOR_MAP = {0:'orange', 1:'green', 2:'cyan', 3:'pink', 4:'m', 5:'yellow', 6:'peru', 7:'lawngreen', 8:'deepskyblue', 9:'blueviolet'}    
    label_color = [LABEL_COLOR_MAP[l] for l in cluster]

    plt.rcParams["figure.figsize"] = (15,10)
    plt.scatter([X_hat[:,0]], [X_hat[:,1]], c=label_color, cmap='viridis')

    for i in range(groups):
        id_list = detail[groups][i]["name_list"]
        a_list = detail[groups][i]["r_a"]
        b_list = detail[groups][i]["r_b"]
        Parked = detail[groups][i]["Parked"]
        Capacity = detail[groups][i]["Capacity"]
        # print(Parked)
        # print(Capacity)
        # print(a_list)
        # print(b_list)
        for j in range(1,len(a_list)-1):
            # print((i,j))
            # print(id_list)
            id = id_list[j-1]
            total_now = Parked[j] - int(a_list[j]) + int(b_list[j])
            # print(space_occupied[id], int(a_list[j-1]+0.4) ,int(b_list[j-1]+0.4),total_now)
            if int(total_now) < int(Capacity[j]) * S_L:
                plt.annotate("%s ( %d / %d )"%("S"+str(id), int(total_now), int(Capacity[j])), (X_hat[id,0], X_hat[id,1]-0.00025), ha='center', color='red')
            elif int(total_now) > int(Capacity[j]) * S_H:
                plt.annotate("%s ( %d / %d )"%("S"+str(id), int(total_now), int(Capacity[j])), (X_hat[id,0], X_hat[id,1]-0.00025), ha='center', color='blue')
            else:
                plt.annotate("%s ( %d / %d )"%("S"+str(id), int(total_now), int(Capacity[j])), (X_hat[id,0], X_hat[id,1]-0.00025), ha='center', color='black')
            
            
    for i in range(groups):
        id_list = detail[groups][i]["name_list"]
        y_list_1 = detail[groups][i]["r_y"]
        items = len(y_list_1)
        for j in range(items):
            y_list = y_list_1[j]
            for k in range(items):
            # j = 0（從虛擬點出發）, y_jk = 1（從虛擬點到 k）, k 的 ID 是 id_list[k-1]
                if j == 0 and y_list[k] > 0:
                    if k == len(y_list) - 1:
                        continue

                if j == 0 and y_list[k] >=0.5: #  and k!= 0 and k != items-1
                    b_ID = id_list[k-1] 
                    plt.scatter(X_hat[b_ID,0],X_hat[b_ID,1], s=100, c=LABEL_COLOR_MAP[i], marker='P',edgecolors='black' , label='Startpoint for cluster '+str(i+1))

                # # y_ab = 1（從虛擬點到 k ）, a_ID = id_list[j-1], b_ID = id_list[k-1]
                elif y_list[k] >= 0.5 and k!= items-1:
                        a_ID = id_list[j-1]
                        b_ID = id_list[k-1]
                        xs = [X_hat[a_ID,0], X_hat[b_ID,0]]
                        ys = [X_hat[a_ID,1], X_hat[b_ID,1]]
                        plt.plot(xs, ys, c='black', linestyle = '-.')
                elif y_list[k] == 1 and k == len(y_list)-1:
                    continue

    plt.title("Detailed map of Ubike 2.0 stations after allocation when K = %d"%(groups))
    plt.legend()
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.savefig('4-%d-3_After Allocation Map.png'%(groups))
    plt.show()
    
def draw_before_ratio(groups):
    
    LABEL_COLOR_MAP = {0:'orange', 1:'green', 2:'cyan', 3:'pink', 4:'m', 5:'yellow', 6:'peru', 7:'lawngreen', 8:'deepskyblue', 9:'blueviolet'}    
    label_color = [LABEL_COLOR_MAP[l] for l in cluster]

    plt.rcParams["figure.figsize"] = (15,10)
    plt.scatter([X_hat[:,0]], [X_hat[:,1]], c=label_color, cmap='viridis')

    for i in range(groups):
        id_list = detail[groups][i]["name_list"]
        a_list = detail[groups][i]["r_a"]
        b_list = detail[groups][i]["r_b"]
        Parked = detail[groups][i]["Parked"]
        Capacity = detail[groups][i]["Capacity"]
        # print(Parked)
        # print(Capacity)
        # print(a_list)
        # print(b_list)
        for j in range(1,len(a_list)-1):
            # print((i,j))
            # print(id_list)
            id = id_list[j-1]
            # total_now = Parked[j] - int(a_list[j]) + int(b_list[j])
            # print(space_occupied[id], int(a_list[j-1]+0.4) ,int(b_list[j-1]+0.4),total_now)
            if int(Parked[j]+0.5) < int(Capacity[j]) * S_L:
                plt.annotate("%s ( %d / %d )"%("S"+str(id), int(Parked[j]+0.5), int(Capacity[j])), (X_hat[id,0], X_hat[id,1]-0.00025), ha='center', color='red')
            elif int(Parked[j]+0.5) > int(Capacity[j]) * S_H:
                plt.annotate("%s ( %d / %d )"%("S"+str(id), int(Parked[j]+0.5), int(Capacity[j])), (X_hat[id,0], X_hat[id,1]-0.00025), ha='center', color='blue')
            else:
                plt.annotate("%s ( %d / %d )"%("S"+str(id), int(Parked[j]+0.5), int(Capacity[j])), (X_hat[id,0], X_hat[id,1]-0.00025), ha='center', color='black')

    for i in range(groups):
        id_list = detail[groups][i]["name_list"]
        y_list_1 = detail[groups][i]["r_y"]
        items = len(y_list_1)
        for j in range(items):
            y_list = y_list_1[j]
            for k in range(items):
            # j = 0（從虛擬點出發）, y_jk = 1（從虛擬點到 k）, k 的 ID 是 id_list[k-1]
                if j == 0 and y_list[k] > 0:
                    if k == len(y_list) - 1:
                        continue

                if j == 0 and y_list[k] >=0.5: #  and k!= 0 and k != items-1
                    b_ID = id_list[k-1] 
                    plt.scatter(X_hat[b_ID,0],X_hat[b_ID,1], s=100, c=LABEL_COLOR_MAP[i], marker='P',edgecolors='black' , label='Startpoint for cluster '+str(i+1))

                # # y_ab = 1（從虛擬點到 k ）, a_ID = id_list[j-1], b_ID = id_list[k-1]
                elif y_list[k] >= 0.5 and k!= items-1:
                        a_ID = id_list[j-1]
                        b_ID = id_list[k-1]
                        xs = [X_hat[a_ID,0], X_hat[b_ID,0]]
                        ys = [X_hat[a_ID,1], X_hat[b_ID,1]]
                        plt.plot(xs, ys, c='black', linestyle = '-.')
                elif y_list[k] == 1 and k == len(y_list)-1:
                    continue



    plt.title("Detailed map of Ubike 2.0 stations before allocation when K = %d"%(groups))
    plt.legend()
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.savefig('4-%d-4_Before Allocation Map.png'%(groups))
    plt.show()

if __name__ == '__main__':
    # 01. K-Means Clustering
    cluster_k = 9
    
    temp_df = pd.DataFrame()

    data = pd.read_excel(data_path)
    df = data[["long", "lati"]]
    
    for i in range(1,cluster_k+1):
        kmeans = KMeans(n_clusters=i)
        labels = kmeans.fit_predict(df)
        # 將分組資料 (分類標籤) 併入原資料
        lb = pd.DataFrame(labels, columns=['labels%s'%i])
        temp_df = pd.concat((temp_df,lb), axis=1)
        
    output = pd.concat((data, temp_df), axis=1)
    output.to_csv("1-1_Cluster Result.csv", index=False)

    
    # k = 1~9 做9次kmeans, 並將每次結果的inertia收集在一個list裡
    kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(df) for k in range(1, cluster_k + 1)]
    inertias = [model.inertia_ for model in kmeans_list]
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.title('K-means SSE Plot (elbow method)')
    plt.plot(range(1, cluster_k + 1), inertias)
    # plt.plot(selected_K, distortions[selected_K - 2], 'go') # 最佳解
    plt.savefig('1-2_K-means SSE Plot.png')
    plt.show()
    
    silhouette_scores = [silhouette_score(df, model.labels_) for model in kmeans_list[1:]]
    plt.figure(figsize=(12, 12))
    plt.subplot(222)
    plt.title('K-means Silhouette Score Plot')
    plt.plot(range(2, cluster_k + 1), silhouette_scores)
    # plt.plot(selected_K, distortions[selected_K - 2], 'go') # 最佳解
    plt.savefig('1-3_K-means Silhouette Score Plot.png')
    plt.show()
    
    # 02. Split matrix
    df_dist = pd.read_excel("distance_sites.xlsx")
    df_dist = df_dist.iloc[:,1:]
    
    for c in range(1, cluster_k+1):
        cluster = output["labels%d"%c]
        cluster_list = [[] for i in range(c)]
    
        for i in range(c):
            for index, elem in enumerate(cluster):
                if elem == i:
                    cluster_list[i].append(index)
    
        matrix_list = []
        final_matrix = []
    
        for i in range(c):
            matrix_list.append(pd.DataFrame(columns = cluster_list[i], index= cluster_list[i]))
            matrix_list[i] = matrix_list[i].merge(df_dist[cluster_list[i]], left_index=True, right_index=True, suffixes=('_x', ''))
            matrix_list[i].drop(matrix_list[i].filter(regex='_x$').columns.tolist(),axis=1, inplace=True)
            
        newpath = "./Kmeans_clustered_dist_matrix"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
            
        writer = pd.ExcelWriter('./Kmeans_clustered_dist_matrix/Clustered_dist_matrix_labels%d.xlsx'%c)
        sheet_name = ["Cluster %d"%(i) for i in range(c)]
        for i, j in zip(matrix_list, sheet_name):
            i.to_excel(writer, sheet_name=j)  
        writer.save()
    
    # 03. Gurobi solver
    # Parameters
    detail = {c+1:{} for c in range(cluster_k)}
    for c in range(cluster_start, cluster_k+1):
        K = c
        A = 16
        lambda_value = lambda_val # 0.4
        S_H = 0.5 + lambda_value
        S_L = 0.5 - lambda_value
        L = 10
        van_cost = 250
        oil_cost = 3.136
        print("============ K : %d ==============="%(c))
        detail[c] = return_detail(K,A,S_H,S_L,L,lambda_value,van_cost,oil_cost)
 

    # 04. Map and routes visualization
    df_result = pd.read_csv("1-1_Cluster Result.csv")
    stations = len(df_result.index)

    X_hat = np.asmatrix(df_result[["long", "lati"]].to_numpy())
    Label = list(df_result["address"])
    
    for i in range(len(Label)):
        Label[i] = Label[i][:5]
        
    for c in range(cluster_start, cluster_k+1):
        cluster = df_result["labels%d"%c]
        solver_result = detail[c]

        # Plotting clustering result
        LABEL_COLOR_MAP = {0:'orange', 1:'green', 2:'cyan', 3:'pink', 4:'m', 5:'yellow', 6:'peru', 7:'lawngreen', 8:'deepskyblue', 9:'blueviolet'}                 
        label_color = [LABEL_COLOR_MAP[l] for l in cluster]
        
        plt.rcParams["figure.figsize"] = (15,10)
        plt.scatter([X_hat[:,0]], [X_hat[:,1]], c=label_color, cmap='viridis')
        # plt.scatter([centroids[:,0]], [centroids[:,1]], s=30, c='r', marker='P', label='Centroid for cluster')
        # labels
        # plt.scatter([X_hat[:,0]], [X_hat[:,1]], c="blue", marker='P', label='Centroid for cluster')
        
        # for i in range(len(Label)):
            # plt.annotate(Label[i], (X_hat[i,0]-0.0008, X_hat[i,1]-0.00025))
        
        station_string = [["" for i in range(53)] for j in range(c)]
        
        for i in range(c):
            if solver_result[i] == "infeasible":
                continue
            id_list = solver_result[i]["name_list"]
            a_list = solver_result[i]["r_a"]
            b_list = solver_result[i]["r_b"]
            for j in range(1,len(a_list)-1):
                id = id_list[j-1]
                if solver_result[i]["r_a"][j] > 0:
                    plt.annotate("%s(取 %d)"%(Label[id], int(solver_result[i]["r_a"][j]+0.5)), (X_hat[id,0], X_hat[id,1]-0.00025), color='red', ha='center')
                    station_string[i][j-1] = "%s (取 %d)"%(Label[id], int(solver_result[i]["r_a"][j]+0.5))
                elif solver_result[i]["r_b"][j] > 0:
                    plt.annotate("%s(放 %d)"%(Label[id], int(solver_result[i]["r_b"][j]+0.5)), (X_hat[id,0], X_hat[id,1]-0.00025), color='blue', ha='center')
                    station_string[i][j-1] = "%s (放 %d)"%(Label[id], int(solver_result[i]["r_b"][j]+0.5))
                # print(i,j)
                # print(y_list)
        
        path_string = [[] for j in range(c)]
        start_end_points = [[] for j in range(c)]
                
        for i in range(c):
            if solver_result[i] == "infeasible":
                continue
            id_list = solver_result[i]["name_list"]
            y_list_1 = solver_result[i]["r_y"]
            items = len(y_list_1)
            for j in range(0,items):
                y_list = y_list_1[j]
                # print(y_list)
                for k in range(0,items):
                # j = 0（從虛擬點出發）, y_jk = 1（從虛擬點到 k）, k 的 ID 是 id_list[k-1]
                    if y_list_1[j][-1] > 0:
                        if len(start_end_points[i]) < 2:
                            start_end_points[i].append(Label[id_list[j-1]])
                        
                    if j == 0 and y_list[k] > 0:
                        if k == len(y_list) - 1:
                            continue
                        else:
                        #print((i,j,k))
                            b_ID = id_list[k-1] 
                            start_end_points[i].append(Label[b_ID])
                            plt.scatter(X_hat[b_ID,0],X_hat[b_ID,1], s=100, c=LABEL_COLOR_MAP[i], marker='P',edgecolors='black' , label='Startpoint for cluster '+str(i+1))
            
                    # # y_ab = 1（從虛擬點到 k ）, a_ID = id_list[j-1], b_ID = id_list[k-1]
                    elif y_list[k] > 0 and k!= items-1:
                            #print((i,j,k))
                            a_ID = id_list[j-1]
                            b_ID = id_list[k-1]
                            xs = [X_hat[a_ID,0], X_hat[b_ID,0]]
                            ys = [X_hat[a_ID,1], X_hat[b_ID,1]]
                            path_string[i].append([Label[a_ID],Label[b_ID]])
                            plt.plot(xs, ys, c='black', linestyle = '-.')
                            # plt.scatter(xs, ys, c='r')
                    elif y_list[k] > 0 and k == len(y_list)-1:
                        continue
                    
    
        plt.title("Allocation schedule map of Ubike 2.0 stations in NTU campus when K = %d"%(c))
        plt.legend()
        plt.xlabel("Latitude")
        plt.ylabel("Longitude")
        plt.savefig('4-%d-1_Allocation Schedule Map.png'%(c))
        plt.show()
        
        for i,sub_string in enumerate(station_string):
            station_string[i] = [x for x in sub_string if x]
            
        depart = {i:[] for i in range(c)}
        arrive = {i:[] for i in range(c)}
        
        for no, clusters in enumerate(path_string):
            for pair in clusters:
                depart[no].append(pair[0])
                arrive[no].append(pair[1])
        try:        
            FINAL_PATH = [[start_end_points[i][0]] for i in range(c)]
        
            for no in range(c):
                stop_loop = False
                trial = 0
                while stop_loop == False:
                    for idx_i, str_i in enumerate(depart[no]):
                        for k in range(len(FINAL_PATH[no])):
                            if FINAL_PATH[no][k] == str_i:
                                if arrive[no][idx_i] not in FINAL_PATH[no]:
                                    FINAL_PATH[no].append(arrive[no][idx_i])
                                    if arrive[no][idx_i] == start_end_points[no][1]:
                                        stop_loop = True
                    trial += 1
                    if trial > 1000:
                        stop_loop = True
            
            for clust_no, clusters in enumerate(FINAL_PATH):
                for str_idx, string in enumerate(clusters):
                    for p in range(len(station_string[clust_no])):
                        if string == station_string[clust_no][p][:5]:
                            FINAL_PATH[clust_no][str_idx] = station_string[clust_no][p]
                            break
    
            x = []
            y = []
            for a in range(len(FINAL_PATH)):
                x.append([i for i in range(len(FINAL_PATH[a]))])
                y.append([0 for i in range(len(FINAL_PATH[a]))])
            
            
            for x_clust, x_axis in enumerate(x):
                if len(x_axis) > 9:
                    for count in range(int(len(x_axis) / 9 + 0.5)):
                        for elem in range(9):
                            if ((count+1)*9 + elem ) < len(x[x_clust]):
                                if count % 2 == 0:
                                    x[x_clust][(count+1)*9 + elem ] = 9 - elem - 1
                                elif count % 2 == 1:
                                    x[x_clust][(count+1)*9 + elem ] = elem
                                y[x_clust][(count+1)*9 + elem ] -= 0.1*(count + 1)
            
            fig, axs = plt.subplots(c,1,figsize=(20,10))
            fig.suptitle('Allocation schedule routes for Ubike 2.0 stations in NTU campus',  size=20)
            for i in range(c):
                axs[i].plot(x[i], y[i], color = LABEL_COLOR_MAP[i], linestyle = '-.', marker = 'o')
                axs[i].axis('off')
                axs[i].set_xlim(-0.5,8.5)
                axs[i].set_ylim(min(y[i])-0.05, 0.05)
                axs[i].scatter(0,0,s=100, marker='P', edgecolors='black', linewidths=1.5, color = LABEL_COLOR_MAP[i])
                for j in range(len(FINAL_PATH[i])):
                    if "放" in FINAL_PATH[i][j]:
                        axs[i].annotate(FINAL_PATH[i][j], (x[i][j]-0.35, y[i][j]-0.03), size=13, color='b')
                    elif "取" in FINAL_PATH[i][j]:
                        axs[i].annotate(FINAL_PATH[i][j], (x[i][j]-0.35, y[i][j]-0.03), size=13, color='red')
            
            plt.savefig('4-%d-2_Allocation Routes.png'%(c))
            plt.show()
           
            draw_before_ratio(c)
            draw_after_ratio(c)
        except:
            continue
                
    # 5. Decide best k by objective values graphs
    obj_list = [[0]]
    obj_sum = []
    infeasible_list = ["" for i in range(cluster_k+1)]    
    
    for i in range(cluster_start,cluster_k+1):
        obj_dic = detail[i]
        obj_list.append([])
        temp = 0
        for j in range(len(obj_dic)):
            if detail[i][j] == "infeasible":
                infeasible_list[i] = "(Partial INF)"
                continue
            obj = int(detail[i][j]["objVal"])
            obj_list[i].append(obj)
            temp += obj
        obj_sum.append(temp)

    plt.figure(figsize=(12, 5))
     
    # Defining the values for x-axis, y-axis
    # and from which dataframe the values are to be picked
    plots = sns.barplot([i+1 for i in range(cluster_start-1, len(obj_sum))], obj_sum)
     
    # Iterrating over the bars one-by-one
    for idx, bar in enumerate(plots.patches):
     
        # Using Matplotlib's annotate function and
        # passing the coordinates where the annotation shall be done
        plots.annotate("{:,}".format(int(bar.get_height())),(bar.get_x() + bar.get_width() / 2, bar.get_height()+obj_sum[0]*0.02), ha='center', va='center',
                       size=15, xytext=(0, 5), textcoords='offset points', color='black')
        plots.annotate(infeasible_list[idx] ,(bar.get_x() + bar.get_width() / 2, bar.get_height()*1.2), ha='center', va='center',
                       size=10, xytext=(0, 5), textcoords='offset points', color='red')
     
     
    # Setting the title for the graph
    plt.title("Objective values under different k when lambda = %2f"%(lambda_value))
    plt.ylim(0, max(obj_sum) *1.5)
    
    
    # Finally showing the plot
    plt.savefig('5_Objective values under different k when lambda is %2f.png'%(lambda_val))
    plt.show()
    
