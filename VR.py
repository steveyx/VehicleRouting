# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 19:20:03 2017

@author: cloud
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import matplotlib.animation
from matplotlib import colors
import six
from sklearn import cluster, mixture
from sklearn.cluster import AgglomerativeClustering
colors_ = list(six.iteritems(colors.cnames))


ctr_x = 11.552931
ctr_y = 104.933636

markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

def KmeansSolution():
    dataXY = data.loc[:nodes-1,['x','y']].values
    clf = cluster.KMeans(n_clusters=driver_num) 
    clf.fit(dataXY)
    np_clf_data = np.zeros((nodes,2),dtype=int)
    np_clf_data[:,0] = np.arange(nodes)
    np_clf_data[:,1] = clf.labels_
    pd_clf_data = pd.DataFrame(np_clf_data,columns=['loc','label'])
    pd_clf_data = pd_clf_data.sort_values(by=['label'])
    matrix_a = pd_clf_data['loc'].values
    grp = pd_clf_data.groupby(by=['label'])
    matrix_m = grp.count()['loc'].values
    s, a, m = getSolution(distXY, matrix_a, nodes, matrix_m)
    return(s,a,m)
    
def GaussianMixtureSolution():
    dataXY = data.loc[:nodes-1,['x','y']].values
    clf = mixture.GaussianMixture(
        n_components=driver_num, covariance_type='full')
    clf.fit(dataXY)
    np_clf_data = np.zeros((nodes,2),dtype=int)
    np_clf_data[:,0] = np.arange(nodes)
    y_pred = clf.predict(dataXY)
#    np_clf_data[:,1] = clf.labels_
    np_clf_data[:,1] = y_pred
    pd_clf_data = pd.DataFrame(np_clf_data,columns=['loc','label'])
    pd_clf_data = pd_clf_data.sort_values(by=['label'])
    matrix_a = pd_clf_data['loc'].values
    grp = pd_clf_data.groupby(by=['label'])
    matrix_m = grp.count()['loc'].values
    s, a, m = getSolution(distXY, matrix_a, nodes, matrix_m)
    return(s,a,m)
    
    
def SpectralSolution():
    dataXY = data.loc[:nodes-1,['x','y']].values
    clf = cluster.SpectralClustering(
        n_clusters=driver_num, eigen_solver='arpack',
        affinity="nearest_neighbors")
    clf.fit(dataXY)
    np_clf_data = np.zeros((nodes,2),dtype=int)
    np_clf_data[:,0] = np.arange(nodes)
    np_clf_data[:,1] = clf.labels_
    pd_clf_data = pd.DataFrame(np_clf_data,columns=['loc','label'])
    pd_clf_data = pd_clf_data.sort_values(by=['label'])
    matrix_a = pd_clf_data['loc'].values
    grp = pd_clf_data.groupby(by=['label'])
    matrix_m = grp.count()['loc'].values
    s, a, m = getSolution(distXY, matrix_a, nodes, matrix_m)
    return(s,a,m)


def AggloClusterSolution():
    dataXY = data.loc[:nodes-1,['x','y']].values
    s_min = 99999.0
    for linkage in ('ward', 'average', 'complete'):
        clf = AgglomerativeClustering(linkage=linkage, n_clusters=driver_num)
        clf.fit(dataXY)
        np_clf_data = np.zeros((nodes,2),dtype=int)
        np_clf_data[:,0] = np.arange(nodes)
        np_clf_data[:,1] = clf.labels_
        pd_clf_data = pd.DataFrame(np_clf_data,columns=['loc','label'])
        pd_clf_data = pd_clf_data.sort_values(by=['label'])
        matrix_a = pd_clf_data['loc'].values
        grp = pd_clf_data.groupby(by=['label'])
        matrix_m = grp.count()['loc'].values
        s, a, m = getSolution(distXY, matrix_a, nodes, matrix_m)
        if s < s_min:
            s_min = s
            a_min = a
            m_min = m
#        print(s, a, m)      
    return(s_min,a_min,m_min)

def getRouteDriverI(matrix_a,matrix_m,d_i):
    m_a = matrix_a
    start = np.sum(matrix_m[:d_i])
    end = start + matrix_m[d_i]    
    return m_a[start:end]
    
def updateRouteDriverI(matrix_a,matrix_m,d_i,new_route):
    m_a = matrix_a.copy()
    start = np.sum(matrix_m[:d_i])
    end = start + matrix_m[d_i]
    m_a[start:end] = new_route
    return m_a


def optSwap(route, i, k):
    new = route[:i]
    ik =  route[i:k]
    ik = ik[::-1]
    k_aft = route[k:]
    return np.concatenate((new,ik,k_aft))
 
def getRouteDist(route,locXY):
    s = 0
    s = s + locXY[-1,route[0]]
    for j in range(len(route)-1):
        s = s + locXY[route[j],route[j+1]]
    s = s + locXY[route[-1],-1]
    return (s)

def getBestRouteBy2opt(route,driver,matrix_a,matrix_m):
    global s_min,m_min,a_min
    existing_route = route
    best_distance = getRouteDist(route,distXY)
    new_distance = best_distance
    driver_nodes = len(route)
    while (True) & (driver_nodes>1):
        flag = False
        for i in range(driver_nodes-1):
            for k in range(i+1, driver_nodes):
                new_route = optSwap(existing_route, i, k)
                new_distance = getRouteDist(new_route,distXY)
                if (new_distance < best_distance):
                  existing_route = new_route
                  best_distance = new_distance
                  flag = True
                  new_a = updateRouteDriverI(matrix_a,matrix_m,driver,new_route)
#                  print(new_a-matrix_a)
                  matrix_a = new_a
                  s1, a1, _ = getSolution(distXY, new_a, nodes, matrix_m)
                  if (s1 < s_min):
                      s_min = s1
                      m_min = matrix_m
                      a_min = a1
                      res.append([len(res),s1, matrix_m, a1])
                  break
            if flag:
                  break
        else:
            break
    return existing_route,matrix_a

def getAllRouteBy2opt(matrix_a,matrix_m,driver_num):
#    t_start =time.clock()
    for driver in range(driver_num):
        route_i = getRouteDriverI(matrix_a,matrix_m,driver) 
#        print("Get route for driver ", driver)
        route_i,matrix_a = getBestRouteBy2opt(route_i,driver,matrix_a,matrix_m)
        new_routes = updateRouteDriverI(matrix_a,matrix_m,driver,route_i)
#        res.append([len(res),s,m,new_routes])
#        print("Updated route for driver ", driver)
#        t_end =time.clock() -t_start
#        print("Updated route time for driver ", t_end)
    return(new_routes)
    
def plotOneSolution(a, m_num):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    lines = []
    start, end = getStartEndIndex(a,m_num)            
    for m_i in range(len(m_num)):
        x =  data.loc[a[start[m_i]:end[m_i]],'x'].values
        y =  data.loc[a[start[m_i]:end[m_i]],'y'].values
        line, = ax.plot(x, y, marker=markers[m_i%12], label='Driver'+str(m_i))
        lines.append(line)
    ax.scatter([ctr_x],[ctr_y],label='Depot') 
    plt.legend()

def getLocData(data):
    ctr_x = 11.552931
    ctr_y = 104.933636
    M = len(data)
    arr_x = data.x.values[:M]
    arr_y = data.y.values[:M]
    arr_x = np.append(arr_x,[ctr_x])
    arr_y = np.append(arr_y,[ctr_y])
    
    N =  len(arr_x)
    xx =  arr_x.reshape(-1,1) * np.ones((N,N))
    xxt = arr_x * np.ones((N,N))
    yy =  arr_y.reshape(-1,1) * np.ones((N,N))
    yyt = arr_y * np.ones((N,N))
    
    distX=xx-xxt
    distY=yy-yyt
    distXY = np.sqrt(distX**2 + distY**2)
#    return distXY[:101,:101]
    return distXY

def getInitRandSolution(locXY,nodes,m):
    a = np.arange(nodes)
    np.random.shuffle(a)
    m_num = np.zeros(m, dtype=int)
#    m_num[0] = np.random.randint(1, nodes-m+2)
    m_num[0] = np.random.randint(1, int(1.8 * nodes/m))
    tot = m_num[0]
    for i in range(1,m-1):
        m_num[i] = np.random.randint(1, nodes-tot-(m-i-1)+1)
        if (m_num[i]> (nodes/m *1.8)):
            m_num[i] = np.random.randint(1, int(1.8* nodes/m))
        tot = tot + m_num[i]
    m_num[m-1] = nodes-tot
    np.random.shuffle(m_num)

    tot = m_num[0]
    start = [0]
    end = [m_num[0]]
    for i in range(1,m-1):
        start.append(tot)
#        m_num[i] = np.random.randint(1, nodes-tot-(m-i-1)+1)
        tot = tot + m_num[i]
        end.append(tot)        
#    m_num[m-1] = nodes-tot
    start.append(tot)
    end.append(nodes)            
    s = 0
    for i in range(m): 
      s = s + locXY[-1,a[start[i]]]
      for j in range(start[i],end[i]-1):
          s = s + locXY[a[j],a[j+1]]
      s = s + locXY[a[end[i]-1],-1]
    return s, a, m_num

def getSolution(locXY, a, nodes, m_num):
    tot = m_num[0]
    start = [0]
    end = [m_num[0]]
    for i in range(1,len(m_num)-1):
        start.append(tot)
        tot = tot + m_num[i]
        end.append(tot)        
    start.append(tot)
    end.append(nodes)            
    s = 0
    for i in range(len(m_num)): 
      s = s + locXY[-1,a[start[i]]]
      for j in range(start[i],end[i]-1):
          s = s + locXY[a[j],a[j+1]]
      s = s + locXY[a[end[i]-1],-1]
    return s, a, m_num

def getMutation(chrom,l_mut):
    n = int(len(a)/2)
    a_1 = random.sample(range(0, n), l_mut)
    a_2 = random.sample(range(n, len(a)), l_mut)
    temp = chrom[a_1]
    chrom[a_1] = chrom[a_2]
    chrom[a_2] = temp
    return chrom

def getStartEndIndex(a,m_num):
    nodes = len(a)
    tot = m_num[0]
    start = [0]
    end = [m_num[0]]
    for i in range(1,len(m_num)-1):
        start.append(tot)
        tot = tot + m_num[i]
        end.append(tot)        
    start.append(tot)
    end.append(nodes)   
    return start,end

def updataAnimation(i):
    a = res[i][3]
    m_num = res[i][2]
    start, end = getStartEndIndex(a,m_num)            
    for m_i in range(len(m_num)):
#        x = locXY[]
        x =  data.loc[a[start[m_i]:end[m_i]],'x'].values
        y =  data.loc[a[start[m_i]:end[m_i]],'y'].values
#        sc[m_i].set_offsets(np.c_[x,y])        
        lines[m_i].set_data(x, y)
#            sc.set_offsets(np.c_[x,y])
    return lines

loc = pd.read_csv("locations1.csv", names =['x','y'], header=None)
data_dup = loc.duplicated()
data = loc.drop_duplicates()
data = data[data.x<20]
data = data.reset_index(drop ='True')

distXY = getLocData(data)
nodes=len(distXY)-1
driver_num =25

res =[]
t_start =time.clock()
s_min = 999999.0
for i in range(10):
    s, a, m = SpectralSolution()
    if s < s_min:
#        print(s)
        res.append([len(res),s,m,a])
#        print("Spectral Solution: ",s)
        s_min = s
        m_min = m
        a_min = a
    #plotOneSolution(a, m)
    a = getAllRouteBy2opt(a,m,driver_num)
#    res.append([len(res),s,m,a])
    t_end =time.clock() -t_start
    print("Iteration: {0}, time: {1:.6f}, total distance: {2:.6f}".format(i,t_end,s_min))
s, a, m = getSolution(distXY, a_min, nodes, m_min)    
res.append([len(res),s,m,a])
print("Final solution - total traveling distance: ", s_min)

 
distPts = pd.DataFrame()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
a = res[0][3]
m_num = res[0][2]
sc = []
lines = []
start, end = getStartEndIndex(a,m_num)            
for m_i in range(len(m_num)):
    x =  data.loc[a[start[m_i]:end[m_i]],'x'].values
    y =  data.loc[a[start[m_i]:end[m_i]],'y'].values
    line, = ax.plot(x, y, marker=markers[m_i % 13], label='Driver'+str(m_i))
    lines.append(line)
ax.scatter([ctr_x],[ctr_y],label='Depot',marker='o',
           facecolors='none', edgecolors='r', linewidths=5,
           s=200) 
plt.legend()
ani = matplotlib.animation.FuncAnimation(fig, updataAnimation,
                                         np.arange(0, len(res)), 
                               interval=10, blit=True,repeat=False)
plt.show()
