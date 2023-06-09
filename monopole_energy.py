import numpy as np
from numpy import linalg as LA
import os

N=180

def Q_ik(theta,a,b,c):
     n1=np.cos(theta)
     n2=np.sin(theta)
     Q_ik=np.zeros([3,3])
     Q_ik[0][0]=a*n1*n1+c*n2*n2
     Q_ik[0][1]=b*n1*n2+c*n2*n1
     Q_ik[0][2]=0
     Q_ik[1][0]=b*n2*n1+c*n1*n2
     Q_ik[1][1]=c*n1*n1+a*n2*n2
     Q_ik[1][2]=0
     Q_ik[2][0]=0
     Q_ik[2][1]=0
     Q_ik[2][2]=c*n1*n1+c*n2*n2
     return Q_ik
     
def R_ik(theta,a,b,c):
     n1=np.cos(theta)
     n2=np.sin(theta)
     m1=-np.sin(theta)
     m2=np.cos(theta)
     R_ik=np.zeros([3,3])
     R_ik[0][0]=a*n1*m1+c*n2*m2
     R_ik[0][1]=b*n1*m2+c*n2*m1
     R_ik[0][2]=0
     R_ik[1][0]=b*n2*m1+c*n1*m2
     R_ik[1][1]=c*n1*m1+a*n2*m2
     R_ik[1][2]=0
     R_ik[2][0]=0
     R_ik[2][1]=0
     R_ik[2][2]=c*n1*m1+c*n2*m2
     return R_ik
     
def T_ik(theta,a,b,c):
     m1=-np.sin(theta)
     m2=np.cos(theta)
     T_ik=np.zeros([3,3])
     T_ik[0][0]=a*m1*m1+c*m2*m2
     T_ik[0][1]=b*m1*m2+c*m2*m1
     T_ik[0][2]=0
     T_ik[1][0]=b*m2*m1+c*m1*m2
     T_ik[1][1]=c*m1*m1+a*m2*m2
     T_ik[1][2]=0
     T_ik[2][0]=0
     T_ik[2][1]=0
     T_ik[2][2]=c*m1*m1+c*m2*m2
     return T_ik

def N_1(theta):
    conv=(1/160.2176621) #GPa to eV/A^3
    #a=conv*240.91
    #b=conv*150.49
    #c=conv*127.137
    
    a=conv*174.475 
    b=conv*127.839 
    c=conv*84.438
    
    return -np.matmul(LA.inv(T_ik(theta,a,b,c)),np.transpose(R_ik(theta,a,b,c)))

def N_2(theta):
    conv=(1/160.2176621) #GPa to eV/A^3
    #a=conv*240.91
    #b=conv*150.49
    #c=conv*127.137
    
    a=conv*174.475 
    b=conv*127.839 
    c=conv*84.438
    return LA.inv(T_ik(theta))
    
def N_3(theta):
    conv=(1/160.2176621) #GPa to eV/A^3
    #a=conv*240.91
    #b=conv*150.49
    #c=conv*127.137
    
    a=conv*174.475 
    b=conv*127.839 
    c=conv*84.438
    return -np.matmul(R_ik(theta,a,b,c),N_1(theta))-Q_ik(theta,a,b,c)

def simpson_integral(a,h):
    integration = a[0] + a[len(a)-1]
    for i in range(1,len(a)-1):
        if i%2 == 0:
            integration = integration + 2 * a[i]
        else:
            integration = integration + 4 * a[i]
    
    # Finding final integration value
    integration = integration * h/3
    
    return integration

def L_calc(N,theta_m):
    
    div=theta_m/N 
    N3_00=[]
    N3_01=[]
    N3_02=[]
    N3_10=[]
    N3_11=[]
    N3_12=[]
    N3_20=[]
    N3_21=[]
    N3_22=[]
    
    L_pi=np.zeros([3,3])
    
    for ki in range(N+1):
          p=N_3(div*ki)
          N3_00.append(p[0][0])
          N3_01.append(p[0][1])
          N3_02.append(p[0][2])
          N3_10.append(p[1][0])
          N3_11.append(p[1][1])
          N3_12.append(p[1][2])
          N3_20.append(p[2][0])
          N3_21.append(p[2][1])
          N3_22.append(p[2][2])
    
    L_pi[0][0]=simpson_integral(N3_00,div)
    L_pi[0][1]=simpson_integral(N3_01,div)
    L_pi[0][2]=simpson_integral(N3_02,div)
    L_pi[1][0]=simpson_integral(N3_10,div)
    L_pi[1][1]=simpson_integral(N3_11,div)
    L_pi[1][2]=simpson_integral(N3_12,div)
    L_pi[2][0]=simpson_integral(N3_20,div)
    L_pi[2][1]=simpson_integral(N3_21,div)
    L_pi[2][2]=simpson_integral(N3_22,div)
    
    return -(1/np.pi)*L_pi

def S_pi_calc(N,theta_m):
    
    div=theta_m/N 
    N1_00=[]
    N1_01=[]
    N1_02=[]
    N1_10=[]
    N1_11=[]
    N1_12=[]
    N1_20=[]
    N1_21=[]
    N1_22=[]
    
    S_pi=np.zeros([3,3])
    
    for ki in range(N+1):
          p=N_1(div*ki)
          N1_00.append(p[0][0])
          N1_01.append(p[0][1])
          N1_02.append(p[0][2])
          N1_10.append(p[1][0])
          N1_11.append(p[1][1])
          N1_12.append(p[1][2])
          N1_20.append(p[2][0])
          N1_21.append(p[2][1])
          N1_22.append(p[2][2])
    
    S_pi[0][0]=simpson_integral(N1_00,div)
    S_pi[0][1]=simpson_integral(N1_01,div)
    S_pi[0][2]=simpson_integral(N1_02,div)
    S_pi[1][0]=simpson_integral(N1_10,div)
    S_pi[1][1]=simpson_integral(N1_11,div)
    S_pi[1][2]=simpson_integral(N1_12,div)
    S_pi[2][0]=simpson_integral(N1_20,div)
    S_pi[2][1]=simpson_integral(N1_21,div)
    S_pi[2][2]=simpson_integral(N1_22,div)
    
    return (1/np.pi)*S_pi 

lx=81
nx=35

init_lx=11
h=(lx-init_lx)/nx
d=[]
for i in range(nx):
   d.append(init_lx+i*h)

D=np.array([0.018,-0.018])
f=np.array([-0.0000,0])
#f=np.array([0.0,0.0])

in_en_monopole=[]
I=np.eye(2)
theta_m=np.pi
L_pi=L_calc(N,theta_m)
S_pi=S_pi_calc(N,theta_m)

for i in range(nx):
   G=-np.matmul(((1/np.pi)*np.log(d[i])*I+0.5*S_pi[:2,:2]),LA.inv(L_pi[:2,:2]))
   e_mp_mp=np.matmul(f,np.matmul(G,np.transpose(f)))
   #print(e_mp_mp)
   g=-(1/d[i])*LA.inv(L_pi[:2,:2])
   e_mp_dp=-2*np.matmul(f,np.matmul(g,np.transpose(D)))
   #print(e_mp_dp)
   g_p=np.array([[1/(np.pi*L_pi[0,0]*d[i]*d[i]),0],[0,1/(np.pi*L_pi[0,0]*d[i]*d[i])]])
   e_dp_dp=np.matmul(D,np.matmul(g_p,np.transpose(D)))
   #print(e_dp_dp)
   in_en_monopole.append(e_mp_mp+e_mp_dp+e_dp_dp)
   #in_en_monopole.append(e_mp_mp+e_dp_dp)
"""
for i in range(nx):
   in_en_monopole.append(3.1521e-6*np.log(d[i])+6.6564e-4/(d[i]*d[i])-1.4182e-5)
"""
import matplotlib.pyplot as plt
plt.rc('font', size=60)          # controls default text sizes
plt.rc('axes', titlesize=40)     # fontsize of the axes title
plt.rc('axes', labelsize=40)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=40)    # fontsize of the tick labels
plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('figure', titlesize=40)

plt.figure(figsize=(20,10))
plt.plot(d,in_en_monopole,'ro-')
plt.show() 
