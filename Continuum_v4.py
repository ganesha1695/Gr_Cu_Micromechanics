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
    a=conv*240.91
    b=conv*150.49
    c=conv*127.137
    
    #a=conv*174.475 
    #b=conv*127.839 
    #c=conv*84.438
    
    return -np.matmul(LA.inv(T_ik(theta,a,b,c)),np.transpose(R_ik(theta,a,b,c)))

def N_2(theta):
    conv=(1/160.2176621) #GPa to eV/A^3
    a=conv*240.91
    b=conv*150.49
    c=conv*127.137
    
    #a=conv*174.475 
    #b=conv*127.839 
    #c=conv*84.438
    return LA.inv(T_ik(theta))
    
def N_3(theta):
    conv=(1/160.2176621) #GPa to eV/A^3
    a=conv*240.91
    b=conv*150.49
    c=conv*127.137
    
    #a=conv*174.475 
    #b=conv*127.839 
    #c=conv*84.438
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

print(S_pi_calc(N,np.pi))

def G(r,theta,S_pi,L_pi):
    I=np.eye(3, dtype=int)
    G_ik=np.zeros([3,3])
    G_ik=-np.matmul(((1/np.pi)*np.log(r)*I+S_pi_calc(N,theta)-0.5*S_pi),LA.inv(L_pi))
    return G_ik

def g(r,theta,L_pi,N_1_theta):
    I=np.eye(3)
    t=I*np.cos(theta)-N_1_theta*np.sin(theta)
    return (1/r)*np.matmul(t,LA.inv(L_pi))

Nx=81
Ny=31
lx=36.15*2
ly=-54.225
nx, ny = (Nx, Ny)
#x = np.linspace(-lx, lx, nx)
#y = np.linspace(ly,0, ny)
G_grid_00=np.zeros([Nx,Ny])
G_grid_01=np.zeros([Nx,Ny])
G_grid_02=np.zeros([Nx,Ny])
G_grid_10=np.zeros([Nx,Ny])
G_grid_11=np.zeros([Nx,Ny])
G_grid_12=np.zeros([Nx,Ny])
G_grid_20=np.zeros([Nx,Ny])
G_grid_21=np.zeros([Nx,Ny])
G_grid_22=np.zeros([Nx,Ny])
g_grid_00=np.zeros([Nx,Ny])
g_grid_01=np.zeros([Nx,Ny])
g_grid_10=np.zeros([Nx,Ny])
g_grid_11=np.zeros([Nx,Ny])
D=np.array([0.15,0.13,0])
#f=np.array([0.001,0,0])
f=np.array([0,0,0])
#xv, yv = np.meshgrid(x, y)

xv=np.zeros([Nx,Ny])
yv=np.zeros([Nx,Ny])
hx=2*lx/nx
hy=ly/ny
for i in range(Nx):
    for j in range(Ny):
        xv[i,j]=-lx+(i*hx)
        yv[i,j]=j*hy

theta_m=np.pi
L_pi=L_calc(N,theta_m)
print(LA.inv(L_pi))
S_pi=S_pi_calc(N,theta_m)
u_x=np.zeros([Nx,Ny])
u_y=np.zeros([Nx,Ny])
for i in range(Nx):
    for j in range(Ny):
       radius=np.sqrt(xv[i,j]**2+yv[i,j]**2)
       if(xv[i,j]>0):
         angle=np.arctan(yv[i,j]/xv[i,j])
       else:
         angle=np.pi+np.arctan(yv[i,j]/xv[i,j])
       kj=G(radius,angle,S_pi,L_pi)
       G_grid_00[i,j]=kj[0][0]
       G_grid_01[i,j]=kj[0][1]
       G_grid_02[i,j]=kj[0][2]
       G_grid_10[i,j]=kj[1][0]
       G_grid_11[i,j]=kj[1][1]
       G_grid_12[i,j]=kj[1][2]
       G_grid_20[i,j]=kj[2][0]
       G_grid_21[i,j]=kj[2][1]
       G_grid_22[i,j]=kj[2][2]
       trt=N_1(angle)
       g_return=g(radius,angle,L_pi,trt)
       g_grid_00[i,j]=g_return[0][0]
       g_grid_01[i,j]=g_return[0][1]
       g_grid_10[i,j]=g_return[1][0]
       g_grid_11[i,j]=g_return[1][1]
       mn=np.matmul(g_return,D)+np.matmul(kj,f)
       if(radius<3.615):
          u_x[i,j]=0
          u_y[i,j]=0
       elif(xv[i,j]>0 and yv[i,j]>-1.5):
          u_x[i,j]=0
          u_y[i,j]=0
       else:
          u_x[i,j]=mn[0]
          u_y[i,j]=mn[1]
       
fid_x=open('Green_function_result.txt','w+')

for j in range(Ny):
    for i in range(Nx):
        fid_x.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(xv[i,j],yv[i,j],g_grid_00[i,j],g_grid_01[i,j],g_grid_10[i,j],g_grid_11[i,j],u_x[i,j],u_y[i,j]))
fid_x.close()  

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize =(14, 8))
cp=ax.quiver(xv, yv, u_x, u_y,scale=2)
ax.set_aspect('equal')
"""
fig,ax=plt.subplots(1,1)
cp = ax.contourf(xv, yv, u_x,cmap="rainbow")
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_aspect('equal')
"""
plt.show() 
