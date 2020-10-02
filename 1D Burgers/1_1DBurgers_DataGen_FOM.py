# -*- coding: utf-8 -*-
"""
Full order model (FOM) data generation for the 1D Burgers problem 
with an initial condition of a square wave

Ref: "Forward sensitivity approach for estimating eddy viscosity closures 
     in nonlinear model reduction", Physical Review E, 2020
     by: Shady Ahmed, Kinjal Bhar, Omer San, Adil Rasheed

    
Code created: Saturday, April, 4, 2020
Last checked: Thursday, October 1, 2020
@author: Shady Ahmed
contact: shady.ahmed@okstate.edu
"""


#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#%% Define Functions

###############################################################################
#compute rhs for numerical solutions
#  r = -u*u' + nu*u''
###############################################################################
def rhs(nx,dx,nu,u):
    r = np.zeros(nx+1)
    u2 = u**2
    up = pade4d(u, dx, nx)
    up2 = pade4d(u2,dx,nx)
    upp = pade4dd(u,dx,nx)

    r = -(u*up + up2)/3 + nu*upp 
    return r

###############################################################################
# Numerical Routines
###############################################################################
# Thomas algorithm for solving tridiagonal systems:    
def tdma(a, b, c, r, up, s, e):
    
    for i in range(s+1,e+1):
        b[i] = b[i] - a[i]/b[i-1]*c[i-1]
        r[i] = r[i] - a[i]/b[i-1]*r[i-1]    
    up[e] = r[e]/b[e]   

    for i in range(e-1,s-1,-1):
        up[i] = (r[i]-c[i]*up[i+1])/b[i]

# Computing first derivatives using the fourth order compact scheme:  
def pade4d(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    ud = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 2.0
    r[i] = (-5.0*u[i] + 4.0*u[i+1] + u[i+2])/(2.0*h)
    
    a[1:n] = 1.0
    b[1:n] = 4.0
    c[1:n] = 1.0
    r[1:n] = 3.0*(u[2:n+1] - u[0:n-1])/h
   
    i = n
    a[i] = 2.0
    b[i] = 1.0
    r[i] = (-5.0*u[i] + 4.0*u[i-1] + u[i-2])/(-2.0*h)
    tdma(a, b, c, r, ud, 0, n)
    return ud
    
# Computing second derivatives using the foruth order compact scheme:  
def pade4dd(u, h, n):
    a, b, c, r = [np.zeros(n+1) for _ in range(4)]
    udd = np.zeros(n+1)
    i = 0
    b[i] = 1.0
    c[i] = 11.0
    r[i] = (13.0*u[i] - 27.0*u[i+1] + 15.0*u[i+2] - u[i+3])/(h*h)

    a[1:n] = 0.1
    b[1:n] = 1.0
    c[1:n] = 0.1
    r[1:n] = 1.2*(u[2:n+1] - 2.0*u[1:n] + u[0:n-1])/(h*h)
    
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd



#%% Main program:
    
# Inputs
nx =  4*1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)

Re  = 1e4   #control Reynolds
nu = 1/Re   #control dissipation

tm = 1      #maximum time
dt = 1e-4   #solver timestep
nt = int(tm/dt) #number of timesteps
t = np.linspace(0, tm, nt+1)

ns = 100 #number of snapshots to save
freq = int(nt/ns) #frequency of data storing

#%%

uFOM = np.zeros((nx+1,ns+1))

#compute initial condition
uu = np.zeros(nx+1)
for i in range(nx+1):
    if(i <= nx/2 ):
        uu[i] = 1.0
    else:
        uu[i] = 0.0

# boundary conditions: b.c. (not updated)
u1 = np.zeros(nx+1)
uu[0] = 0.0
uu[nx] = 0.0
u1[0] = 0.0
u1[nx]= 0.0       
    
#check for stability
neu = nu*dt/(dx*dx)
cfl = np.max(uu)*dt/dx
if (neu >= 0.25):
    print('Neu condition: reduce dt')
    sys.exit()
if (cfl >=  0.5):
    print('CFL condition: reduce dt')
    sys.exit()
      
#time integration
if1 = 0
uFOM[:,0] = uu 
for jj in range(1,nt+1):
    #RK3 scheme -- RK4 can be used instead
    # first step
    rr = rhs(nx,dx,nu,uu)
    u1[1:nx] = uu[1:nx] + dt*rr[1:nx]
    
    # second step
    rr = rhs(nx,dx,nu,u1)
    u1[1:nx] = 0.75*uu[1:nx] + 0.25*u1[1:nx] + 0.25*dt*rr[1:nx]
    	
    # third step
    rr = rhs(nx,dx,nu,u1)
    uu[1:nx] = 1.0/3.0*uu[1:nx] + 2.0/3.0*u1[1:nx] + 2.0/3.0*dt*rr[1:nx]
    
    # If you like to use RK4 instead:
    # k1 = rhs(nx,dx,nu,uu)
    # k2 = rhs(nx,dx,nu,uu+k1*dt/2)
    # k3 = rhs(nx,dx,nu,uu++k2*dt/2)
    # k4 = rhs(nx,dx,nu,uu++k3*dt)
    # uu = uu + (dt/6)*(k1+2*k2+2*k3+k4)
        
    # store  velocity field
    if(np.mod(jj,freq) == 0):
        if1=if1+1
        uFOM[:,if1] = uu 
        print(if1)

    # check for CFL
    cfl = np.max(uu)*dt/dx

    #check for numerical stability
    if (cfl > 1.0):
        print('Error: CFL limit exceeded')
        break
    
#%% Saving data

#create data folder
if os.path.isdir("./Data"):
    print('Data folder already exists')
else: 
    print('Creating data folder')
    os.makedirs("./Data")
 
print('Saving data')      
np.save('./Data/uFOM_Re'+str(int(Re))+'.npy',uFOM)


# create plot folder
if os.path.isdir("./Plots"):
    print('Plots folder already exists')
else: 
    print('Creating Plots folder')
    os.makedirs("./Plots")
    
plt.figure(figsize=(8,6))
plt.plot(x,uFOM[:,::5])
plt.xlabel(r'$x$')
plt.ylabel(r'$u$')
plt.savefig('./Plots/Burgers_Square_Re' + str(Re) + '.png', dpi = 500, bbox_inches = 'tight')
plt.show()


