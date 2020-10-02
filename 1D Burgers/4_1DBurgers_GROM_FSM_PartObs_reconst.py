# -*- coding: utf-8 -*-
"""
FSM estimation of global eddy viscosity parameter from sparse measurements with reconstruction map approach 
for the 1D Burgers problem with an initial condition of a square wave

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
from numpy import linalg as LA
from scipy.linalg import block_diag

from numpy.random import seed
seed(2)

import os
import time as clck

#%% Define Functions

###############################################################################
#POD Routines
###############################################################################         
def POD(u,R): #Basis Construction
    n,ns = u.shape
    U,S,Vh = LA.svd(u, full_matrices=False)
    Phi = U[:,:R]
    L = S**2
    #compute RIC (relative inportance index)
    RIC = sum(L[:R])/sum(L)*100   
    return Phi,L,RIC

def PODproj(u,Phi): #Projection
    a = np.dot(u.T,Phi)  # u = Phi * a.T
    return a

def PODrec(a,Phi): #Reconstruction    
    u = np.dot(Phi,a.T)    
    return u


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
    for i in range(1,n):
        a[i] = 1.0
        b[i] = 4.0
        c[i] = 1.0
        r[i] = 3.0*(u[i+1] - u[i-1])/h
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
    for i in range(1,n):
        a[i] = 0.1
        b[i] = 1.0
        c[i] = 0.1
        r[i] = 1.2*(u[i+1] - 2.0*u[i] + u[i-1])/(h*h)
    i = n
    a[i] = 11.0
    b[i] = 1.0
    r[i] = (13.0*u[i] - 27.0*u[i-1] + 15.0*u[i-2] - u[i-3])/(h*h)
    
    tdma(a, b, c, r, udd, 0, n)
    return udd



# Galerkin Projection
def rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a):
    r1, r2, r3 = [np.zeros(nr) for _ in range(3)]
    
    a = a.ravel()
    for k in range(nr):
        r1[k] = b_c[k] + nue*b_cc[k]
        r2[k] = np.sum(b_l[:,k]*a) + nue*np.sum(b_lc[:,k]*a)
    
    for k in range(nr):
        for i in range(nr):
            r3[k] = r3[k] + np.sum(b_nl[i,:,k]*a)*a[i]

    r = r1 + r2 + r3
    r = r.reshape(-1,1)    
    return r

###############################################################################
# Forward Sensitivity Method routines
###############################################################################    
# Model map
def M(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a0, dt):   
    
    a0 = a0.reshape(-1,1)
    k1 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a0)
    k2 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a0+k1*dt/2)
    k3 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a0+k2*dt/2)
    k4 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a0+k3*dt)
    a = a0 + (dt/6)*(k1+2*k2+2*k3+k4)
    return a


# Jacobian of RHS
def Jrhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a): #f(u)
    a = a.ravel()

    df = np.zeros((nr,nr+1))
    for k in range(nr):
        for i in range(nr):
            df[k,i] = b_l[i,k] + nue*b_lc[i,k] + + np.sum(b_nl[i,:,k]*a) + np.sum(b_nl[:,i,k]*a)
        df[k,nr] = b_cc[k] + np.sum(b_lc[:,k]*a)
    return df   


# Jacobian of model
def DM(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a):
    global dt
    
    k1 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a)
    k2 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a+k1*dt/2)
    k3 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a+k2*dt/2)
    #k4 = rhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a+k3*dt)
    
    k1p = Jrhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a)
    
    k2p = Jrhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a+k1*dt/2) @ \
         (np.eye(nr+1) + np.vstack((k1p,np.zeros((1,nr+1)) ))*dt/2)  
         
    k3p = Jrhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a+k2*dt/2) @ \
         (np.eye(nr+1) + np.vstack((k2p,np.zeros((1,nr+1)) ))*dt/2)
         
    k4p = Jrhs(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a+k3*dt) @ \
         (np.eye(nr+1) + np.vstack((k3p,np.zeros((1,nr+1)) ))*dt)
    
    D = np.hstack(( np.eye(nr),np.zeros((nr,1)) )) + (dt/6) * (k1p+2*k2p+2*k3p+k4p)
    Da = D[:,:nr]
    Dnue = D[:,nr].reshape(-1,1)
    return Da,Dnue

# Observational map
def h(a):
    global Phi_s
    z = Phi_s@a
    return z

# Jacobian of observational map
def Dh(a):
    global Phi_s
    D = Phi_s
    return D

   
    
#%% Main program:
    
# Inputs
nx =  4*1024  #spatial resolution
lx = 1.0    #spatial domain
dx = lx/nx
x = np.linspace(0, lx, nx+1)

Re  = 1e4 #control Reynolds
nu = 1/Re   #control dissipation

tm = 1      #maximum time
ns = 100   #number of snapshot per each parameter value 
dt = tm/ns
t = np.linspace(0, tm, ns+1)

nr = 8     #number of modes

#%% FOM snapshot generation for training
print('Reading FOM snapshots...')
uFOM = np.load('./Data/uFOM_Re'+str(int(Re))+'.npy')

um = np.mean(uFOM,1)
u = uFOM - um.reshape(-1,1)

#%% POD basis computation for training data
print('Computing POD basis...')
Phi = np.zeros((nx+1,nr)) # POD modes     
L = np.zeros((ns+1)) #Eigenvalues      
Phi, L, RIC  = POD(u, nr) 
        
#%% Calculating true POD modal coefficients
aTrue = np.zeros((ns+1,nr))
print('Computing true POD coefficients...')
aTrue = PODproj(u,Phi)
#Unifying signs for proper training and interpolation
Phi = Phi/np.sign(aTrue[0,:])
aTrue = aTrue/np.sign(aTrue[0,:])
aTrue = aTrue.T

#%% Galerkin projection precomputed coefficients

b_c = np.zeros((nr))
b_cc = np.zeros((nr))

b_l = np.zeros((nr,nr))
b_lc = np.zeros((nr,nr))

b_nl = np.zeros((nr,nr,nr))
Phid = np.zeros((nx+1,nr))
Phidd = np.zeros((nx+1,nr))


for i in range(nr):
    Phid[:,i] = pade4d(Phi[:,i],dx,nx)
    Phidd[:,i] = pade4dd(Phi[:,i],dx,nx)

umd = pade4d(um,dx,nx)
umdd = pade4dd(um,dx,nx)

# constant term
for k in range(nr):
    temp = -um*umd + (1/Re)*umdd
    b_c[k] = np.dot( temp.T, Phi[:,k] ) 

# linear term   
for k in range(nr):
    for i in range(nr):
        temp = -um*Phid[:,i] - Phi[:,i]*umd + (1/Re)*Phidd[:,i]
        b_l[i,k] = np.dot(temp.T , Phi[:,k]) 
                 
# nonlinear term 
for k in range(nr):
    for j in range(nr):
        for i in range(nr):
            temp = Phi[:,i]*Phid[:,j]
            b_nl[i,j,k] = - np.dot( temp.T, Phi[:,k] ) 
            
#% closure terms
# constant term
for k in range(nr):
    temp = umdd
    b_cc[k] = np.dot( temp.T, Phi[:,k] ) 


# linear term   
for k in range(nr):
    for i in range(nr):
        temp = Phidd[:,i]
        b_lc[i,k] = np.dot(temp.T , Phi[:,k]) 



#%% Generate Observations from a twin experiment
sig2 = 0.01
sig = np.sqrt(sig2)
R = sig**2 * np.eye(nr)
Ri = np.linalg.inv(R)

loc = np.arange(1,9,1)*int(nx/8) #locations of observations

Nz = 2 #number of observations per assimilation window
t_wind = 0.5 #assimilation window
N = int(t_wind/dt) #number of timesteps per assimilation window
Ntz = int(N/Nz) #number of timesteps between observations

ind = np.arange(Ntz,N+1,Ntz)

Phi_s = Phi[loc,:]
uObs = uFOM[loc,:]
uObs = uObs[:,ind] + np.random.normal(0,sig,uObs[:,ind].shape) 

z = uObs- um[loc].reshape(-1,1) #mean subtraction at measurement locations


#%% FSM eddy viscosity estimation itertations
a0 = aTrue[:,0]
a0 = a0.reshape(-1,1)
nue_b = 0*nu #some initial guess for eddy viscosity

nue = nue_b

max_iter= 50
for jj in range(max_iter):
    U = np.eye(nr,nr)
    V = np.zeros((nr,1))

    H = np.zeros((1,1))
    e = np.zeros((1,1))
    W = np.zeros((1,1)) #weighting matrix
    k = 0
    a = a0
    for i in range(N):
        Da , Dnue = DM(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a)
        #U = DM_a(u,mu) @ U
        V = Da @ V + Dnue
        a = M(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, a, dt)
        
        if i+1 == ind[k]:
            Hk = Dh(a) @ V
            H = np.vstack((H,Hk))
            ek = h(a) - (z[:,k]).reshape(-1,1)
            e = np.vstack((e,ek))
            W = block_diag(W,Ri)
            k = k+1
            
    H = np.delete(H, (0), axis=0)
    e = np.delete(e, (0), axis=0)
    W = np.delete(W, (0), axis=0)
    W = np.delete(W, (0), axis=1)
    
    # solve weighted least-squares
    W1 = np.sqrt(W) 
    dc = np.linalg.lstsq(W1@H, -W1@e, rcond=None)[0]
    print(dc)
    nue = nue + dc
    if np.linalg.norm(dc) <= 1e-6:
        break
   
#%% Solving ROM        
     
aGP = np.zeros((nr,ns+1))
aGPFSM = np.zeros((nr,ns+1))
# initial condition
aGP[:,0] = a0.ravel()
aGPFSM[:,0] = a0.ravel()
# time integration using fourth-order Runge Kutta method
for k in range(1,ns+1):    
    aGP[:,k] = M(nr, nu, 0, b_c, b_cc, b_l, b_lc, b_nl, aGP[:,k-1],dt).ravel() #solving GROM without closure          
    aGPFSM[:,k] = M(nr, nu, nue, b_c, b_cc, b_l, b_lc, b_nl, aGPFSM[:,k-1],dt).ravel()  #solving GROM with FSM eddy viscosity closure
    
    
#%% Reconstruction
uPOD = PODrec(aTrue.T,Phi) + um.reshape(-1,1)#Reconstruction   
uGP = PODrec(aGP.T,Phi) + um.reshape(-1,1)#Reconstruction   
uGPFSM = PODrec(aGPFSM.T,Phi) + um.reshape(-1,1) #Reconstruction   


def RMSE(ua,ub):
    er = (ua-ub)**2
    er = np.mean(er)
    er = np.sqrt(er)
    return er

# Computing RMSE
LPOD = np.zeros(ns+1)
LGP = np.zeros(ns+1)
LGPFSM = np.zeros(ns+1)

for i in range(ns+1):  
    LPOD[i] = RMSE(uFOM[:,i] , uPOD[:,i])
    LGP[i] = RMSE(uFOM[:,i] , uGP[:,i])
    LGPFSM[i] = RMSE(uFOM[:,i] , uGPFSM[:,i])
    
#%% Plotting   
import matplotlib

matplotlib.rc('text', usetex=True)

matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 26}

matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.default'] = 'it'

#%% Figure 8

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(nrows=int(nr/2),ncols=2, figsize=(8,7))
ax = ax.flat
for k in range(nr):
    ax[k].plot(t,aTrue[k,:], label=r'\bf{True Projection}', color = 'C0', linewidth=3)
    ax[k].plot(t,aGP[k,:], label=r'\bf{GROM}',linestyle='-.', color = 'C1', linewidth=3)
    ax[k].plot(t,aGPFSM[k,:], label=r'\bf{GROM-FSM}',linestyle='--', color = 'C2', linewidth=3)
    ax[k].plot(t[ind],z[k,:],'o', label=r'\bf{From Observations}', color = 'k', \
               fillstyle = 'none', markersize = 7, markeredgewidth = 2)
    ax[k].set_xlabel(r'$t$',fontsize=20,labelpad = -12)
    ax[k].set_ylabel(r'$a_{'+str(k+1) +'}(t)$',fontsize=20)
    
ax[0].set_ylim([-30,30])
ax[1].set_ylim([-20,20])
ax[2].set_ylim([-20,20])
ax[3].set_ylim([-15,15])
ax[4].set_ylim([-15,15])
ax[5].set_ylim([-17,17])
ax[6].set_ylim([-20,20])
ax[7].set_ylim([-28,28])

ax[0].legend(loc="center", bbox_to_anchor=(1.25,1.7), ncol =2, fontsize = 17, handletextpad=0.4 ,columnspacing=2)
          
           
fig.subplots_adjust(hspace=0.9, wspace=0.5)

plt.savefig('./Plots/fig8.png', dpi = 500, bbox_inches = 'tight')
fig.show()


#%% Figure 9

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 26}

matplotlib.rc('font', **font)


fig, ax = plt.subplots(nrows=2,ncols=1, figsize=(8,16))
ax = ax.flat

ax[0].plot(x,uFOM[:,-1], label=r'\bf{FOM}', color = 'k',linewidth=3)
ax[0].plot(x,uPOD[:,-1], label=r'\bf{True Projection}', color = 'C0', linewidth=3)
ax[0].plot(x,uGP[:,-1],'-.', label=r'\bf{GROM}', color = 'C1', linewidth=3)
ax[0].plot(x,uGPFSM[:,-1],'--', label=r'\bf{GROM-FSM}', color = 'C2', linewidth=3)
ax[0].set_xlabel(r'$x$', fontsize=26)
ax[0].set_ylabel(r'$u(x)$', fontsize=26)
ax[0].legend(fontsize=20)


ax[1].plot(t,LPOD, label=r'\bf{True Projection}', color = 'C0', linewidth=3)
ax[1].plot(t,LGP, label=r'\bf{GROM}',linestyle='-.', color = 'C1', linewidth=3)
ax[1].plot(t,LGPFSM, label=r'\bf{GROM-FSM}',linestyle='--', color = 'C2', linewidth=3)
ax[1].set_xlabel(r'$t$',fontsize=26)
ax[1].set_ylabel(r'$RMSE(t)$',fontsize=26)
ax[1].legend(fontsize=20)

#fig.subplots_adjust(bottom=0.15,hspace=0.3, wspace=0.35)

plt.savefig('./Plots/fig9.png', dpi = 500, bbox_inches = 'tight')

fig.show()
