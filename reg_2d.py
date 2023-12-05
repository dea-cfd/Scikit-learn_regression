print(__doc__)


# Author: D. Chibouti (ChD)
import os
import sys
import time
import scipy
import random
import sklearn
import numpy as np
import numpy.polynomial.polynomial as nppol

from decimal              import *
from math                 import pi
from itertools            import product
from matplotlib           import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.special        import erf

from sklearn                          import preprocessing
from sklearn.preprocessing            import StandardScaler
from sklearn.preprocessing            import MinMaxScaler,minmax_scale
from sklearn.metrics                  import r2_score
from sklearn.linear_model             import BayesianRidge,ARDRegression,LinearRegression
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel


os.system("cp base0.txt base.txt")
os.system("rm -f *.png *.dat")
os.system("rm -f save_fct_ML")
os.system("rm -f save_fct_f")
os.system("rm -f save_fct_dfdx1")
os.system("rm -f save_fct_dfdx2")
os.system("rm -f convergence_GP_DF1")
os.system("rm -f convergence_GP_DF2")
# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
nRestart=100
nLearn=30

Nx1=51;Nx2=51

random_num=0   # 0: no random ; 1: get a random number

show_plot=1
idKernel=-1    # 1 RBF, 2 Matern32, 3 Matern52, -1; Bayesien Basis
# ----------------------------------------------------------------------

amp_noise=0.01


p1=ORDRE_P
p2=ORDRE_U

base=VAL_BASE


# ----------------------------------------------------------------------
# Init random numbers
# ----------------------------------------------------------------------
np.random.seed(1)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Basis function def for Bayesian Regression
# ----------------------------------------------------------------------
# def phi(x1,x2):
#     var_x1=1./x1[0]  #1/(1+np.exp(-x1[0]))
#     var_x2=x2[0]#**(0.5)
#     tab1=np.array([1, var_x1, var_x1**2]).ravel()
#     tab2=np.array([1, var_x2]).ravel()
#     #tab2=np.array([1]).ravel()
#     tab3=np.array([])
#     #tab3=np.concatenate((tab1,tab1),axis=None)
#     for i in range(tab1.shape[0]):
#         for j in range(tab2.shape[0]):
#             tmp=np.array([np.multiply(tab1[i],tab2[j])])
#             tab3=np.concatenate((tab3,tmp),axis=None)
#     return tab3
# # ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def phi(x1,x2):
    if base > 0:
        var_x1=x1[0]  #1/(1+np.exp(-x1[0]))
    else:
        var_x1=1./x1[0]

    var_x2=x2[0]#**(0.5)
    tab1=(np.array([var_x1**i for i in range(p1+1) ])).ravel()
    tab2=(np.array([var_x2**i for i in range(p2+1) ])).ravel()
    tab3=np.array([])
    for i in range(p1+1):
        for j in range(p2+1):
            tmp=np.array([np.multiply(tab1[i],tab2[j])])
            tab3=np.concatenate((tab3,tmp),axis=None)
    return tab3
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# function definition
# ----------------------------------------------------------------------
def f(x1,x2):

    #return 1 + 1./x1 + 4./x1/x1 - 4./x1/x1/x1 + x2 #+ 5./x1/x1/x1

    # valeurs directes

    sigma_u=1
    uK=x2
    uP=0
    h=2.05e-7
    mu=1.78e-5
    p=x1
    const_r=296.8
    T=300

    # valeurs script sh
    #sigma_u=VAL_SIG
    #uK=x2
    #uP=0
    #h=1
    #mu=1
    #p=x1
    #const_r=1./pi
    #T=2

    lpm=mu/p*np.sqrt(pi*const_r*T/2)

    order_DL=2
    DL=0
    for i in range (order_DL):
        DL=DL+(-lpm)**i

    return (uK-uP)/h * DL

    #print("lpm/h=",lpm/h)
    #return x2*(1.+1./x1+1./x1/x1)
    # return 1+x2/x1+x2+1/x1  # err 1e-9 base ip1u1
    return (uK-uP)/(h+sigma_u*lpm) # err 1e-3 base ip1u1
# ----------------------------------------------------------------------








# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
print("Here we go! ChD makes Machine learning fun :)" )
# ----------------------------------------------------------------------
for iLearn in range(nLearn):
    print("# ------------------------------------------------------" )
    print("#  Learning number: \033[33m ",iLearn+1, "\033[0m       " )
    print("# ------------------------------------------------------" )
    # ----------------------------------------------------------------------
    # Database X reading
    # ----------------------------------------------------------------------
    X_read=np.loadtxt('base.txt')
    # ----------------------------------------------------------------------
    # training target
    # ----------------------------------------------------------------------
    t1 = f(X_read[:,0],X_read[:,1]).ravel()
    # ----------------------------------------------------------------------
    # Noise
    # ----------------------------------------------------------------------
    #noise = np.random.normal(0, 1e-2)
    noise = np.random.normal(0, amp_noise*t1, t1.shape)
    t1  += noise

    #print("t1",t1)
    #print("noise",noise)
    #quit() # ChD
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Input space discretization Nx1,Nx2 and array for output
    # ----------------------------------------------------------------------
    x1 = np.linspace(X_read[:,0].min(), X_read[:,0].max(), num=Nx1)
    x2 = np.linspace(X_read[:,1].min(), X_read[:,1].max(), num=Nx2)
    # ----------------------------------------------------------------------
    x_test = np.array(list(product(x1, x2)))
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Scale X matrix and scaling properties
    # ----------------------------------------------------------------------
    # scaler    = preprocessing.MinMaxScaler()#feature_range=(0.5, 1))
    # scaler_y  = preprocessing.MinMaxScaler()
    # scaler_phi= preprocessing.MinMaxScaler()

    scaler    = preprocessing.StandardScaler()#feature_range=(0.5, 1))
    scaler_y  = preprocessing.StandardScaler()
    scaler_phi= preprocessing.StandardScalerr()
    # ----------------------------------------------------------------------
    t         = scaler_y.fit_transform(t1.reshape(-1,1))
    # ----------------------------------------------------------------------
    # No basis scalling
    # ----------------------------------------------------------------------
    X         = X_read
    X_test    = x_test
    # ----------------------------------------------------------------------
    # Observations
    # ----------------------------------------------------------------------
    y_true    = f(x1,x2)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Design Matrix (#200723)
    # ----------------------------------------------------------------------
    if idKernel > 0:
        Phi           = X               # scaler.fit_transform(X_read)
        Phi_test      = X_test          # scaler.transform(x_test)
        save_Phi_test = Phi_test
        # ----------------------------------------------------------------------
        # Phi= scaler_phi.fit_transform(Phi)
        # Phi_test= scaler_phi.fit_transform(Phi_test)
        # ----------------------------------------------------------------------
    elif idKernel < 0:
        # ----------------------------------------------------------------------
        size=phi(np.array([1]),np.array([1])).shape[0]
        Phi=np.zeros((X.shape[0], size))
        for i in range (X.shape[0]):
            tmp=np.array([phi(
                np.array([X[i,0]]),
                np.array([X[i,1]]))]).ravel()
            Phi[i,:]=tmp
        # ----------------------------------------------------------------------
        Phi_test=np.zeros((x_test.shape[0], size))
        for i in range (x_test.shape[0]):
            tmp=np.array([phi(
                np.array([X_test[i,0]]),
                np.array([X_test[i,1]]))]).ravel()
            Phi_test[i,:]=tmp
        # ----------------------------------------------------------------------
        #print('Phi_test',(Phi_test))
        #print('Phi_test',np.size(Phi_test))
        save_Phi_test=Phi_test
        # nedded for phi_min, phi_max and Dphi
        # scalling for NS
        # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    Phi      = scaler_phi.fit_transform(Phi)
    Phi_test = scaler_phi.transform(Phi_test)

    #print("Phi",Phi)

    # Phi_test= scaler_phi.fit_transform(Phi_test)
    # quit('done !')
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # fitting and prediction
    # Bayesian or GP
    # ----------------------------------------------------------------------
    if idKernel > 0:
        # ----------------------------------------------------------------------
        # Kernel choice and limits
        # ----------------------------------------------------------------------
        if idKernel == 1:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) *\
                RBF(np.ones(X.shape[1]), (1e-2, 1e2))
        elif idKernel == 2:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) *\
                Matern(np.ones(X.shape[1]), (1e-2, 1e2), nu=1.5)
        elif idKernel == 3:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) *\
                Matern(np.ones(X.shape[1]), (1e-2, 1e2), nu=2.5)
        else:
            quit("Quit, no kernel")
        # ----------------------------------------------------------------------


        # ----------------------------------------------------------------------
        # GP Regression method
        # ----------------------------------------------------------------------
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2 ** 2,
                                      n_restarts_optimizer=nRestart)
        # ----------------------------------------------------------------------
        # Fit with optimization of kernel parameters
        # ----------------------------------------------------------------------
        gp.fit(Phi, t)
        # ----------------------------------------------------------------------
        # Fit properties
        # ----------------------------------------------------------------------
        params = gp.kernel_.get_params()
        print("gp.params_:\n",params)
        #----------------------------------------------------------------------
        k1_c = params['k1__constant_value']
        k2_l = params['k2__length_scale']
        print("k1_c:",k1_c)
        print("k2_l:",k2_l)
        print(k2_l[0],k2_l[1])
        l_x1=k2_l[0]
        l_x2=k2_l[1]
        #quit()
        #----------------------------------------------------------------------
        #for hyperparameter in kernel.hyperparameters: print('hyperP:',hyperparameter)
        #print("length_scale=\n",gp.kernel_.get_params((length_scale)))
        #print("gp.kernel_:\n",gp.kernel_)
        #print("gp.score:\n",gp.score)
        print("gp.alpha_:\n",gp.alpha_)
        # ----------------------------------------------------------------------
        # Prediction on x_test grid
        # ----------------------------------------------------------------------
        y_pred, sigma = gp.predict(Phi_test, return_std=True)
        y_pred=scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
        sigma =scaler_y.inverse_transform(sigma.reshape(-1,1)).ravel()
        print('y_pred',y_pred)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # matrice de k
        # ----------------------------------------------------------------------
        list_alpha=gp.alpha_[:,0]
        print('list_alpha',list_alpha,X[:,0].size)
        # ----------------------------------------------------------------------
        # if idKernel == 1:   # RBF X: known values in base.txt no scaling
        #     f_x=gp_RBF(X[:,0],X[:,1],list_alpha,X[:,0].size,\
        #                l_x1,l_x2,k1_c,k=0)
        #     df_x1=gp_RBF(X[:,0],X[:,1],list_alpha,X[:,0].size,\
        #                  l_x1,l_x2,k1_c,k=1)
        #     df_x2=gp_RBF(X[:,0],X[:,1],list_alpha,X[:,0].size,\
        #                  l_x1,l_x2,k1_c,k=2)
        #     #print('df_x1=',df_x1)
        #     #print('df_x2=',df_x2)
        #     #quit()
        # ----------------------------------------------------------------------

    elif idKernel < 0:
        # ----------------------------------------------------------------------
        # Ridge Bayesian Regression method
        # ----------------------------------------------------------------------
        reg = BayesianRidge(n_iter=10000, tol=1e-6,\
                            fit_intercept=False, compute_score=True, verbose=True)
        reg = ARDRegression(n_iter=10000, tol=1e-6,\
                            fit_intercept=False, compute_score=True, verbose=True)
       #reg = linear_model.LinearRegression()
        # ----------------------------------------------------------------------
        # Fit
        # ----------------------------------------------------------------------
        #init = [1.0, 1e-3]
        #reg.set_params(alpha_init=init[0], lambda_init=init[1])

        reg.fit(Phi, t.ravel())
        # ----------------------------------------------------------------------
        # Fit properties
        # ----------------------------------------------------------------------
        print("reg.scores_:",reg.scores_)
        print("reg.alpha_ :",reg.alpha_)
        print("reg.lambda_:",reg.lambda_)
        print("reg.coef_  :",reg.coef_)
        #print("n_iter_    :",reg.n_iter_) commenté si ARD
        #print(f'Coeff1 {reg.coef_[0]:.3f}.')
        #print(f'Coeff2 {reg.coef_[1]:.3f}.')
        print("reg.get_params():",reg.get_params())
        # ----------------------------------------------------------------------
        # Prediction on x_test grid
        # ----------------------------------------------------------------------
        y_pred, sigma = reg.predict(Phi_test, return_std=True)
        y_pred=scaler_y.inverse_transform(y_pred.reshape(-1,1)).ravel()
        # print('y_pred',y_pred)
        # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Plot Output (p: plot)
    # ----------------------------------------------------------------------
    x1_plot  = x_test[:,0].reshape(Nx1,Nx2)
    x2_plot  = x_test[:,1].reshape(Nx1,Nx2)
    yp_plot  = np.reshape(y_pred,(Nx1,Nx2))
    ypp_plot = np.reshape(y_pred+1.96*sigma,(Nx1,Nx2))
    ypm_plot = np.reshape(y_pred-1.96*sigma,(Nx1,Nx2))
    S_plot   = np.reshape(sigma,(Nx1,Nx2))
    # ----------------------------------------------------------------------
    y_true   = f(x_test[:,0],x_test[:,1])
    Fp_true  = np.reshape(y_true,(Nx1,Nx2))

    #print("Fp_true",Fp_true.shape)
    #a = np.arange(9) - 4
    #print(a)
    #b = a.reshape((3, 3))
    #print(b)
    #c = b.reshape((3*3))
    #print(c)
    # ----------------------------------------------------------------------
    # norm L1, L2, inf
    # ----------------------------------------------------------------------
    from numpy import array,inf
    from numpy.linalg import norm

    #print(norm(a,2),norm(b,2))
    #print(norm(a,1),norm(b,1))
    #print(norm(a,inf),norm(b,inf))
    #print(np.linalg.norm(a),np.linalg.norm(b))
    #print("shape",Fp_true.shape)


    eps_L1 = norm((Fp_true-yp_plot),1)  /norm((Fp_true),1)
    eps_L2 = norm((Fp_true-yp_plot),2)  /norm((Fp_true),2)
    eps_inf= norm((Fp_true-yp_plot),inf)/norm((Fp_true),inf)

    print("\033[36m eps_L1      :", X_read.shape[0]+1,eps_L1 )
    print("\033[36m eps_L2      :", X_read.shape[0]+1,eps_L2 )
    print("\033[36m eps_inf     :", X_read.shape[0]+1,eps_inf,'\033[0m' )
    #quit()
    # ----------------------------------------------------------------------
    # plotting 2D
    # ----------------------------------------------------------------------
    if show_plot:

        top=max(yp_plot.max(),Fp_true.max())
        bottom=min(yp_plot.min(),Fp_true.min())

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x1_plot, x2_plot, yp_plot,
                        color="b", rstride=1, cstride=1, alpha=0.25)
        ax.plot_surface(x1_plot, x2_plot, Fp_true,
                        color="r", rstride=1, cstride=1, alpha=0.25)
        # ax.contourf(x1_plot, x2_plot, S_plot,
        #             zdir='z', offset=1.1*y_pred.min(), cmap=cm.coolwarm)
        ax.contourf(x1_plot, x2_plot, S_plot,
                    zdir='z', offset=1.1*y_pred.min(), #     bottom, # rstride=1, cstride=1, alpha=0.5,
                    cmap=cm.jet) #colormap_lim=[1.e-5, 2.e-1])#cm.coolwarm #200723)
        ax.scatter(X_read[:,0], X_read[:,1], t1,
                   s=50, c='b', marker='o') #, label='Observations')
        #plt.legend(loc='upper left')
        ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        ax.set_zlim3d(bottom,top)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f(x_1,x_2)$')
        #ax.autoscaleZon = True

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # New point
    # ----------------------------------------------------------------------
    test=np.where(S_plot==np.amax(S_plot))
    test2=list(zip(test[0], test[1]))

    i1=test2[0][0]
    i2=test2[0][1]

    print('i1,i2 ',test2[0][0],test2[0][1])
    x1_new=x1[i1]
    x2_new=x2[i2]


    #x1_new=random.uniform(X_read[0,0],X_read[1,0])
    #x2_new=random.uniform(X_read[0,1],X_read[1,1])

    print("\033[31m Next point should be :",x1[i1],x2[i2],'\033[0m')
    print("\033[32m Max_sigma            :",X_read.shape[0]+1,np.amax(S_plot),'\033[0m')
    #quit()
    # ----------------------------------------------------------------------
    if show_plot:
        #ax.scatter(x1[i1], x2[i2], f(x1[i1], x2[i2]),
        #           s=50, c='r', marker='o') #, label='New point')
        ax.scatter(x1_new,x2_new, f(x1_new,x2_new),
                   s=75, c='r', marker='x', label='New point')
        #plt.legend(loc='upper left')
        ax.legend(loc='upper left', fontsize=18)
        #title = "$\\sigma$={:.1e} ; $\\epsilon_1$={:.1e}; $\\epsilon_2$={:.1e};\
        #$\\epsilon_i$={:.1e}".format(np.amax(S_plot),eps_L1,eps_L2,eps_inf)
        title = "$\\sigma_N$={:.1e} ; $\\epsilon_2$={:.1e}"\
            .format(np.amax(S_plot),eps_L2)
        ax.set_title(title, fontsize=18,color="blue",loc='left')
        #plt.show(block=False)
        #plt.pause(2)
        plt.savefig("f_i"+str(iLearn)+".png", dpi=150)
        plt.close()
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Increment base
    # ----------------------------------------------------------------------
    with open("base.txt", "a") as myfile:
        #string="{} {}\n".format(x1[i1],x2[i2])
        string="{} {}\n".format(x1_new,x2_new)
        myfile.write(string)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Error computation L2 norm
    # ----------------------------------------------------------------------

    print("\033[33m convergence :", X_read.shape[0]+1,
          np.linalg.norm(Fp_true-yp_plot)/np.linalg.norm(Fp_true),'\033[0m')
    # ----------------------------------------------------------------------
    f_convergence=np.linalg.norm(Fp_true-yp_plot)/np.linalg.norm(Fp_true)
    # ----------------------------------------------------------------
    file_convergence= open("f_convergence_sigma.dat","a")
    file_convergence.write(str(X_read.shape[0]+1)+' '+str(f_convergence)\
                           +' '+str(np.amax(S_plot))+"\n")
    file_convergence.close()

    print("Err2", eps_L2, f_convergence)

# ----------------------------------------------------------------------
# Regression lineaire #180619
# ----------------------------------------------------------------------


X_read=np.loadtxt('base.txt')

t1 = f(X_read[:,0],X_read[:,1]).ravel()

noise = np.random.normal(0, amp_noise*t1, t1.shape)
t1  += noise

print("t1   ",t1)
print("noise",noise)


x1 = np.linspace(X_read[:,0].min(), X_read[:,0].max(), num=Nx1)
x2 = np.linspace(X_read[:,1].min(), X_read[:,1].max(), num=Nx2)
x_test = np.array(list(product(x1, x2)))

scaler    = preprocessing.MinMaxScaler()#feature_range=(0.5, 1))
scaler_y  = preprocessing.MinMaxScaler()
scaler_phi= preprocessing.MinMaxScaler()

t         = scaler_y.fit_transform(t1.reshape(-1,1))

X         = X_read
X_test    = x_test

y_true    = f(x1,x2)


size=phi(np.array([1]),np.array([1])).shape[0]

Phi=np.zeros((X.shape[0], size))
for i in range (X.shape[0]):
    tmp=np.array([phi(
        np.array([X[i,0]]),
        np.array([X[i,1]]))]).ravel()
    Phi[i,:]=tmp
    # ----------------------------------------------------------------------
Phi_test=np.zeros((x_test.shape[0], size))
for i in range (x_test.shape[0]):
    tmp=np.array([phi(
        np.array([X_test[i,0]]),
        np.array([X_test[i,1]]))]).ravel()
    Phi_test[i,:]=tmp

# ----------------------------------------------------------------------
Phi      = scaler_phi.fit_transform(Phi)
Phi_test = scaler_phi.transform(Phi_test)



# ----------------------------------------------------------------------
# Linear Regression method #180619
# ----------------------------------------------------------------------
reg = LinearRegression()
reg.fit(Phi, t.ravel())
# ----------------------------------------------------------------------
# Fit properties
# ----------------------------------------------------------------------
print("reg.coef_  :",reg.coef_)
print("reg.scores_:",reg.score(Phi, t.ravel()))
# ----------------------------------------------------------------------
# Prediction on x_test grid
# ----------------------------------------------------------------------
y_pred_lin = reg.predict(Phi_test)
y_pred_lin = scaler_y.inverse_transform(y_pred_lin.reshape(-1,1)).ravel()
        # print('y_pred',y_pred)
        # ----------------------------------------------------------------------

y_true   = f(x_test[:,0],x_test[:,1])

eps_L1 = norm((y_pred_lin-y_true),1)  /norm((y_true),1)
eps_L2 = norm((y_pred_lin-y_true),2)  /norm((y_true),2)
eps_LInf = norm((y_pred_lin-y_true),inf)  /norm((y_true),inf)

print("Err brute", norm((y_pred_lin-y_true),inf), norm((y_pred_lin-y_true),1), norm((y_pred_lin-y_true),2))
print("Normalisation", norm((y_true),inf), norm((y_true),1), norm((y_true),2))
print("Err relative, fit lineaire",eps_LInf,eps_L1,eps_L2)

iLearn=1000000

if show_plot:

    yp_lin_plot  = np.reshape(y_pred_lin,(Nx1,Nx2))


    top=max(yp_plot.max(),Fp_true.max())
    bottom=min(yp_plot.min(),Fp_true.min())

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1_plot, x2_plot, yp_plot,
                    color="b", rstride=1, cstride=1, alpha=0.25)
    ax.plot_surface(x1_plot, x2_plot, Fp_true,
                    color="r", rstride=1, cstride=1, alpha=0.25)
    ax.plot_surface(x1_plot, x2_plot, yp_lin_plot,
                    color="g", rstride=1, cstride=1, alpha=0.25)
    # ax.contourf(x1_plot, x2_plot, S_plot,
    #             zdir='z', offset=1.1*y_pred.min(), cmap=cm.coolwarm)
    ax.contourf(x1_plot, x2_plot, S_plot,
                zdir='z', offset=1.1*y_pred.min(), #     bottom, # rstride=1, cstride=1, alpha=0.5,
                cmap=cm.jet) #colormap_lim=[1.e-8, 6.e-1])#cm.coolwarm)
    ax.scatter(X_read[:,0], X_read[:,1], t1,
               s=50, c='b', marker='o') #, label='Observations')
    #plt.legend(loc='upper left')
    ax.ticklabel_format(axis="z", style="sci", scilimits=(0,0))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    ax.set_zlim3d(bottom,top)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')
    #ax.autoscaleZon = True
    #ax.scatter(x1[i1], x2[i2], f(x1[i1], x2[i2]),
    #           s=50, c='r', marker='o') #, label='New point')
    ax.scatter(x1_new,x2_new, f(x1_new,x2_new),
               s=75, c='r', marker='x', label='New point')
    #plt.legend(loc='upper left')
    ax.legend(loc='upper left', fontsize=18)
    #title = "$\\sigma$={:.1e} ; $\\epsilon_1$={:.1e}; $\\epsilon_2$={:.1e};\
        #$\\epsilon_i$={:.1e}".format(np.amax(S_plot),eps_L1,eps_L2,eps_inf)
    title = "$\\sigma_N$={:.1e} ; $\\epsilon_2$={:.1e}"\
        .format(np.amax(S_plot),eps_L2)
    ax.set_title(title, fontsize=18,color="blue",loc='left')
    #plt.show(block=False)
    #plt.pause(2)
    plt.savefig("f_i"+str(iLearn)+".png", dpi=150)
    plt.close()
