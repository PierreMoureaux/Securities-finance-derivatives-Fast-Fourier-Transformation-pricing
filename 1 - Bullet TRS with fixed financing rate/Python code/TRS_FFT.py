# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:16:19 2023

@author: pmoureaux
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import time

def TRSPriceCOSMthd(cf,payOff,S0,r,tau,K, sTRS,N,L):
    # cf   - characteristic function as a function
    # payOff   -  Preceiv for performance receiver TRS and Ppay for performance payer TRS
    # S0   - Initial stock price
    # r    - interest rate (constant)
    # tau  - time to maturity
    # K    - list of strikes
    # N    - Number of expansion terms
    # L    - size of truncation domain (typ.:L=8 or L=10)  
        
    # reshape K to a column vector --> Ktild including TRS financing spread
    Ktild = np.array(K).reshape([len(K),1]) + sTRS
    
    x0 = np.log(S0 / Ktild)   
    
    # truncation domain
    a = 0.0 - L * np.sqrt(tau)
    b = 0.0 + L * np.sqrt(tau)
    
    # sumation from k = 0 to k=N-1
    j = np.linspace(0,N-1,N).reshape([N,1])  
    u = j * np.pi / (b - a);  

    # Determine coefficients for TRS Prices  
    TRS_j = TRSCoefficients(payOff,a,b,j)
       
    mat = np.exp(1j * np.outer((x0 - a) , u))

    temp = cf(u) * TRS_j
    temp[0] = 0.5 * temp[0]    
    
    value = np.exp(-r * tau) * Ktild * np.real(mat.dot(temp))
         
    return value

""" 
Determine coefficients for TRS Prices 
"""
def TRSCoefficients(payOff,a,b,j):
    if str(payOff).lower()=="preceiv":                  
        coef = Chi_Psi(a,b,j)
        Chi_j = coef["chi"]
        Psi_j = coef["psi"]
        if a < b and b < 0.0:
            TRS_j = np.zeros([len(j),1])
        else:
            TRS_j = 2.0 / (b - a) * (Chi_j - Psi_j)    
    elif str(payOff).lower()=="ppay":
        coef = Chi_Psi(a,b,j)
        Chi_j = coef["chi"]
        Psi_j = coef["psi"]
        TRS_j = 2.0 / (b - a) * (- Chi_j + Psi_j)               
    return TRS_j   

def Chi_Psi(a,b,j):
    psi = np.array(j)
    psi[1:] = 0
    psi[0] = b - a
    
    chi = 1.0 / (1.0 + np.power((j * np.pi / (b - a)) , 2.0)) * (np.cos(j * np.pi) * np.exp(b)-  np.exp(a))
    
    value = {"chi":chi,"psi":psi }
    return value
    

def TRS_Price(payOff,S_0,K,sTRS,tau,r):
    #TRS option price under risk-neutral expectations
    payOff = str(payOff).lower()
    K = np.array(K).reshape([len(K),1])
    if payOff == "preceiv":
        value = S_0 - np.exp(-r * tau)*(K - sTRS)
    elif payOff == "ppay":
        value = -(S_0 - np.exp(-r * tau)*(K - sTRS))
    return value

def mainCalculation():
    
    payOff = "preceiv"
    S0 = 100.0
    r = 0.1
    tau = 0.1
    sigma = 0.25
    K = [80.0, 90.0, 100.0, 110, 120.0]
    N = 4*32
    L = 10
    sTRS = 0.05
    
    # Definition of the characteristic function for the asset dynamic under GBM, this is an input for the COS method
    cf = lambda u: np.exp((r - 0.5 * np.power(sigma,2.0)) * 1j * u * tau - 0.5 
                          * np.power(sigma, 2.0) * np.power(u, 2.0) * tau)
    
    # Timing results 
    NoOfIterations = 100
    time_start = time.time() 
    for k in range(0,NoOfIterations,1):
        val_COS = TRSPriceCOSMthd(cf,payOff,S0,r,tau,K,sTRS,N,L)
    time_stop = time.time()
    print("It took {0} seconds to price.".format((time_stop-time_start)/float(NoOfIterations)))
    
    # evaluate analytical through risk-neutral expectation
    val_Exact = TRS_Price(payOff,S0,K,sTRS,tau,r)
    plt.plot(K,val_COS, 'b')
    plt.plot(K,val_Exact,'--',color='r')
    plt.xlabel("strike, K")
    plt.ylabel("TRS Price")
    plt.legend(["COS Price","BS model"])
    plt.grid()    
    
    #Price computation
    price = []
    for i in range(0,len(K)):
        price.append(val_COS[i][0])
        print("FFT Price for strike {0} is equal to {1:.2E}".format(K[i],price[i]))
    
    #Error comuputation
    error = []
    for i in range(0,len(K)):
        error.append(np.abs(val_COS[i]-val_Exact[i])[0])
        print("Abs error for strike {0} is equal to {1:.2E}".format(K[i],error[i]))
        
mainCalculation()