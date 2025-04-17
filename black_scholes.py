import numpy as np
from scipy.stats import norm
Normes = norm.cdf

def BS_CALL(S, K, T, r, q, sigma):
    
    d1 = (np.log(np.abs(S/K)) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q*T) * Normes(d1) - K * np.exp(-r*T)* Normes(d2)

def BS_PUT(S, K, T, r, q, sigma):
    
    d1 = (np.log(np.abs(S/K)) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*Normes(-d2) - S*np.exp(-q*T)*Normes(-d1)