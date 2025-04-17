import numpy as np
import typing as tp
from numpy import typing as npt
from scipy.stats import norm
Normes = norm.cdf

class Model:
    """ Class for model params description
    gives us dynamics on stock """
    
    def __init__(self,
                 r: tp.Callable[npt.NDArray, npt.NDArray],
                 q: tp.Callable[npt.NDArray, npt.NDArray],
                 sigma: tp.Callable[npt.NDArray, npt.NDArray],
                 S_min: np.float64,
                 S_max: np.float64,
                 N_steps: int,
                 times: npt.NDArray[np.float64]):
        
        self.r = r
        self.q = q
        self.sigma = sigma
        self.S_min = S_min
        self.S_max = S_max
        self.N_steps = N_steps
        self.times = times
        
        self.stock_grid = np.linspace(S_min, S_max, N_steps)
        self.delta_s = (S_max - S_min) / N_steps
        
        self.r_grid = r(times)
        self.q_grid = q(times)
        self.sigma_grid = sigma(times)
        
        
    def rollback(self, prices: "Slice", time_to: int) -> None:
        """- What do we say to abstract classes?
           - Not today...
           
           Class realized according to Prof.Kramkov"""
    
        assert time_to < prices.iTime, 'Wrong order!'
        
        time = prices.iTime
        
        while time > time_to:
            
            delta_t = self.times[time] - self.times[time - 1]
            
            alpha = self.sigma_grid[time]**2 * self.stock_grid**2 * delta_t / (2 * self.delta_s**2)
            beta = (self.r_grid[time] - self.q_grid[time]) * self.stock_grid * delta_t / (2 * self.delta_s)
            
            d = 1 + self.r_grid[time] * delta_t + 2 * alpha
            
            l = beta - alpha
            
            u = -beta - alpha
            
            prices.current_price[self.N_steps - 1] = 2 * prices.current_price[self.N_steps - 2]
            prices.current_price[self.N_steps - 1] -= prices.current_price[self.N_steps - 3]
            prices.current_price[0] = 2 * prices.current_price[2] - prices.current_price[1]
            
            transform_matrix = np.diag(d[1:self.N_steps - 1])
            transform_matrix += np.diag(l[2:self.N_steps - 1], -1)
            transform_matrix += np.diag(u[1:self.N_steps - 2], 1) 
            transform_matrix[0][0] += l[1]

            transform_matrix[self.N_steps - 3][self.N_steps - 3] += u[self.N_steps - 2]

            prices.current_price[1:self.N_steps - 1] = np.linalg.solve(transform_matrix,
                                                                       prices.current_price[1:self.N_steps - 1])
            
            prices.iTime -= 1
            time -=1


class Slice:
    """ here we define params for num approx 
    stock grid is uniform """
    
    def __init__(self,
                model: Model):
        
        self.current_price = np.linspace(model.S_min, 
                                         model.S_max, 
                                         model.N_steps)
        self.iTime = model.times.shape[0] - 1
        self.model = model
        
        
    def rollback(self, time_to: int) -> None:
        
        self.model.rollback(self, time_to)
