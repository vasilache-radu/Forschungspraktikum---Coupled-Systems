import numpy as np
import scipy.integrate as scipy
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt

import pandas as pd
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)

#Defining the system of coupled equations
def sysODE (t, y, k_s, k_u, m_s, m_u, c, A, omega):

    x1, y1, x2, y2 = y
    z = A* math.sin(omega * t)
    return [y1,
            (-k_s*pow((x1-x2), 3) - c* (y1-y2))/m_s,
            y2,
            (k_s *pow( x1-x2, 3) + c* (y1-y2) + k_u*(z-x2))/m_u] 


#Defining the parameters
k_s=2000
k_u=2000 
m_s=20 
m_u =40
c =600
A =0.1
omega= 2* math.pi

#Set some fixed initial conditions y0 = 0, 0, 0, 0
y0 = 0, 0 , 0, 0

#Run the system (calculate the ODE) for each instance of T
dt = 0.01
t_0=0
t_final=35
t_eval= np.arange(0, 35.01, 0.01)

sol = solve_ivp(sysODE, [t_0, t_final], y0, args=(k_s, k_u, m_s, m_u, c, A, omega), t_eval=t_eval)

#Set the data set ( {x1(t), t} over the time)

complete_data = np.column_stack((A* np.sin(omega*sol.t), sol.y[0]))
data = complete_data[0:int(30/dt)]
validation = complete_data [ int(30/dt)::]

print(data)


model = FROLS(
    order_selection=True,
    n_info_values=3,
    extended_least_squares=False,
    ylag=2,
    xlag=2,
    info_criteria="aic",
    estimator="least_squares",
    basis_function=Polynomial(degree =2)
)

model.fit(X=data[:, 0].reshape(-1, 1), y= data[:, 1].reshape(-1, 1))

prediction = model.predict(X=validation[:, 0].reshape(-1, 1), y=validation[:, 1].reshape(-1, 1))

rrse = root_relative_squared_error(validation[:, 1], prediction)
print(rrse)