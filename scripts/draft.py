import numpy as np
from enum import IntEnum

uf = hover_thrust/(g*cos(phi)*cos(theta)) * (param.k_ze*e_dz+ref.ref_accel.linear.z+g+param.k_zs*tanh(param.k_zt*s_z))
ux = param.hover_thrust/(uf*g)*(param.k_xe*e_dx+ref.ref_accel.linear.x+param.k_xs*tanh(param.k_xt*s_x))
