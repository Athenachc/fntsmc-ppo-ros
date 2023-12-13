import numpy as np
import os, sys

sys.path.append(os.getcwd() + '/src/adp-smc-uav-ros/scripts/')
sys.path.append(os.getcwd() + '/src/adp-smc-uav-ros/scripts/smc_ctrl/')

from utils import *


class UAV_ROS:
	def __init__(self, m: float = 1.5, g: float = 9.8, kt: float = 1e-3, dt:float = 0.01, time_max: float=60.):
		self.m = m  # 无人机质量
		self.g = g  # 重力加速度
		self.kt = kt  # 平移阻尼系数

		self.x = 0.
		self.y = 0.
		self.z = 0.

		self.vx = 0.
		self.vy = 0.
		self.vz = 0.

		self.phi = 0.
		self.theta = 0.
		self.psi = 0.

		self.p = 0.
		self.q = 0.
		self.r = 0.

		self.dt = dt
		self.n = 0  # 记录走过的拍数
		self.time = 0.  # 当前时间
		self.time_max = time_max  # 每回合最大时间

		'''control'''
		self.throttle = self.m * self.g  # 油门
		self.phi_d = 0.
		self.theta_d = 0.
		'''control'''


	def rk44(self, action: list, uav_state: np.ndarray):
		self.phi_d = action[0]
		self.theta_d = action[1]
		self.throttle = action[2]

		self.x = uav_state[0]
		self.y = uav_state[1]
		self.z = uav_state[2]

		self.vx = uav_state[3]
		self.vy = uav_state[4]
		self.vz = uav_state[5]

		self.phi = uav_state[6]
		self.theta = uav_state[7]
		self.psi = uav_state[8]

		self.p = uav_state[9]
		self.q = uav_state[10]
		self.r = uav_state[11]

		self.n += 1  # 拍数 +1
		self.time += self.dt

	def uav_state_call_back(self):
		return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r])

	def uav_pos_vel_call_back(self):
		return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz])

	def uav_att_pqr_call_back(self):
		return np.array([self.phi, self.theta, self.psi, self.p, self.q, self.r])

	def uav_pos(self):
		return np.array([self.x, self.y, self.z])

	def uav_vel(self):
		return np.array([self.vx, self.vy, self.vz])

	def uav_att(self):
		return np.array([self.phi, self.theta, self.psi])

	def uav_pqr(self):
		return np.array([self.p, self.q, self.r])

	def T_pqr_2_dot_att(self):
		return np.array([[1, np.sin(self.phi) * np.tan(self.theta), np.cos(self.phi) * np.tan(self.theta)],
						 [0, np.cos(self.phi), -np.sin(self.phi)],
						 [0, np.sin(self.phi) / np.cos(self.theta), np.cos(self.phi) / np.cos(self.theta)]])

	def uav_dot_att(self):
		return np.dot(self.T_pqr_2_dot_att(), self.uav_pqr())

	def set_state(self, xx: np.ndarray):
		[self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r] = xx[:]

	def eta(self):
		return np.array([self.x, self.y, self.z])

	def dot_eta(self):
		return np.array([self.vx, self.vy, self.vz])

	def A(self):
		return self.throttle / self.m * np.array([C(self.phi) * C(self.psi) * S(self.theta) + S(self.phi) * S(self.psi),
												  C(self.phi) * S(self.psi) * S(self.theta) - S(self.phi) * C(self.psi),
												  C(self.phi) * C(self.theta)]) - np.array([0., 0., self.g])
