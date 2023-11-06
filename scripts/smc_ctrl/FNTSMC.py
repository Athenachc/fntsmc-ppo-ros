import numpy as np


class fntsmc_param:
    def __init__(self):
        self.k1: np.ndarray = np.array([1.2, 0.8, 1.5])
        self.k2: np.ndarray = np.array([0.2, 0.6, 1.5])
        self.alpha: np.ndarray = np.array([1.2, 1.5, 1.2])
        self.beta: np.ndarray = np.array([0.3, 0.3, 0.3])
        self.gamma: np.ndarray = np.array([0.2, 0.2, 0.2])
        self.lmd: np.ndarray = np.array([2.0, 2.0, 2.0])
        self.vel_c= np.array([0.05, 0.05, 0.])
        self.acc_c = np.array([0.0, 0.0, 0.9])
        self.dim: int = 3
        self.dt: float = 0.01
        self.ctrl0: np.ndarray = np.array([0., 0., 0.])

    def print_param(self):
        print('==== PARAM ====')
        print('k1:     ', self.k1)
        print('k2:     ', self.k2)
        print('alpha:  ', self.alpha)
        print('beta:   ', self.beta)
        print('gamma:  ', self.gamma)
        print('lambda: ', self.lmd)
        print('dim:    ', self.dim)
        print('dt', self.dt)
        print('ctrl0:', self.ctrl0)
        print('==== PARAM ====')


class fntsmc_pos:
    def __init__(self, param: fntsmc_param):
        self.k1 = param.k1
        self.k2 = param.k2
        self.alpha = param.alpha
        self.beta = param.beta
        self.gamma = param.gamma
        self.lmd = param.lmd
        self.dt = param.dt
        self.dim = param.dim
        self.vel_c = param.vel_c
        self.acc_c = param.acc_c

        self.sigma_o = np.zeros(self.dim)
        self.dot_sigma_o1 = np.zeros(self.dim)
        self.sigma_o1 = np.zeros(self.dim)
        self.so = self.sigma_o + self.lmd * self.sigma_o1
        self.control = param.ctrl0

    def control_update(self,
                       kp: float,
                       m: float,
                       vel: np.ndarray,
                       e: np.ndarray,
                       de: np.ndarray,
                       d_ref: np.ndarray,
                       dd_ref: np.ndarray,
                       obs: np.ndarray):
        k_tanh_e = 5
        k_tanh_sigma0 = 5
        self.sigma_o = (de + self.vel_c * d_ref) + self.k1 * e + self.gamma * np.fabs(e) ** self.alpha * np.tanh(k_tanh_e * e)
        self.dot_sigma_o1 = np.fabs(self.sigma_o) ** self.beta * np.tanh(k_tanh_sigma0 * self.sigma_o)
        self.sigma_o1 += self.dot_sigma_o1 * self.dt
        self.so = self.sigma_o + self.lmd * self.sigma_o1

        uo1 = (kp / m * vel
               + dd_ref
               - self.k1 * (de + self.vel_c * d_ref)
               - self.gamma * self.alpha * np.fabs(e) ** (self.alpha - 1) * (de + self.vel_c * d_ref)
               - self.lmd * self.dot_sigma_o1)
        uo2 = -self.k2 * self.so - obs

        self.control = uo1 + uo2 + self.acc_c * dd_ref

    def fntsmc_pos_reset(self):
        self.sigma_o = np.zeros(self.dim)
        self.dot_sigma_o1 = np.zeros(self.dim)
        self.sigma_o1 = np.zeros(self.dim)
        self.so = self.sigma_o + self.lmd * self.sigma_o1

    def fntsmc_pos_reset_with_new_param(self, param: fntsmc_param):
        self.k1 = param.k1
        self.k2 = param.k2
        self.alpha = param.alpha
        self.beta = param.beta
        self.gamma = param.gamma
        self.lmd = param.lmd
        self.dt = param.dt
        self.dim = param.dim
        self.vel_c = param.vel_c
        self.acc_c = param.acc_c

        self.sigma_o = np.zeros(self.dim)
        self.dot_sigma_o1 = np.zeros(self.dim)
        self.sigma_o1 = np.zeros(self.dim)
        self.so = self.sigma_o + self.lmd * self.sigma_o1
        self.control = param.ctrl0

    def get_param_from_actor(self, action_from_actor: np.ndarray, update_k2: bool = True, update_z: bool = False):
        """
        @param action_from_actor:
        @return:
        """
        if np.min(action_from_actor) < 0:
            print('ERROR!!!!')
        if update_z:
            for i in range(3):
                if action_from_actor[i] > 0:
                    self.k1[i] = action_from_actor[i]
                if action_from_actor[i + 3] > 0 and update_k2:
                    self.k2[i] = action_from_actor[i + 3]
            if action_from_actor[6] > 0:
                self.gamma[:] = action_from_actor[6]
            if action_from_actor[7] > 0:
                self.lmd[:] = action_from_actor[7]
        else:
            for i in range(2):
                if action_from_actor[i] > 0:
                    self.k1[i] = action_from_actor[i]
                if action_from_actor[i + 3] > 0 and update_k2:
                    self.k2[i] = action_from_actor[i + 3]
            if action_from_actor[6] > 0:
                self.gamma[0: 2] = action_from_actor[6]
            if action_from_actor[7] > 0:
                self.lmd[0: 2] = action_from_actor[7]
