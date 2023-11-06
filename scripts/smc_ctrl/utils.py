import numpy as np


def deg2rad(deg: float) -> float:
    """
    :brief:         omit
    :param deg:     degree
    :return:        radian
    """
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    """
    :brief:         omit
    :param rad:     radian
    :return:        degree
    """
    return rad * 180.8 / np.pi


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def uo_2_ref_angle(uo: np.ndarray, psi_d: float, m: float, uf: float, angle_max: float):
    ux = uo[0]
    uy = uo[1]
    asin_phi_d = min(max((ux * np.sin(psi_d) - uy * np.cos(psi_d)) * m / uf, -1), 1)
    phi_d = np.arcsin(asin_phi_d)
    asin_theta_d = min(max((ux * np.cos(psi_d) + uy * np.sin(psi_d)) * m / (uf * np.cos(phi_d)), -1), 1)
    theta_d = np.arcsin(asin_theta_d)
    phi_d = max(min(phi_d, angle_max), -angle_max)
    theta_d = max(min(theta_d, angle_max), -angle_max)
    return phi_d, theta_d


def uo_2_ref_angle_throttle(control: np.ndarray, attitude: np.ndarray, m: float, g: float, limit=None, att_limitation: bool = False):
    ux = control[0]
    uy = control[1]
    uz = control[2]
    uf = (uz + g) * m / (np.cos(attitude[0]) * np.cos(attitude[1]))
    asin_phi_d = min(max((ux * np.sin(attitude[2]) - uy * np.cos(attitude[2])) * m / uf, -1), 1)
    phi_d = np.arcsin(asin_phi_d)
    asin_theta_d = min(max((ux * np.cos(attitude[2]) + uy * np.sin(attitude[2])) * m / (uf * np.cos(phi_d)), -1), 1)
    theta_d = np.arcsin(asin_theta_d)
    if att_limitation:
        if limit is not None:
            phi_d = max(min(phi_d, limit[0]), -limit[0])
            theta_d = max(min(theta_d, limit[1]), -limit[1])
    return phi_d, theta_d, uf


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating, update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x