#! /usr/bin/python3
import os
import numpy as np
import pandas as pd
import torch

import rospy
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
import tf
from tf.transformations import quaternion_matrix

from smc_ctrl.uav_ros import UAV_ROS
from smc_ctrl.FNTSMC import fntsmc_param, fntsmc_pos
from smc_ctrl.observer import neso
from smc_ctrl.collector import data_collector
from smc_ctrl.rl import PPOActor_Gaussian
from smc_ctrl.ref_cmd import *
from smc_ctrl.utils import *


print('No error for import python modules.')


def state_cb(msg):
    global current_state
    current_state = msg


def uav_odom_cb(msg: Odometry):
    uav_odom.pose.pose.position.x = msg.pose.pose.position.x
    uav_odom.pose.pose.position.y = msg.pose.pose.position.y
    uav_odom.pose.pose.position.z = msg.pose.pose.position.z
    uav_odom.pose.pose.orientation.x = msg.pose.pose.orientation.x
    uav_odom.pose.pose.orientation.y = msg.pose.pose.orientation.y
    uav_odom.pose.pose.orientation.z = msg.pose.pose.orientation.z
    uav_odom.pose.pose.orientation.w = msg.pose.pose.orientation.w

    uav_odom.twist.twist.linear.x = msg.twist.twist.linear.x
    uav_odom.twist.twist.linear.y = msg.twist.twist.linear.y
    uav_odom.twist.twist.linear.z = msg.twist.twist.linear.z
    uav_odom.twist.twist.angular.x = msg.twist.twist.angular.x
    uav_odom.twist.twist.angular.y = msg.twist.twist.angular.y
    uav_odom.twist.twist.angular.z = msg.twist.twist.angular.z


def uav_battery_cb(msg: BatteryState):
    global voltage
    voltage = msg.voltage


def uav_odom_2_uav_state(odom: Odometry) -> np.ndarray:
    _orientation = odom.pose.pose.orientation
    _w = _orientation.w
    _x = _orientation.x
    _y = _orientation.y
    _z = _orientation.z
    rpy = tf.transformations.euler_from_quaternion([_x, _y, _z, _w])
    _uav_state = np.array([
        odom.pose.pose.position.x,  # x
        odom.pose.pose.position.y,  # y
        odom.pose.pose.position.z,  # z
        odom.twist.twist.linear.x,  # vx
        odom.twist.twist.linear.y,  # vy
        odom.twist.twist.linear.z,  # vz
        rpy[0],  # phi
        rpy[1],  # theta
        rpy[2],  # psi
        odom.twist.twist.angular.x,  # p
        odom.twist.twist.angular.y,  # q
        odom.twist.twist.angular.z  # r
    ])
    return _uav_state


def thrust_2_throttle(thrust: float):
    """

    """
    '''gazebo 线性模型'''
    k = 0.391 / 0.797 / 9.8
    _throttle = k * thrust
    _throttle = max(min(_throttle, 0.9), 0.10)
    '''gazebo 线性模型'''

    '''姜百伦飞机模型'''
    # _m = 0.453	# uav 质量
    # _g = 9.8
    # # voltage = 12.0
    # _y = -0.0426229508 * voltage + 0.7269726774		# 不同电压下的悬停油门
    # _y = min(max(_y, 0.21), 0.27)
    # k = _y / (_m * _g)
    # _throttle = k * thrust
    # _throttle = max(min(_throttle, 0.4), 0.10)
    '''姜百伦飞机模型'''

    '''330飞机模型'''
    # _m = 0.797	# uav 质量
    # _g = 9.8
    # _y = -0.0181 * voltage + 0.4854		# 不同电压下的悬停油门
    # _y = min(max(_y, 0.19), 0.23)
    # k = _y / (_m * _g)
    # _throttle = k * thrust
    # _throttle = max(min(_throttle, 0.6), 0.10)
    '''330飞机模型'''
    return _throttle


def approaching(_t: float, ap_flag: bool, threshold: float):
    ref_amplitude = np.array([0., 0., 0., 0.])
    _ref, _, _, _ = ref_uav(0., ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
    pose.pose.position.x = _ref[0]
    pose.pose.position.y = _ref[1]
    pose.pose.position.z = _ref[2]
    uav_pos = uav_odom_2_uav_state(uav_odom)[0: 3]
    local_pos_pub.publish(pose)
    _bool = False
    if np.linalg.norm(_ref[0: 3] - uav_pos) < 0.25:
        if ap_flag:
            _bool = True if rospy.Time.now().to_sec() - _t >= threshold else False
        else:
            ap_flag = True
    else:
        _bool = False
        ap_flag = False
    return ap_flag, _bool


def get_normalizer_from_file(dim, path, file):
    norm = Normalization(dim)
    data = pd.read_csv(path + file, header=0).to_numpy()
    norm.running_ms.n = data[0, 0]
    norm.running_ms.mean = data[:, 1]
    norm.running_ms.std = data[:, 2]
    norm.running_ms.S = data[:, 3]
    norm.running_ms.n = data[0, 4]
    norm.running_ms.mean = data[:, 5]
    norm.running_ms.std = data[:, 6]
    norm.running_ms.S = data[:, 7]
    return norm


'''Some pre-defined parameters'''
current_state = State()  # monitor uav status
pose = PoseStamped()  # publish offboard [x_d y_d z_d] cmd
uav_odom = Odometry()  # subscribe uav state x y z vx vy vz phi theta psi p q r
ctrl_cmd = AttitudeTarget()  # publish offboard expected [phi_d theta_d psi_d throttle] cmd
voltage = 11.4  # subscribe voltage from the battery
global_flag = 0  # UAV working mode monitoring
# UAV working mode
# 0: connect to onboard computer, arm, load parameters, prepare
# 1: approaching and initialization
# 2: control by SMC ([phi_d theta_d psi_d throttle])
# 3: finish and switch to OFFBOARD-position
'''Some pre-defined parameters'''

'''Parameter list of the position controller'''
DT = 0.01
pos_ctrl_param = fntsmc_param()
pos_ctrl_param.k1 = np.array([1.2, 0.8, 4.0])
pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
pos_ctrl_param.alpha = np.array([1.2, 1.5, 2.5])    # 原来 alpha_z = 1.2, 1.5, 1.2
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.75])     # 原来 beta_z = 0.8
pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
pos_ctrl_param.vel_c = np.array([0.0, 0.0, 0.0])   # 0.05, 0.05, -0.005   使用RL的时候，不需要使用补偿了，哈哈哈哈
pos_ctrl_param.acc_c = np.array([0., 0., 0.])
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller'''


if __name__ == "__main__":
    rospy.init_node("offb_node_py")  # 初始化一个节点

    '''topic subscribe'''
    # 订阅回来 uav 的 state 信息，包括连接状态，工作模式，电机的解锁状态
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    # # subscribe the position of the UAV
    # uav_pos_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=uav_pos_cb)

    # subscribe the odom of the UAV
    uav_vel_sub = rospy.Subscriber("mavros/local_position/odom", Odometry, callback=uav_odom_cb)
    uav_battery_sub = rospy.Subscriber("mavros/battery", BatteryState, callback=uav_battery_cb)
    '''topic subscribe'''

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
    uav_att_throttle_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
    '''Publish 位置指令给 UAV'''

    '''arming service'''
    rospy.wait_for_service("/mavros/cmd/arming")  # 等待解锁电机的 service 建立
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    '''working mode service'''
    rospy.wait_for_service("/mavros/set_mode")  # 等待设置 UAV 工作模式的 service 建立
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    rate = rospy.Rate(1 / DT)

    '''如果没有连接上，就等待'''
    while (not rospy.is_shutdown()) and (not current_state.connected):
        rate.sleep()

    pose.pose.position.x = uav_odom.pose.pose.position.x
    pose.pose.position.y = uav_odom.pose.pose.position.y
    pose.pose.position.z = uav_odom.pose.pose.position.z

    for i in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()  # 先设置工作模式为 offboard
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True  # 通过指令将电机解锁

    while (current_state.mode != "OFFBOARD") and (not rospy.is_shutdown()):  # 等待
        if set_mode_client.call(offb_set_mode).mode_sent:
            print('Switching to OFFBOARD mode is available...waiting for 1 seconds')
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    t0 = rospy.Time.now().to_sec()

    while rospy.Time.now().to_sec() - t0 < 1.0:
        local_pos_pub.publish(pose)
        rate.sleep()

    while (not current_state.armed) and (not rospy.is_shutdown()):
        if arming_client.call(arm_cmd).success:
            print('UAV is armed now...waiting for 1 seconds')
            break
        local_pos_pub.publish(pose)
        rate.sleep()

    t0 = rospy.Time.now().to_sec()

    while rospy.Time.now().to_sec() - t0 < 1.0:  # OK
        local_pos_pub.publish(pose)
        rate.sleep()

    print('Start......')
    print('Approaching...')
    global_flag = 1

    '''generate reference command'''
    # ref_amplitude = np.array([0., 0., 0., 0])  # xd yd zd psid 振幅
    ref_amplitude = np.array([1.5, 1.5, 0.3, 0])  # xd yd zd psid 振幅
    ref_period = np.array([5, 5, 5, 10])  # xd yd zd psid 周期
    ref_bias_a = np.array([0, 0, 1.0, 0])  # xd yd zd psid 幅值偏移
    ref_bias_phase = np.array([np.pi / 2, 0, 0, 0])  # xd yd zd psid 相位偏移
    '''generate reference command'''

    t0 = rospy.Time.now().to_sec()
    approaching_flag = False

    '''define controllers and observers'''
    uav_ros = None
    controller = None
    obs = None
    data_record = None
    '''define controllers and observers'''

    '''load actor'''
    opt_pos = PPOActor_Gaussian(state_dim=6, action_dim=8)
    # optPathPos = os.getcwd() + '/src/adp-smc-uav-ros/nets/pos_new1-260/'  # 仿真最好的，实际最差的
    optPathPos = os.getcwd() + '/src/adp-smc-uav-ros/nets/opt1/'        # 最好的
    # optPathPos = os.getcwd() + '/src/adp-smc-uav-ros/nets/opt2/'      # 第二好的
    opt_pos.load_state_dict(torch.load(optPathPos + 'actor'))
    pos_norm = get_normalizer_from_file(6, optPathPos, 'state_norm.csv')
    '''load actor'''

    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec()
        if global_flag == 1:  # approaching
            approaching_flag, ok = approaching(t0, approaching_flag, 10.0)
            # ok = True
            if ok:
                print('OFFBOARD, start to initialize...')
                uav_ros = UAV_ROS(m=0.797, g=9.8, kt=1e-3, dt=DT)
                controller = fntsmc_pos(pos_ctrl_param)

                # obs = neso(l1=np.array([3., 3., 3.]),
                #        l2=np.array([3., 3., 3.]),
                #        l3=np.array([1., 1., 3.]),
                #        r=np.array([20., 20., 20.]),
                #        k1=np.array([0.7, 0.7, 0.7]),
                #        k2=np.array([0.001, 0.001, 0.001]),
                #        dim=3,
                #        dt=DT)
                obs = neso(l1=np.array([0.1, 0.1, 0.3]),
                           l2=np.array([0.1, 0.1, 0.15]),
                           l3=np.array([0.08, 0.08, 0.08]),
                           r=np.array([0.5, 0.5, 0.5]),     # r 越小，增益越小 奥利给兄弟们干了
                           k1=np.array([0.7, 0.7, 0.7]),
                           k2=np.array([0.01, 0.01, 0.01]),
                           dim=3,
                           dt=DT)

                data_record = data_collector(N=round(uav_ros.time_max / DT))

                print('Control...')
                t0 = rospy.Time.now().to_sec()
                global_flag = 2
        elif global_flag == 2:  # control
            t_now = round(t - t0, 4)
            if uav_ros.n % 100 == 0:
                print('time: ', t_now, data_record.index)

            '''1. generate reference command and uncertainty'''
            rax = max(min(0.5 * t_now, 1.5), 0.0)
            ray = max(min(0.5 * t_now, 1.5), 0.0)
            raz = max(min(0.06 * t_now, 0.3), 0.0)
            rapsi = max(min(0.5 * t_now, 0), 0.0)
            ref_amplitude = np.array([rax, ray, raz, rapsi])
            ref, dot_ref, dot2_ref, dot3_ref = ref_uav(t_now,
                                                       ref_amplitude,
                                                       ref_period,
                                                       ref_bias_a,
                                                       ref_bias_phase)

            '''2. generate outer-loop reference signal 'eta_d' and its 1st, 2nd, and 3rd-order derivatives'''
            eta_d = ref[0: 3]
            dot_eta_d = dot_ref[0: 3]
            dot2_eta_d = dot2_ref[0: 3]
            e = uav_ros.eta() - eta_d
            de = uav_ros.dot_eta() - dot_eta_d

            USE_OBSERVER = True
            if USE_OBSERVER:
                syst_dynamic = -uav_ros.kt / uav_ros.m * uav_ros.dot_eta() + uav_ros.A()
                observe, _ = obs.observe(x=uav_ros.eta(), syst_dynamic=syst_dynamic)
            else:
                observe = np.zeros(3)

            CONTROLLER = 'RL'  # or 'FNTSMC'
            '''3. Update the parameters of FNTSMC if RL is used'''
            if CONTROLLER == 'RL':
                pos_s = np.concatenate((e, de))
                param_pos = opt_pos.evaluate(pos_norm(pos_s))  # new position control parameter
                controller.get_param_from_actor(param_pos, update_k2=True, update_z=False)  # update position control parameter

            '''3. generate phi_d, theta_d, throttle'''
            controller.control_update(uav_ros.kt, uav_ros.m, uav_ros.uav_vel(), e, de, dot_eta_d, dot2_eta_d, obs=observe)
            phi_d, theta_d, uf = uo_2_ref_angle_throttle(controller.control, uav_ros.uav_att(),
                                                         uav_ros.m, uav_ros.g,
                                                         limit=[np.pi / 4, np.pi / 4],
                                                         att_limitation=True)
            # phi_d = 0.
            # theta_d = 0.
            # uf = uav_ros.m * uav_ros.g

            '''4. publish'''
            ctrl_cmd.header.stamp = rospy.Time.now()
            ctrl_cmd.type_mask = AttitudeTarget.IGNORE_ROLL_RATE + AttitudeTarget.IGNORE_PITCH_RATE + AttitudeTarget.IGNORE_YAW_RATE
            cmd_q = tf.transformations.quaternion_from_euler(phi_d, theta_d, ref[3])  # ref[3] - psi_d
            ctrl_cmd.orientation.x = cmd_q[0]
            ctrl_cmd.orientation.y = cmd_q[1]
            ctrl_cmd.orientation.z = cmd_q[2]
            ctrl_cmd.orientation.w = cmd_q[3]
            ctrl_cmd.thrust = thrust_2_throttle(uf)
            uav_att_throttle_pub.publish(ctrl_cmd)

            # pose.pose.position.x = ref[0]
            # pose.pose.position.y = ref[1]
            # pose.pose.position.z = ref[2]
            # local_pos_pub.publish(pose)

            '''5. get new uav states from Gazebo'''
            uav_ros.rk44(action=[phi_d, theta_d, uf], uav_state=uav_odom_2_uav_state(uav_odom))

            '''6. data storage'''
            data_block = {'time': uav_ros.time,  # simulation time
                          'throttle': uf,
                          'thrust': ctrl_cmd.thrust,
                          'ref_angle': np.array([phi_d, theta_d, ref[3]]),
                          'ref_pos': ref[0: 3],
                          'ref_vel': dot_ref[0: 3],
                          'd_out_obs': observe,
                          'state': uav_ros.uav_state_call_back(),
                          'dot_angle': uav_ros.uav_dot_att()}
            data_record.record(data_block)

            if data_record.index == data_record.N:
                print('Data collection finish. Switching offboard position...')
                data_record.package2file(path=os.getcwd() + '/src/adp-smc-uav-ros/scripts/datasave/')
                global_flag = 3
        elif global_flag == 3:  # finish, back to offboard position
            pose.pose.position.x = 0
            pose.pose.position.y = 0
            pose.pose.position.z = 0.5
            local_pos_pub.publish(pose)
        else:
            pose.pose.position.x = 0
            pose.pose.position.y = 0
            pose.pose.position.z = 0.5
            local_pos_pub.publish(pose)
            print('WORKING MODE ERROR...')
        rate.sleep()
