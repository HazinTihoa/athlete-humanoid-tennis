##
#
# Assortment of useful rotation functions from unitree_rl_gym
# https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy/deploy_real/common
#
##

############################################################################
# UNITREE ROTATION
############################################################################

# rotation helper functions
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


############################################################################
# UNITREE REMOTE
############################################################################

import struct


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


############################################################################
# RECURRENT THREAD
############################################################################

import threading
import time


class RecurrentThread:
    """Minimal fixed-rate background thread.

    Drop-in replacement for unitree_sdk2py.utils.thread.RecurrentThread so the
    hardware control loop no longer depends on the pure-Python SDK. Calls
    ``target`` every ``interval`` seconds on a daemon thread, compensating for
    the work time so the average rate stays close to the requested interval.
    """

    def __init__(self, interval, target, name="recurrent"):
        self._interval = float(interval)
        self._target = target
        self._name = name
        self._thread = None
        self._running = False

    def Start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name=self._name, daemon=True
        )
        self._thread.start()

    def Wait(self):
        if self._thread is not None:
            self._thread.join()

    def Stop(self):
        self._running = False
        self.Wait()

    def _run(self):
        next_t = time.perf_counter()
        while self._running:
            self._target()
            next_t += self._interval
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind: reset the schedule to avoid busy-spinning to catch up.
                next_t = time.perf_counter()
