import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

class ControllerState:
    def __init__(self):
        self.prev_steering_error = 0.0
        self.velocity_integral = 0.0
        self.last_target_idx = 0


controller_state = ControllerState()


def find_nearest_point_on_path(position, path, start_idx=0):
    position = np.asarray(position, dtype=float)
    path = np.asarray(path, dtype=float)

    n = len(path)

    window = 100 if n > 100 else n

    best_idx = start_idx % n
    best_dist = np.linalg.norm(path[best_idx] - position)

    for k in range(1, window):
        idx = (start_idx + k) % n
        dist = np.linalg.norm(path[idx] - position)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx

    return best_idx, path[best_idx]


def compute_lookahead_point(path, nearest_idx, lookahead):
    path = np.asarray(path, dtype=float)
    n = len(path)

    cumulative_distance = 0.0
    lookahead_idx = nearest_idx

    for i in range(1, n):
        next_idx = (nearest_idx + i) % n
        prev_idx = (nearest_idx + i - 1) % n 

        segment_length = np.linalg.norm(path[next_idx] - path[prev_idx])
        cumulative_distance += segment_length

        if cumulative_distance >= lookahead:
            lookahead_idx = next_idx
            break

    return lookahead_idx, path[lookahead_idx]


def compute_path_curvature(path, idx):
    window = 3
    path = np.asarray(path, dtype=float)
    n = len(path)

    idx_before = (idx - window) % n
    idx_after = (idx + window) % n

    before_point = path[idx_before]
    middle_point = path[idx]
    after_point = path[idx_after]

    v1 = middle_point - before_point
    v2 = after_point - middle_point

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0

    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    dot_product = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle_change = float(np.arccos(dot_product))

    arc_length = v1_norm + v2_norm
    if arc_length < 1e-5:
        return 0.0

    curvature = angle_change / arc_length
    return curvature


def compute_path_heading(path, idx):
    path = np.asarray(path, dtype=float)
    n = len(path)
    next_idx = (idx + 1) % n 

    dx = path[next_idx][0] - path[idx][0]
    dy = path[next_idx][1] - path[idx][1]

    return float(np.arctan2(dy, dx))


def compute_desired_velocity(path, nearest_idx):
    SPEED_LOOKAHEAD_POINTS = 24
    CURVATURE_EPS = 1e-3
    CURVATURE_GAIN = 4.4
    MIN_SPEED = 15.0
    MAX_SPEED = 100.0

    path = np.asarray(path, dtype=float)
    n = len(path)

    maxCurv = 0.0
    for i in range(SPEED_LOOKAHEAD_POINTS):
        idx = (nearest_idx + i) % n
        curvature = compute_path_curvature(path, idx)
        if curvature > maxCurv:
            maxCurv = curvature

    if maxCurv > 1e-6:
        desired_v = np.sqrt(CURVATURE_GAIN / (maxCurv + CURVATURE_EPS))
    else:
        desired_v = MAX_SPEED

    desired_v = float(np.clip(desired_v, MIN_SPEED, MAX_SPEED))
    return desired_v


def compute_desired_steering_angle(position, heading, velocity, path, nearest_idx, wheelbase):
    BASE_LOOKAHEAD = 18.0
    VEL_LOOKAHEAD_SCALER = 40.0
    VEL_LOOKAHEAD_MIN = 12.0
    VEL_LOOKAHEAD_MAX = 45.0
    TIGHT_CURV_THRESHOLD = 0.12
    TIGHT_LOOKAHEAD_SCALE = 0.8
    LOOKAHEAD_MIN = 8.0
    LOOKAHEAD_MAX = 35.0
    K_STAN = 0.22
    K_HEADING_MIX = 0.6
    VEL_MIN_STANLEY = 5.0

    position = np.asarray(position, dtype=float)
    path = np.asarray(path, dtype=float)

    velFactor = max(abs(velocity), 15.0) / VEL_LOOKAHEAD_SCALER
    lookahead = BASE_LOOKAHEAD * velFactor
    lookahead = float(np.clip(lookahead, VEL_LOOKAHEAD_MIN, VEL_LOOKAHEAD_MAX))

    curv = compute_path_curvature(path, nearest_idx)
    if curv > TIGHT_CURV_THRESHOLD:
        lookahead *= TIGHT_LOOKAHEAD_SCALE

    lookahead = float(np.clip(lookahead, LOOKAHEAD_MIN, LOOKAHEAD_MAX))

    _, target = compute_lookahead_point(path, nearest_idx, lookahead)

    dx = target[0] - position[0]
    dy = target[1] - position[1]
    dist = float(np.sqrt(dx**2 + dy**2))

    if dist < 0.1:
        return 0.0

    alpha = np.arctan2(dy, dx) - heading
    # normalize angle
    alpha = float(np.arctan2(np.sin(alpha), np.cos(alpha)))

    closest = path[nearest_idx]
    cte = float(np.linalg.norm(position - closest))

    pathHeading = compute_path_heading(path, nearest_idx)
    perpHeading = pathHeading + np.pi / 2.0
    to_car = position - closest

    cross_track_sign = np.sign(np.sin(perpHeading) * to_car[0] - np.cos(perpHeading) * to_car[1])
    signedCTE = float(cross_track_sign * cte)

    hErr = pathHeading - heading
    hErr = float(np.arctan2(np.sin(hErr), np.cos(hErr)))

    velAdj = max(abs(velocity), VEL_MIN_STANLEY)

    stanley_term = hErr + np.arctan(K_STAN * signedCTE / velAdj)
    pure_pursuit_term = np.arctan(2.0 * wheelbase * np.sin(alpha) / dist)

    steering_angle = (K_HEADING_MIX * stanley_term + (1.0 - K_HEADING_MIX) * pure_pursuit_term)

    return float(steering_angle)


def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    K_STEER_P = 1.3
    K_STEER_D = 0.6
    K_VEL_P = 1.5
    K_VEL_I = 0.08
    VEL_INT_CLIP = 100.0

    assert(desired.shape == (2,))

    global controller_state

    state = np.asarray(state, dtype=float)
    desired = np.asarray(desired, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    currentSteer = float(state[2])
    vel = float(state[3])

    desiredSteer = float(desired[0])
    targetVel = float(desired[1])

    max_steering_vel = float(parameters[9])
    max_acceleration = float(parameters[10])

    steerErr = desiredSteer - currentSteer
    steerErrDeriv = steerErr - controller_state.prev_steering_error

    steerRate = K_STEER_P * steerErr + K_STEER_D * steerErrDeriv
    steerRate = float(np.clip(steerRate, -max_steering_vel, max_steering_vel))

    controller_state.prev_steering_error = steerErr

    velErr = targetVel - vel

    controller_state.velocity_integral += velErr
    controller_state.velocity_integral = float(np.clip(controller_state.velocity_integral, -VEL_INT_CLIP, VEL_INT_CLIP))

    acceleration = K_VEL_P * velErr + K_VEL_I * controller_state.velocity_integral
    acceleration = float(np.clip(acceleration, -max_acceleration, max_acceleration))

    return np.array([steerRate, acceleration], dtype=float)


def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:
    ERR_CROSS_SCALE = 15.0
    K_ERR_SLOW = 0.8
    MIN_SLOW_FACTOR = 0.4

    global controller_state

    state = np.asarray(state, dtype=float)
    parameters = np.asarray(parameters, dtype=float)

    position = state[0:2]
    vel = float(state[3])
    heading = float(state[4])
    wheelbase = float(parameters[0])

    path = racetrack.raceline

    nearest_idx, closest = find_nearest_point_on_path(position, path, controller_state.last_target_idx)
    controller_state.last_target_idx = nearest_idx

    targetVel = compute_desired_velocity(path, nearest_idx)

    pathHeading = compute_path_heading(path, nearest_idx)
    hErr = pathHeading - heading
    hErr = float(np.arctan2(np.sin(hErr), np.cos(hErr)))

    cte = float(np.linalg.norm(position - closest))

    err_magnitude = abs(hErr) + 0.5 * (cte / ERR_CROSS_SCALE)
    slow_factor = float(np.clip(1.0 - K_ERR_SLOW * err_magnitude, MIN_SLOW_FACTOR, 1.0))
    targetVel *= slow_factor

    desired_steering_angle = compute_desired_steering_angle(position, heading, vel, path, nearest_idx, wheelbase)

    return np.array([desired_steering_angle, targetVel], dtype=float)


