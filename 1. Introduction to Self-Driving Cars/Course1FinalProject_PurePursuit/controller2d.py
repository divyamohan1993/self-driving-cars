import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars = cutils.CUtils()
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_frame = 0
        self._current_timestamp = 0
        self._start_control_loop = False
        self._set_throttle = 0
        self._set_brake = 0
        self._set_steer = 0
        self._waypoints = waypoints
        self._waypoints_array = np.array(waypoints)
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed
        self._current_timestamp = timestamp
        self._current_frame = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        waypoints = self._waypoints_array
        dx = waypoints[:, 0] - self._current_x
        dy = waypoints[:, 1] - self._current_y
        dists = np.hypot(dx, dy)
        min_idx = np.argmin(dists)
        # lookahead_distance = 300.0 # Sushil Divyadristi Driver
        lookahead_distance = 10.0 # Drunk Racetrack Mode
        if min_idx >= len(waypoints) - 1:
            lookahead_idx = len(waypoints) - 1
        else:
            dx_segments = waypoints[min_idx+1:, 0] - waypoints[min_idx:-1, 0]
            dy_segments = waypoints[min_idx+1:, 1] - waypoints[min_idx:-1, 1]
            segment_distances = np.hypot(dx_segments, dy_segments)
            cumulative_distances = np.cumsum(segment_distances)
            lookahead_idx_offset = np.searchsorted(cumulative_distances, lookahead_distance, side='right')
            lookahead_idx = min_idx + lookahead_idx_offset + 1
            lookahead_idx = min(lookahead_idx, len(waypoints) - 1)
        max_speed_ahead = np.max(waypoints[min_idx:lookahead_idx+1, 2])
        # desired_speed = min(max_speed_ahead + 3.0, 8.2) # Sanskari Driver Mode
        desired_speed = min(max_speed_ahead + 10.0, 200.0) # Pi ke talli - Racetrack Mode - no speed limit
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints
        self._waypoints_array = np.array(new_waypoints)

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        self._set_throttle = np.clip(input_throttle, 0.0, 1.0)

    def set_steer(self, input_steer_in_rad):
        input_steer = self._conv_rad_to_steer * input_steer_in_rad
        self._set_steer = np.clip(input_steer, -1.0, 1.0)

    def set_brake(self, input_brake):
        self._set_brake = np.clip(input_brake, 0.0, 1.0)

    def update_controls(self):
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        self.update_desired_speed()
        v_desired = self._desired_speed
        t = self._current_timestamp
        waypoints = self._waypoints_array
        throttle_output = 0
        steer_output = 0
        brake_output = 0
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('v_error_integral', 0.0)
        self.vars.create_var('v_error_previous', 0.0)
        self.vars.create_var('Lf', 2.5)
        if self._start_control_loop:
            t_delta = max(t - self.vars.t_previous, 0.1)
            v_error = v_desired - v
            self.vars.v_error_integral += v_error * t_delta
            v_error_derivative = (v_error - self.vars.v_error_previous) / t_delta
            u = 5.0 * v_error + 0.5 * v_error_derivative 
            if u > 0:
                throttle_output = np.clip(u, 0.0, 1.0)
                brake_output = 0.0
            else:
                throttle_output = 0.0
                brake_output = np.clip(-u, 0.0, 1.0)
            Lf = self.vars.Lf
            # Ld = 20.0 * v + 15.0 # Sanskari acche ghar ka bade khandaan ka driver
            Ld = 0.1 * v + 1.0 # After 10 pegs of 10,000 year old 'Old Monk' - Ludicrous Mode
            dx = waypoints[:, 0] - x
            dy = waypoints[:, 1] - y
            dists = np.hypot(dx, dy)
            min_idx = np.argmin(dists)
            if min_idx >= len(waypoints) - 1:
                lookahead_point = waypoints[-1]
            else:
                dx_segments = waypoints[min_idx+1:, 0] - waypoints[min_idx:-1, 0]
                dy_segments = waypoints[min_idx+1:, 1] - waypoints[min_idx:-1, 1]
                segment_distances = np.hypot(dx_segments, dy_segments)
                cumulative_distances = np.cumsum(segment_distances)
                lookahead_idx_offset = np.searchsorted(cumulative_distances, Ld, side='right')
                lookahead_idx = min_idx + lookahead_idx_offset + 1
                if lookahead_idx >= len(waypoints):
                    lookahead_point = waypoints[-1]
                else:
                    lookahead_point = waypoints[lookahead_idx]
            dx = lookahead_point[0] - x
            dy = lookahead_point[1] - y
            local_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
            local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)
            if local_x != 0:
                steering_angle = np.arctan2(2 * Lf * local_y, local_x**2 + local_y**2)
            else:
                steering_angle = 0.0
            steer_output = np.clip(steering_angle, -1.22, 1.22)
            self.set_throttle(throttle_output)
            self.set_steer(steer_output)
            self.set_brake(brake_output)
        self.vars.v_previous = v
        self.vars.t_previous = t
        self.vars.v_error_previous = v_error
