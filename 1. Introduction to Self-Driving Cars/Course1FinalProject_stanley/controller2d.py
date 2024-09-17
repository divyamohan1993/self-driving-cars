#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np
# Importing necessary modules: `cutils` (a custom utility module) and `numpy` (for numerical operations).

class Controller2D(object):
    # This defines a class named `Controller2D`, which handles the 2D control logic for the self-driving car.
    
    def __init__(self, waypoints):
        # Constructor method, initializes an instance of the `Controller2D` class with the given waypoints.

        self.vars                = cutils.CUtils()
        # `self.vars` is an instance of `CUtils` class from `cutils`, used to manage variables.

        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        # Initialize various state variables related to the car's current position, speed, and control state.

        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        # Initialize control commands for throttle, brake, and steering.

        self._waypoints          = waypoints
        # Store the provided waypoints, which the car should follow.

        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        # Conversion factor from radians to steering units, specific to the vehicle's steering configuration.

        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi
        # Store common constants (π and 2π) for easier reference.

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        # Method to update the car's current state (position, orientation, speed, etc.).

        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        # Update internal state variables with the new values provided as arguments.

        if self._current_frame:
            self._start_control_loop = True
        # If a valid frame number is provided, start the control loop.

    def update_desired_speed(self):
        # Method to update the desired speed based on the closest waypoint.

        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        # Initialize variables to find the closest waypoint and its corresponding speed.

        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            # Calculate the Euclidean distance between the current position and each waypoint.

            if dist < min_dist:
                min_dist = dist
                min_idx = i
            # Update the minimum distance and index if a closer waypoint is found.

        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        # Set the desired speed to that of the closest waypoint, or the last waypoint if at the end.

        self._desired_speed = desired_speed
        # Update the internal state with the newly calculated desired speed.

    def update_waypoints(self, new_waypoints):
        # Method to update the list of waypoints.

        self._waypoints = new_waypoints
        # Replace the current waypoints with a new set.

    def get_commands(self):
        # Method to return the current control commands (throttle, steer, brake).

        return self._set_throttle, self._set_steer, self._set_brake
        # Return the throttle, steer, and brake values as a tuple.

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds       
        # Method to set the throttle command.

        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        # Constrain the input throttle to the range [0, 1].

        self._set_throttle = throttle
        # Update the internal throttle command.

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        # Method to set the steering command.

        input_steer = self._conv_rad_to_steer * input_steer_in_rad
        # Clamp the steering command to valid bounds
        # Convert the input steering angle from radians to steering units.        
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        # Constrain the steering command to the range [-1, 1].

        self._set_steer = steer
        # Update the internal steering command.

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        # Method to set the brake command.        
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        # Constrain the brake input to the range [0, 1].

        self._set_brake = brake
        # Update the internal brake command.

    def update_controls(self):
        # Main control loop method to update the car's control commands based on current state and waypoints.

        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################

        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        # Assign current state variables to local variables for easier reference.

        self.update_desired_speed()
        # Update the desired speed based on the current position and waypoints.

        v_desired       = self._desired_speed
        # Assign the desired speed to a local variable.

        t               = self._current_timestamp
        waypoints       = self._waypoints
        # Assign the current timestamp and waypoints to local variables.

        throttle_output = 0
        steer_output    = 0
        brake_output    = 0
        # Initialize output commands for throttle, steering, and braking.

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """

        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('t_previous', 0.0)
        self.vars.create_var('error_previous', 0.0)
        self.vars.create_var('integral_error_previous', 0.0)
        self.vars.create_var('throttle_previous', 0.0)
        # Create or retrieve persistent variables needed for control calculations.

        # Skip the first frame to store previous values properly        
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
                        
            # Only run the control logic if the control loop has been started.

            kp = 1.0
            ki = 0.2
            kd = 0.01
            # Define PID controller constants for throttle control.

            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            throttle_output = 0
            brake_output    = 0
            # Reset throttle and brake outputs.

            st = t - self.vars.t_previous
            e_v = v_desired - v
            inte_v = self.vars.integral_error_previous + e_v * st
            derivate = (e_v - self.vars.error_previous) / st
            acc = kp * e_v + ki * inte_v + kd * derivate
            # Calculate the acceleration (or deceleration) using a PID controller based on speed error.

            if acc > 0:
                throttle_output = (np.tanh(acc) + 1)/2
                if throttle_output - self.vars.throttle_previous > 0.1:
                    throttle_output = self.vars.throttle_previous + 0.1
                # Apply a throttle command if acceleration is positive, with a rate limit for smooth control.
            else:
                throttle_output = 0
                # No throttle if the calculated acceleration is negative.

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change the steer output with the lateral Stanley's controller.             
            steer_output = 0
            k_e = 0.3
            # Initialize steering output and define a constant for cross-track error correction.

            slope = (waypoints[-1][1]-waypoints[0][1])/ (waypoints[-1][0]-waypoints[0][0])
            a = -slope
            b = 1.0
            c = (slope*waypoints[0][0]) - waypoints[0][1]
            # Calculate the slope of the path between the first and last waypoints.

            yaw_path = np.arctan2(waypoints[-1][1]-waypoints[0][1], waypoints[-1][0]-waypoints[0][0])
            yaw_diff_heading = yaw_path - yaw 
            if yaw_diff_heading > np.pi:
                yaw_diff_heading -= 2 * np.pi
            if yaw_diff_heading < - np.pi:
                yaw_diff_heading += 2 * np.pi
            # Calculate the difference between the car's current heading and the desired path heading,
            # adjusting for the wrap-around at ±π.

            current_xy = np.array([x, y])
            crosstrack_error = np.min(np.sum((current_xy - np.array(waypoints)[:, :2])**2, axis=1))
            # Calculate the cross-track error (distance from the car's position to the closest point on the path).

            yaw_cross_track = np.arctan2(y-waypoints[0][1], x-waypoints[0][0])
            yaw_path2ct = yaw_path - yaw_cross_track
            if yaw_path2ct > np.pi:
                yaw_path2ct -= 2 * np.pi
            if yaw_path2ct < - np.pi:
                yaw_path2ct += 2 * np.pi
            if yaw_path2ct > 0:
                crosstrack_error = abs(crosstrack_error)
            else:
                crosstrack_error = - abs(crosstrack_error)
            # Calculate the yaw angle to the cross-track error and adjust it for the direction of the path.

            yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (v))
            # Calculate the steering angle needed to correct the cross-track error.

            steer_expect = yaw_diff_crosstrack + yaw_diff_heading
            if steer_expect > np.pi:
                steer_expect -= 2 * np.pi
            if steer_expect < - np.pi:
                steer_expect += 2 * np.pi
            steer_expect = min(1.22, steer_expect)
            steer_expect = max(-1.22, steer_expect)
            # Combine the cross-track correction and heading correction to get the expected steering command,
            # and constrain it within the allowed steering range.

            steer_output = steer_expect
            # Set the steering output to the calculated value.

            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)
            # Apply the calculated throttle, steering, and brake commands.

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v # Store forward speed to be used in next step
        self.vars.throttle_previous = throttle_output
        self.vars.t_previous = t
        self.vars.error_previous = e_v
        self.vars.integral_error_previous = inte_v
        # Update persistent variables with the latest values for use in the next control loop iteration.
