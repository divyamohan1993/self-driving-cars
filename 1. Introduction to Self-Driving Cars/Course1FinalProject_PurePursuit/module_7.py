#!/usr/bin/env python3

"""
==== RACETRACK AUTOPILOT RACER ==== NO MERCY ==== NO COMFORT ====

Author: Divya Mohan
Modified on Sep 17, 2024

This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.

Based on the CARLA waypoint follower assessment client script
provided by Self-Driving Cars by University of Toronto.

It is a controller assessment to follow a given trajectory,
where the trajectory has been defined using way-points.

RACE STARTING in a moment...

Best Time: 00:01:11:04 - HH:MM:SS:MS
"""

from __future__ import print_function
from __future__ import division
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import controller2d
import configparser
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
import live_plotter as lv
from carla import sensor
from carla.client import make_carla_client, VehicleControl
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.controller import utils

ITER_FOR_SIM_TIMESTEP = 10
WAIT_TIME_BEFORE_START = 5.00
TOTAL_RUN_TIME = 200.00
TOTAL_FRAME_BUFFER = 300

NUM_PEDESTRIANS = 0
NUM_VEHICLES = 0
SEED_PEDESTRIANS = 0
SEED_VEHICLES = 0

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}

SIMWEATHER = WEATHERID["CLEARNOON"]
PLAYER_START_INDEX = 1

FIGSIZE_X_INCHES = 8
FIGSIZE_Y_INCHES = 8
PLOT_LEFT = 0.1
PLOT_BOT = 0.1
PLOT_WIDTH = 0.8
PLOT_HEIGHT = 0.8

WAYPOINTS_FILENAME = 'racetrack_waypoints.txt'
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0
INTERP_MAX_POINTS_PLOT = 10
INTERP_LOOKAHEAD_DISTANCE = 20
INTERP_DISTANCE_RES = 0.01

CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/controller_output/'


def make_carla_settings(args):
    settings = CarlaSettings()
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings


class Timer(object):
    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) / \
            self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time


def get_current_pose(measurement):
    x = measurement.player_measurements.transform.location.x
    y = measurement.player_measurements.transform.location.y
    yaw = math.radians(
        measurement.player_measurements.transform.rotation.yaw)
    return (x, y, yaw)


def get_start_pos(scene):
    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)
    return (x, y, yaw)


def send_control_command(client, throttle, steer, brake,
                         hand_brake=False, reverse=False):
    control = VehicleControl()
    steer = max(min(steer, 1.0), -1.0)
    throttle = max(min(throttle, 1.0), 0)
    brake = max(min(brake, 1.0), 0)
    control.steer = steer
    control.throttle = throttle
    control.brake = brake
    control.hand_brake = hand_brake
    control.reverse = reverse
    client.send_control(control)


def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def store_trajectory_plot(graph, fname):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)


def write_trajectory_file(x_list, y_list, v_list, t_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')
    with open(file_name, 'w') as trajectory_file:
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f\n' %
                                  (x_list[i], y_list[i], v_list[i], t_list[i]))


def exec_waypoint_nav_demo(args):
    with make_carla_client(args.host, args.port) as client:
        print('Carla client connected.')
        settings = make_carla_settings(args)
        scene = client.load_settings(settings)
        player_start = PLAYER_START_INDEX
        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(player_start)

        # Read configuration file
        config = configparser.ConfigParser()
        config.read(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))
        demo_opt = config['Demo Parameters']

        # Read 'live_plotting' parameter
        enable_live_plot = demo_opt.get(
            'live_plotting', 'true').capitalize()
        enable_live_plot = enable_live_plot == 'True'
        live_plot_period = float(demo_opt.get('live_plotting_period', 0))
        live_plot_timer = Timer(live_plot_period)

        # Read 'gen_controller_output' parameter
        enable_controller_output = demo_opt.get(
            'gen_controller_output', 'true').capitalize()
        enable_controller_output = enable_controller_output == 'True'

        waypoints_file = WAYPOINTS_FILENAME
        waypoints_np = None
        with open(waypoints_file) as waypoints_file_handle:
            waypoints = list(csv.reader(waypoints_file_handle,
                                        delimiter=',',
                                        quoting=csv.QUOTE_NONNUMERIC))
            waypoints_np = np.array(waypoints)

        delta_wp = waypoints_np[1:, :] - waypoints_np[:-1, :]
        wp_distance = np.linalg.norm(delta_wp[:, :2], axis=1)
        wp_distance = np.append(wp_distance, 0)

        wp_interp = []
        wp_interp_hash = []
        interp_counter = 0

        for i in range(waypoints_np.shape[0] - 1):
            wp_interp.append(list(waypoints_np[i]))
            wp_interp_hash.append(interp_counter)
            interp_counter += 1

            num_pts_to_interp = int(np.floor(wp_distance[i] /
                                             float(INTERP_DISTANCE_RES)) - 1)
            wp_vector = waypoints_np[i+1] - waypoints_np[i]
            wp_uvector = wp_vector / np.linalg.norm(wp_vector)
            for j in range(num_pts_to_interp):
                next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                wp_interp.append(list(waypoints_np[i] + next_wp_vector))
                interp_counter += 1

        wp_interp.append(list(waypoints_np[-1]))
        wp_interp_hash.append(interp_counter)
        interp_counter += 1

        controller = controller2d.Controller2D(waypoints)

        num_iterations = ITER_FOR_SIM_TIMESTEP
        if (ITER_FOR_SIM_TIMESTEP < 1):
            num_iterations = 1

        measurement_data, sensor_data = client.read_data()
        sim_start_stamp = measurement_data.game_timestamp / 1000.0

        send_control_command(client, throttle=0.0, steer=0, brake=1.0)

        sim_duration = 0
        for i in range(num_iterations):
            measurement_data, sensor_data = client.read_data()
            send_control_command(client, throttle=0.0, steer=0, brake=1.0)
            if i == num_iterations - 1:
                sim_duration = measurement_data.game_timestamp / 1000.0 - \
                    sim_start_stamp

        SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
        print("SERVER SIMULATION STEP APPROXIMATION: " +
              str(SIMULATION_TIME_STEP))

        TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) /
                                   SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

        measurement_data, sensor_data = client.read_data()
        start_x, start_y, start_yaw = get_current_pose(measurement_data)
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)

        # Initialize history lists only if controller output is enabled
        if enable_controller_output:
            x_history = [start_x]
            y_history = [start_y]
            yaw_history = [start_yaw]
            time_history = [0]
            speed_history = [0]

        # Conditionally create LivePlotter objects and figures
        if enable_controller_output:
            lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
            lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

            # Create trajectory figure
            trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
                title='Vehicle Trajectory',
                figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
                edgecolor="black",
                rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])
            trajectory_fig.set_invert_x_axis()
            trajectory_fig.set_axis_equal()
            trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                                     x0=waypoints_np[:, 0], y0=waypoints_np[:, 1],
                                     linestyle="-", marker="", color='g')
            trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                                     x0=[start_x]*TOTAL_EPISODE_FRAMES,
                                     y0=[start_y]*TOTAL_EPISODE_FRAMES,
                                     color=[1, 0.5, 0])
            trajectory_fig.add_graph("lookahead_path",
                                     window_size=INTERP_MAX_POINTS_PLOT,
                                     x0=[start_x]*INTERP_MAX_POINTS_PLOT,
                                     y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                                     color=[0, 0.7, 0.7],
                                     linewidth=4)
            trajectory_fig.add_graph("start_pos", window_size=1,
                                     x0=[start_x], y0=[start_y],
                                     marker=11, color=[1, 0.5, 0],
                                     markertext="Start", marker_text_offset=1)
            trajectory_fig.add_graph("end_pos", window_size=1,
                                     x0=[waypoints_np[-1, 0]],
                                     y0=[waypoints_np[-1, 1]],
                                     marker="D", color='r',
                                     markertext="End", marker_text_offset=1)
            trajectory_fig.add_graph("car", window_size=1,
                                     marker="s", color='b', markertext="Car",
                                     marker_text_offset=1)

            forward_speed_fig = \
                lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
            forward_speed_fig.add_graph("forward_speed",
                                        label="forward_speed",
                                        window_size=TOTAL_EPISODE_FRAMES)
            forward_speed_fig.add_graph("reference_signal",
                                        label="reference_Signal",
                                        window_size=TOTAL_EPISODE_FRAMES)

            throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
            throttle_fig.add_graph("throttle",
                                   label="throttle",
                                   window_size=TOTAL_EPISODE_FRAMES)

            brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
            brake_fig.add_graph("brake",
                                label="brake",
                                window_size=TOTAL_EPISODE_FRAMES)

            steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
            steer_fig.add_graph("steer",
                                label="steer",
                                window_size=TOTAL_EPISODE_FRAMES)

            if not enable_live_plot:
                lp_traj._root.withdraw()
                lp_1d._root.withdraw()

        reached_the_end = False
        skip_first_frame = True
        closest_index = 0
        closest_distance = 0

        for frame in range(TOTAL_EPISODE_FRAMES):
            measurement_data, sensor_data = client.read_data()

            current_x, current_y, current_yaw = \
                get_current_pose(measurement_data)
            current_speed = measurement_data.player_measurements.forward_speed
            current_timestamp = float(
                measurement_data.game_timestamp) / 1000.0

            if current_timestamp <= WAIT_TIME_BEFORE_START:
                send_control_command(
                    client, throttle=0.0, steer=0, brake=1.0)
                continue
            else:
                current_timestamp = current_timestamp - \
                    WAIT_TIME_BEFORE_START

            # Update history lists if controller output is enabled
            if enable_controller_output:
                x_history.append(current_x)
                y_history.append(current_y)
                yaw_history.append(current_yaw)
                speed_history.append(current_speed)
                time_history.append(current_timestamp)

            # Find the closest waypoint
            closest_distance = np.linalg.norm(np.array([
                waypoints_np[closest_index, 0] - current_x,
                waypoints_np[closest_index, 1] - current_y]))
            new_distance = closest_distance
            new_index = closest_index

            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index = new_index
                new_index += 1
                if new_index >= waypoints_np.shape[0]:
                    break
                new_distance = np.linalg.norm(np.array([
                    waypoints_np[new_index, 0] - current_x,
                    waypoints_np[new_index, 1] - current_y]))

            new_distance = closest_distance
            new_index = closest_index

            while new_distance <= closest_distance:
                closest_distance = new_distance
                closest_index = new_index
                new_index -= 1
                if new_index < 0:
                    break
                new_distance = np.linalg.norm(np.array([
                    waypoints_np[new_index, 0] - current_x,
                    waypoints_np[new_index, 1] - current_y]))

            waypoint_subset_first_index = closest_index - 1
            if waypoint_subset_first_index < 0:
                waypoint_subset_first_index = 0
            waypoint_subset_last_index = closest_index
            total_distance_ahead = 0

            while total_distance_ahead < INTERP_LOOKAHEAD_DISTANCE:
                total_distance_ahead += wp_distance[waypoint_subset_last_index]
                waypoint_subset_last_index += 1
                if waypoint_subset_last_index >= waypoints_np.shape[0]:
                    waypoint_subset_last_index = waypoints_np.shape[0] - 1
                    break

            new_waypoints = \
                wp_interp[wp_interp_hash[waypoint_subset_first_index]:
                          wp_interp_hash[waypoint_subset_last_index] + 1]
            controller.update_waypoints(new_waypoints)

            controller.update_values(current_x, current_y, current_yaw,
                                     current_speed,
                                     current_timestamp, frame)
            controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()

            # Update figures and plots if controller output is enabled
            if enable_controller_output:
                if skip_first_frame and frame == 0:
                    pass
                else:
                    trajectory_fig.roll(
                        "trajectory", current_x, current_y)
                    trajectory_fig.roll("car", current_x, current_y)
                    new_waypoints_np = np.array(new_waypoints)
                    path_indices = np.floor(np.linspace(0,
                                                        new_waypoints_np.shape[0]-1,
                                                        INTERP_MAX_POINTS_PLOT))
                    trajectory_fig.update("lookahead_path",
                                          new_waypoints_np[path_indices.astype(int), 0],
                                          new_waypoints_np[path_indices.astype(int), 1],
                                          new_colour=[0, 0.7, 0.7])
                    forward_speed_fig.roll("forward_speed",
                                           current_timestamp,
                                           current_speed)
                    forward_speed_fig.roll("reference_signal",
                                           current_timestamp,
                                           controller._desired_speed)
                    throttle_fig.roll(
                        "throttle", current_timestamp, cmd_throttle)
                    brake_fig.roll(
                        "brake", current_timestamp, cmd_brake)
                    steer_fig.roll(
                        "steer", current_timestamp, cmd_steer)

                    if enable_live_plot and live_plot_timer.has_exceeded_lap_period():
                        lp_traj.refresh()
                        lp_1d.refresh()
                        live_plot_timer.lap()

            send_control_command(client,
                                 throttle=cmd_throttle,
                                 steer=cmd_steer,
                                 brake=cmd_brake)

            dist_to_last_waypoint = np.linalg.norm(np.array([
                waypoints[-1][0] - current_x,
                waypoints[-1][1] - current_y]))
            if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
                reached_the_end = True
            if reached_the_end:
                break

        if reached_the_end:
            print("Reached the end of path.")
        else:
            print("Exceeded assessment time.")

        send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)

        # Conditionally store plots and write output files
        if enable_controller_output:
            store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
            store_trajectory_plot(
                forward_speed_fig.fig, 'forward_speed.png')
            store_trajectory_plot(
                throttle_fig.fig, 'throttle_output.png')
            store_trajectory_plot(
                brake_fig.fig, 'brake_output.png')
            store_trajectory_plot(
                steer_fig.fig, 'steer_output.png')
            write_trajectory_file(
                x_history, y_history, speed_history, time_history)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA waypoint follower client.')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'
    while True:
        try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
