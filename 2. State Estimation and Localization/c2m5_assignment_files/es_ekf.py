# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt3_data.pkl', 'rb') as file:  # Open the pickle file containing Part 1 / 2 / 3 data.
    data = pickle.load(file)  # Load the data from the pickle file into the 'data' dictionary.

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']  # Ground truth data: contains acceleration, velocity, position, rotational acceleration, etc.
imu_f = data['imu_f']  # IMU specific force data in the vehicle frame.
imu_w = data['imu_w']  # IMU rotational velocity data in the vehicle frame.
gnss = data['gnss']  # GNSS position data.
lidar = data['lidar']  # LIDAR position data.

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()  # Create a new figure for the ground truth plot.
ax = gt_fig.add_subplot(111, projection='3d')  # Add a 3D subplot to the figure.
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])  # Plot the ground truth position trajectory.
ax.set_xlabel('x [m]')  # Label the x-axis.
ax.set_ylabel('y [m]')  # Label the y-axis.
ax.set_zlabel('z [m]')  # Label the z-axis.
ax.set_title('Ground Truth trajectory')  # Set the title of the plot.
ax.set_zlim(-1, 5)  # Set limits for the z-axis.
plt.show()  # Display the plot.

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],  # Rotation matrix element [0, 0], [0, 1], [0, 2].
   [ 0.09971,  0.99401, -0.04475],  # Rotation matrix element [1, 0], [1, 1], [1, 2].
   [-0.04998,  0.04992,  0.9975 ]   # Rotation matrix element [2, 0], [2, 1], [2, 2].
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
# C_li = np.array([
#      [ 0.9975 , -0.04742,  0.05235],
#      [ 0.04992,  0.99763, -0.04742],
#      [-0.04998,  0.04992,  0.9975 ]
# ])

t_i_li = np.array([0.5, 0.1, 0.5])  # Translation vector from the LIDAR frame to the IMU frame.

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li   # Apply the rotation matrix and translation to transform the LIDAR data.

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.10  # Variance for the IMU specific force.
var_imu_w = 0.25  # Variance for the IMU rotational velocity.
var_gnss  = 0.01  # Variance for the GNSS position.
var_lidar = 1.00  # Variance for the LIDAR position.


################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # Gravity vector (downward acceleration).
l_jac = np.zeros([9, 6])  # Motion model noise Jacobian, initialized as zeros.
l_jac[3:, :] = np.eye(6)  # Populate the bottom part of the Jacobian with an identity matrix.
h_jac = np.zeros([3, 9])  # Measurement model Jacobian, initialized as zeros.
h_jac[:, :3] = np.eye(3)  # Populate the top-left corner of the Jacobian with an identity matrix.


#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # Position estimates initialized as zeros.
v_est = np.zeros([imu_f.data.shape[0], 3])  # Velocity estimates initialized as zeros.
q_est = np.zeros([imu_f.data.shape[0], 4])  # Orientation estimates initialized as zeros (quaternions).
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # Covariance matrices initialized as zeros.

# Set initial values.
p_est[0] = gt.p[0]  # Initialize position estimate with the ground truth position.
v_est[0] = gt.v[0]  # Initialize velocity estimate with the ground truth velocity.
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()  # Initialize orientation estimate with the ground truth Euler angles.
p_cov[0] = np.zeros(9)  # Initialize the covariance of the estimate as zeros.
gnss_i  = 0  # Index for GNSS data.
lidar_i = 0  # Index for LIDAR data.

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain    
    I = np.eye(3)  # Identity matrix for the measurement model.
    R = I * sensor_var  # Measurement noise covariance matrix.
    K = p_cov_check @ h_jac.T @ np.linalg.inv(h_jac @ p_cov_check @ h_jac.T + R)  # Kalman gain.

    # 3.2 Compute error state
    error_state = K @ (y_k - p_check)  # Error state computed from the difference between measurement and prediction.

    # 3.3 Correct predicted state
    delta_position = error_state[:3]  # Extract position error.
    delta_velocity = error_state[3:6]  # Extract velocity error.
    delta_orientation = error_state[6:]  # Extract orientation error.

    p_hat = p_check + delta_position  # Corrected position estimate.
    v_hat = v_check + delta_velocity  # Corrected velocity estimate.
    q_hat = Quaternion(euler=delta_orientation).quat_mult_right(q_check)  # Corrected orientation estimate (as quaternion).

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(9) - K @ h_jac) @ p_cov_check  # Corrected covariance matrix.

    return p_hat, v_hat, q_hat, p_cov_hat  # Return the corrected estimates and covariance.

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
for k in range(1, imu_f.data.shape[0]):  # Loop through each time step, starting at 1 because we have initial prediction from gt.
    delta_t = imu_f.t[k] - imu_f.t[k - 1]  # Time difference between the current and previous time steps.

    # 1. Update state with IMU inputs
    rotation_matrix = Quaternion(*q_est[k-1]).to_mat()  # Convert the previous quaternion to a rotation matrix.

    # 1.1 Linearize the motion model and compute Jacobians
    p_est[k] = p_est[k-1] + delta_t * v_est[k-1] + 0.5 * (delta_t**2) * (rotation_matrix @ imu_f.data[k-1] + g)  # Position update.
    v_est[k] = v_est[k-1] + delta_t * (rotation_matrix @ imu_f.data[k-1] + g)  # Velocity update.
    q_est[k] = Quaternion(axis_angle=imu_w.data[k-1] * delta_t).quat_mult_right(q_est[k-1])  # Orientation update.

    # 2. Propagate uncertainty
    F = np.eye(9)  # State transition model.
    Q = np.eye(6)  # Process noise covariance matrix.
    F[:3, 3:6] = delta_t * np.eye(3)  # Update position-velocity relationship in the state transition model.
    F[3:6, 6:] = -rotation_matrix @ skew_symmetric(imu_f.data[k-1].reshape((3, 1))) * delta_t  # Update velocity-orientation relationship.
    Q[:3, :3] *= delta_t**2 * var_imu_f  # Update process noise covariance for position.
    Q[3:, 3:] *= delta_t**2 * var_imu_w  # Update process noise covariance for orientation.
    p_cov[k] = F @ p_cov[k-1] @ F.T + l_jac @ Q @ l_jac.T  # Propagate the covariance matrix.

    # 3. Check availability of GNSS and LIDAR measurements
    if lidar_i < lidar.t.shape[0] and lidar.t[lidar_i] == imu_f.t[k-1]:  # Check if there is a LIDAR measurement at this time step.
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar, p_cov[k], lidar.data[lidar_i].T, p_est[k], v_est[k], q_est[k])  # Perform measurement update with LIDAR.
        lidar_i += 1  # Increment the LIDAR index.
    if gnss_i < gnss.t.shape[0] and gnss.t[gnss_i] == imu_f.t[k-1]:  # Check if there is a GNSS measurement at this time step.
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[gnss_i].T, p_est[k], v_est[k], q_est[k])  # Perform measurement update with GNSS.
        gnss_i += 1  # Increment the GNSS index.

    # Update states (save)

#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()  # Create a new figure for the estimated trajectory plot.
ax = est_traj_fig.add_subplot(111, projection='3d')  # Add a 3D subplot to the figure.
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')  # Plot the estimated trajectory.
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')  # Plot the ground truth trajectory.
ax.set_xlabel('Easting [m]')  # Label the x-axis.
ax.set_ylabel('Northing [m]')  # Label the y-axis.
ax.set_zlabel('Up [m]')  # Label the z-axis.
ax.set_title('Ground Truth and Estimated Trajectory')  # Set the title of the plot.
ax.set_xlim(0, 200)  # Set limits for the x-axis.
ax.set_ylim(0, 200)  # Set limits for the y-axis.
ax.set_zlim(-2, 2)  # Set limits for the z-axis.
ax.set_xticks([0, 50, 100, 150, 200])  # Set x-axis tick marks.
ax.set_yticks([0, 50, 100, 150, 200])  # Set y-axis tick marks.
ax.set_zticks([-2, -1, 0, 1, 2])  # Set z-axis tick marks.
ax.legend(loc=(0.62,0.77))  # Add a legend to the plot.
ax.view_init(elev=45, azim=-50)  # Set the view angle of the plot.
plt.show()  # Display the plot.

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)  # Create a figure with a 2x3 grid of subplots for error plots.
error_fig.suptitle('Error Plots')  # Set the title for the entire figure.
num_gt = gt.p.shape[0]  # Number of ground truth data points.
p_est_euler = []  # List to store estimated Euler angles.
p_cov_euler_std = []  # List to store standard deviations of Euler angle estimates.

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])  # Extract quaternion for each time step.
    p_est_euler.append(qc.to_euler())  # Convert quaternion to Euler angles.

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())  # Compute Jacobian for Euler angle covariance.
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))  # Calculate the standard deviation.

p_est_euler = np.array(p_est_euler)  # Convert the list of Euler angles to a NumPy array.
p_cov_euler_std = np.array(p_cov_euler_std)  # Convert the list of standard deviations to a NumPy array.

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))  # Calculate standard deviations for position and velocity.

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']  # Titles for each of the 6 plots.
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])  # Plot the error in position (ground truth - estimate).
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')  # Plot the +3σ uncertainty bound.
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')  # Plot the -3σ uncertainty bound.
    ax[0, i].set_title(titles[i])  # Set the title for each plot.
ax[0,0].set_ylabel('Meters')  # Label the y-axis for position plots.

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))  # Plot the error in orientation (ground truth - estimate).
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')  # Plot the +3σ uncertainty bound.
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')  # Plot the -3σ uncertainty bound.
    ax[1, i].set_title(titles[i+3])  # Set the title for each plot.
ax[1,0].set_ylabel('Radians')  # Label the y-axis for orientation plots.
plt.show()  # Display the plots.

#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
# p1_indices = [9000, 9400, 9800, 10200, 10600]  # Indices to extract for Part 1 submission.
# p1_str = ''  # Initialize the string to store the submission data.
# for val in p1_indices:
#     for i in range(3):
#         p1_str += '%.3f ' % (p_est[val, i])  # Format the position estimates to 3 decimal places.
# with open('pt1_submission.txt', 'w') as file:  # Open a file to save the submission data.
#     file.write(p1_str)  # Write the data to the file.

# Pt. 2 submission
# p2_indices = [9000, 9400, 9800, 10200, 10600]  # Indices to extract for Part 2 submission.
# p2_str = ''  # Initialize the string to store the submission data.
# for val in p2_indices:
#     for i in range(3):
#         p2_str += '%.3f ' % (p_est[val, i])  # Format the position estimates to 3 decimal places.
# with open('pt2_submission.txt', 'w') as file:  # Open a file to save the submission data.
#     file.write(p2_str)  # Write the data to the file.

# Pt. 3 submission
p3_indices = [6800, 7600, 8400, 9200, 10000]  # Indices to extract for Part 3 submission.
p3_str = ''  # Initialize the string to store the submission data.
for val in p3_indices:
    for i in range(3):
        p3_str += '%.3f ' % (p_est[val, i])  # Format the position estimates to 3 decimal places.
with open('pt3_submission.txt', 'w') as file:  # Open a file to save the submission data.
    file.write(p3_str)  # Write the data to the file.