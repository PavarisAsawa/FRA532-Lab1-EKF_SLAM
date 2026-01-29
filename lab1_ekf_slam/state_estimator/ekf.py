import numpy as np
import math

class EKF():
    def __init__(self):
        self.r = 0.033 # Robot wheel radius 33 mm
        self.dt = 0.05
        self.L = 0.105 # Distance between wheel 105 mm

        # Odom param
        self.x = 0
        self.y = 0
        self.theta = 0

        # Store previous encoder/joint positions
        self.last_joint_pos = None

        # EKF Hyperparam (NEED FINE TUNING)
        # predicted state covariance matrix (assume orthogonal XY)
        self.PPred = np.array([
            [ 0.01 , 0 , 0],
            [ 0 , 0.01 , 0],
            [ 0 , 0 , 0.05],
        ])

        # self.PEst = np.eye(4)
        self.PEst = np.array([  # Create buffer for estimated P
            [ 0.01 , 0 , 0],
            [ 0 , 0.01 , 0],
            [ 0 , 0 , 0.02],
        ])
         
        self.Q = np.array([  # Process noise
            [ 0.01 , 0 , 0],
            [ 0 , 0.01 , 0],
            [ 0 , 0 , 0.05],
        ])
        
        self.R = np.array([  # Measurement noise
            [ 0.1 , 0 , 0],
            [ 0 , 0.1 , 0],
            [ 0 , 0 , 0.1],
        ])
        
    def cal_u(self,observation):
        """
        Calculates odometry based on absolute joint positions (rad).
        current_joint_pos: [left_rad, right_rad]
        """
        if self.last_joint_pos is None:
            self.last_joint_pos = observation
            return self.x, self.y, self.theta
        
        # atan2 for handles encoder "rollover" & calculate difference angle
        d_phi_left = math.atan2(math.sin(observation[0] - self.last_joint_pos[0]), 
                                math.cos(observation[0] - self.last_joint_pos[0]))
        d_phi_right = math.atan2(math.sin(observation[1] - self.last_joint_pos[1]), 
                                 math.cos(observation[1] - self.last_joint_pos[1]))
        self.last_joint_pos = observation # update position for the next iteration
        return [d_phi_left,d_phi_right]

    def motion_model(self, state , u):
        """
        Calculates odometry based on absolute joint positions (rad).
        current_joint_pos: [delta of LEFT wheel position, delta of RIGHT wheel position]
        """
        # state = [x,y,theta]
        state[0] += self.r*(u[0] + u[1]) / 2.0 * math.cos(state[2] + self.r *(u[1] - u[0])/(2.0*self.L))
        state[1] += self.r*(u[0] + u[1]) / 2.0 * math.sin(state[2] + self.r *(u[1] - u[0])/(2.0*self.L))
        state[2] += self.r*(u[1] - u[0]) / self.L

        state[2] = math.atan2(math.sin(state[2]), math.cos(state[2])) # Normalize theta to [-pi, pi]
        return state
    
    def motion_jacobian(self, u):
        """
        Calculating Jacobian based on current state and input        
        dx/dtheta = -r(u0-u1)/2 * sin(theta+(r*(u1-u2))/(2L))
        dy/dtheta = r(u0-u1)/2 * cos(theta+(r*(u1-u2))/(2L))
        """
        jacobian = np.array([
            [ 1.0   , 0.0   , -(self.r*(u[0]-u[1])/2)*math.sin(self.theta + (self.r *(u[1]-u[0]))/(2*self.L))],
            [ 0.0   , 1.0   , (self.r*(u[0]-u[1])/2)*math.cos(self.theta + (self.r *(u[1]-u[0]))/(2*self.L))],
            [ 0.0   , 0.0   , 1.0]
        ])
        return jacobian
    
    # ---------------------------------------------------------------------------------------------------------------------

    def observation_model(self,observation):
        """
            observation: IMU data: [IMU orientation (r,p,y)]
        """
        # state = [x,y,yaw]^T
        observation = np.array(observation)
        H = np.array([
            [0 ,0 , 1]
        ])
        z = H @ observation
        return z
    
    def observation_jacobian(self):
        return np.array([
            [0 ,0 , 0],
            [0 ,0 , 0],
            [0 ,0 , 1]
        ])

    def predict(self, u,z):
        """
        State Estimator with EKF!
        u : joint position(rad) : [left , right]
        z : sensor data(IMU data) : [r,p,y]
        """
        # ----------- Predict
        u = self.cal_u(u)
        xPred = self.motion_model([self.x,self.y,self.theta] , u) # Predict state
        jF = self.motion_jacobian(u) # calculating jacobian of motion model
        # self.x,self.y,self.theta = xPred
        # return xPred
        self.PPred = jF @ self.PEst @ jF.T + self.Q # predict state uncertainty

        # ----------- Update
        jH = self.observation_jacobian()
        zPred = self.observation_model(xPred) # from predict state, what should you read with sensor
        y = z - zPred # error
        S = jH @ self.PPred @ jH.T + self.R
        K = self.PPred @ jH.T @ np.linalg.inv(S) # Kalman Gain

        xEst = xPred + K @ y
        self.PEst = (np.eye(len(xEst)) - K @ jH) @ self.PPred

        self.x,self.y,self.theta = xEst
        return xEst

