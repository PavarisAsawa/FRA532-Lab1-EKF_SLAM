import numpy as np
import math

class DiffDriveIMU():
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

    def cal_u(self,observation):
        """
        Calculates odometry based on absolute joint positions (rad).
        observation: [left_rad, right_rad]
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

    def update(self,x,u,z_yaw):
        """
        motion model
        x : [x,y,theta]
        """
        x[0] += self.r*(u[0] + u[1]) / 2.0 * math.cos(x[2] + self.r *(u[1] - u[0])/(2.0*self.L))
        x[1] += self.r*(u[0] + u[1]) / 2.0 * math.sin(x[2] + self.r *(u[1] - u[0])/(2.0*self.L))
        x[2] = z_yaw

        x[2] = math.atan2(math.sin(x[2]), math.cos(x[2])) # Normalize theta to [-pi, pi]
        return x[0],x[1],x[2]
        # return self.x, self.y, self.theta

    def predict(self, observation ,z_yaw):
        u = self.cal_u(observation)

        self.x,self.y,self.theta = self.update(x=[self.x,self.y,self.theta],u=u,z_yaw=z_yaw)

        return self.x, self.y, self.theta