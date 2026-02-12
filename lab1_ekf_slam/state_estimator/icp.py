from .ekf import *
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
class ICP():
    def __init__(self):
        self.y = np.array([]) # current scan
        self.x = np.array([]) # prev scan

        self.EPS = 0.005
        self.MAX_ITER = 30

        # Odom param
        self.x, self.y, self.theta = 0, 0, 0

    def data_association(self, prev_point, curr_point):
        tree = KDTree(prev_point.T)
        distances, indexes = tree.query(curr_point.T)
        error = np.mean(distances)
        return indexes, error

    def svd_motion_estimation(self, previous_points, current_points):
        # Calculate centroid
        pm = np.mean(previous_points, axis=1)
        cm = np.mean(current_points, axis=1)
        # Shifted data
        p_shift = previous_points - pm[:, np.newaxis]
        c_shift = current_points - cm[:, np.newaxis]
        
        W = c_shift @ p_shift.T
        
        u, s, vh = np.linalg.svd(W)
        R = (u @ vh).T
        # R = vh.T @ u

        if np.linalg.det(R) < 0:
            vh[1, :] *= -1
            R = (u @ vh).T

        t = pm - (R @ cm)
        return R, t
    
    def update_homogeneous_matrix(self, Hin, R, T):
        r_size = R.shape[0]
        H = np.eye(r_size + 1)
        H[0:r_size, 0:r_size] = R
        H[0:r_size, r_size] = T

        if Hin is None:
            return H
        else:
            return Hin @ H
        

    def icp_matching(self, previous_points, current_points, initial_R=None, initial_T=None):
        '''
        Iterative Closest Point matching
        - input
        previous_points: 2D or 3D points in the previous frame
        current_points: 2D or 3D points in the current frame
        - output
        R: Rotation matrix
        T: Translation vector
        '''
        if previous_points is None or current_points is None:
            return np.eye(2), np.zeros(2), False
        if previous_points.shape[1] < 5 or current_points.shape[1] < 5:
            # print("Not enough points for ICP")
            return np.eye(2), np.zeros(2), False
        
        if initial_R is not None and initial_T is not None:
            current_points = (initial_R @ current_points) + initial_T[:, np.newaxis]
            H = self.update_homogeneous_matrix(None, initial_R, initial_T)
        else:
            H = np.eye(3)
        
        
        dError = np.inf
        preError = np.inf
        count = 0


        while dError >= self.EPS and count < self.MAX_ITER:
            count += 1
            indexes, error = self.data_association(previous_points, current_points)
            Rt, Tt = self.svd_motion_estimation(previous_points[:, indexes], current_points)

            current_points = (Rt @ current_points) + Tt[:, np.newaxis]

            dError = preError - error
            if dError < 0: break
            
            preError = error
            H = self.update_homogeneous_matrix(H, Rt, Tt)

        R = H[0:2, 0:2]
        T = H[0:2, 2]
        success = (dError < self.EPS) and (count < self.MAX_ITER)
        print(success , count , dError)
        return R, T , success
    
