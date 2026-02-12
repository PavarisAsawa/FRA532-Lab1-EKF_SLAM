import numpy as np
from scipy.spatial import KDTree

def data_association(previous_points , current_points, max_corr_dist=None):
    # (2 , n)
    previous_points = previous_points
    current_points = current_points

    tree = KDTree(previous_points)
    dists, idx = tree.query(current_points, k=1)
    previous_points_match = previous_points[idx,:]

    if max_corr_dist is not None:
        mask = dists < max_corr_dist
        return current_points[mask], previous_points_match[mask], dists[mask], idx[mask]
    return current_points, previous_points_match, dists, idx

def icp_matching(previous_points, current_points,init_x=None, init_y=None, init_theta=None , MAX_ITERATION=10,ERROR_BOUND=0.02):
    """
    Iterative Closest Point matching
    - input
    previous_points: 2D points in the previous frame / Can be map, odom frame
    current_points: 2D points in the current frame , base frame
    - output
    R: Rotation matrix
    T: Translation vector
    """
    prevError = np.inf
    H = None  # homogeneous transformation matrix
    x , y , theta = init_x,init_y,init_theta
    count = 0
    for i in range(MAX_ITERATION):
        count += 1

        
        # --- Initial Guess
        if init_x is not None and init_y is not None and init_theta is not None:
            # Transform current scan to current pose
            initial_guess = np.column_stack(
                [
                    current_points[:,0] * np.cos(theta) - current_points[:,1] * np.sin(theta) + x,
                    current_points[:,0] * np.sin(theta) + current_points[:,1] * np.cos(theta) + y, 
                ]
            )
            current_points = initial_guess

        # --- Data Asso
        _ , prev_match , dist, idx = data_association(previous_points,current_points) # find 
        
        error = np.mean(dist) # Find error
        
        if(prevError - error) < ERROR_BOUND:
            break
        
        prevError = error

        # ----- SVD
        source_point = current_points
        target_point = prev_match

        source_centroid = np.mean(current_points,axis=0) # base_frame
        target_centroid = np.mean(prev_match,axis=0) # odom frame

        W = (target_centroid - target_point).T @ (source_centroid - source_point)
        U, _ , Vt = np.linalg.svd(W)

        d = np.linalg.det(U @ Vt)
        R = U @ np.diag([1.0 , np.sign(d)]) @ Vt

        T = target_centroid - R @ source_centroid

        x = T[0]
        y = T[1]
        theta = np.arctan2(R[1,0] , R[0,0])
    success = error < ERROR_BOUND and count < MAX_ITERATION
    # print(x,y,theta,count)

    return x , y , theta, count , error , success

# def icp_matching(previous_points, current_points,
#                  init_x=0.0, init_y=0.0, init_theta=0.0,
#                  MAX_ITERATION=20, ERROR_BOUND=1e-4, max_corr_dist=0.3):

#     # --- Copy
#     prev_pts = previous_points.copy()
#     curr_pts = current_points.copy()

#     # --- Apply initial guess ONCE
#     c, s = np.cos(init_theta), np.sin(init_theta)
#     R = np.array([[c, -s],
#                   [s,  c]])
#     t = np.array([init_x, init_y])

#     curr_pts = (R @ curr_pts.T).T + t

#     prevError = np.inf
#     count = 0

#     for _ in range(MAX_ITERATION):
#         count += 1

#         # 1) Data association
#         src, tgt, dists, index = data_association(prev_pts, curr_pts, max_corr_dist=max_corr_dist)
#         if src.shape[0] < 3:
#             break

#         error = np.mean(dists)

#         # stopping condition
#         if abs(prevError - error) < ERROR_BOUND:
#             prevError = error
#             break
#         prevError = error

#         # 2) Compute best-fit transform (SVD / Kabsch)
#         src_cent = src.mean(axis=0)
#         tgt_cent = tgt.mean(axis=0)

#         src0 = src - src_cent
#         tgt0 = tgt - tgt_cent

#         H = src0.T @ tgt0
#         U, _, Vt = np.linalg.svd(H)
#         d = np.linalg.det(Vt.T @ U.T)

#         dR = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T
#         dt = tgt_cent - dR @ src_cent

#         # 3) Apply incremental update to ALL current points
#         curr_pts = (dR @ curr_pts.T).T + dt

#         # 4) Accumulate total transform
#         R = dR @ R
#         t = dR @ t + dt

#     theta = np.arctan2(R[1, 0], R[0, 0])
#     success = (count < MAX_ITERATION) and (prevError < np.inf)

#     return t[0], t[1], theta, count, prevError, success


