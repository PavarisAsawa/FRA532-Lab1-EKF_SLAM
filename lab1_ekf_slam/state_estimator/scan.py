import numpy as np
from scipy.spatial import KDTree

def scan_to_pointcloud(LaserScanMsg):
    ranges = np.array(LaserScanMsg.ranges)
    angles = LaserScanMsg.angle_min + np.arange(len(ranges)) * LaserScanMsg.angle_increment
    
    valid_indices = (ranges >= LaserScanMsg.range_min) & (ranges <= LaserScanMsg.range_max)
    r = ranges[valid_indices]
    phi = angles[valid_indices]
    
    # Convert Polar to Cartesian coordinates (x, y, z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.zeros_like(x) # 2D scan usually has 0 for height

    # point_cloud = np.vstack((x, y, z))
    point_cloud = np.stack((x, y),axis=0).T
    # print(point_cloud.shape)
    
    return point_cloud

def pointcloud_to_odom(point_cloud, x , y, theta):
    X = point_cloud[:, 0]
    Y = point_cloud[:, 1]

    c = np.cos(theta)
    s = np.sin(theta)

    Xw = c * X - s * Y + x
    Yw = s * X + c * Y + y
    return np.column_stack((Xw, Yw))   # shape (N, 2)

def v2T(dx,dy,dyaw):
    # To transformation matrix
    c, s = np.cos(dyaw), np.sin(dyaw)
    return np.array([[c, -s, dx],
                     [s,  c, dy],
                     [0,  0, 1]], dtype=float)

def voxel_grid_filter(points, leaf_size=0.05):
        """ 
        points: numpy array (2, N) 
        leaf_size:  voxel size (m)
        """
        if points.shape[0] == 0:
            return points

        grid_pts = np.round(points / leaf_size) * leaf_size
        
        _, unique_indices = np.unique(grid_pts, axis=0, return_index=True)
        
        return points[unique_indices, :] # [n , 2]
 
def remove_outliers(points, radius=0.2, min_neighbors=5):
        """
        Outlier Removal
        - points: numpy array (N, 2)
        - radius: radius for seek (เมตร)
        - min_neighbors: minimum neighbour
        """
        if points.shape[0] < min_neighbors:
            return points
        tree = KDTree(points) 
        
        neighbors_count = tree.query_ball_point(points, r=radius, return_length=True)
        neighbors_count = np.array(neighbors_count)
        
        mask = neighbors_count > min_neighbors 
        
        return points[mask, :]

def wrap(a):
    """    
    Preventing theta wraping
    """
    return np.arctan2(np.sin(a), np.cos(a))

def se2_compose(a, b):
    ax, ay, ath = a
    bx, by, bth = b
    c, s = np.cos(ath), np.sin(ath)
    x = ax + c*bx - s*by
    y = ay + s*bx + c*by
    th = wrap(ath + bth)
    return np.array([x, y, th])

def se2_inverse(a):
    x, y, th = a
    c, s = np.cos(th), np.sin(th)
    xi = -(c*x + s*y)
    yi = -(-s*x + c*y)
    thi = wrap(-th)
    return np.array([xi, yi, thi])