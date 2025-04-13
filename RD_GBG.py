import math
from collections import Counter
import numpy as np

""" GranularBall and GranularBallManager classes for managing granular balls.
    These classes provide methods for generating granular balls without overlapping"""
class GranularBall:
    def __init__(self, data) -> None:
        self.data = data
        self.center = []    
        self.radius = 0.0
        self.num = len(data)
        self.label = 0
        
    def distance_to_point(self, point):
        """Calculate the distance from a point to a GranularBall"""
        return np.sqrt(np.sum((point - self.center)**2)) - self.radius


class GranularBallManager:
    """Manager class for GranularBall"""
    def __init__(self):
        self.balls = []
        
    def calculate_dist(self, A, B, vector_mode=True):
        """Calculate the distance between two points"""
        if vector_mode:
            return np.sqrt(np.sum((A - B)**2, axis=1))
        else:
            return np.sqrt(np.sum((A - B)**2))
    
    def calculate_point_to_balls_dist(self, point, ball_list):
        """Calculate distances (representing the distance to the ball's surface) from a point to a list of GranularBalls"""
        # Extract point features (excluding label and index)
        point_features = point[0][1:-1]
        
        # Calculate distances from the point to each ball in the list
        distances = [ball.distance_to_point(point_features) for ball in ball_list]
        
        return distances
    
    def calculate_centers(self, data):
        """Calculate mean vectors for each class as cluster centers.
        Classes are processed in descending order of frequency."""
        # Get counts of each class label
        label_counts = Counter(data[:, 0])
        # Sort labels by frequency (descending)
        sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
        center_list = []
        # Calculate centroid for each class
        for label in sorted_labels:
            # Extract feature vectors (excluding label and index)
            class_features = data[data[:, 0] == label][:, 1:-1]
            # Calculate mean feature vector (centroid)
            centroid = np.mean(class_features, axis=0)
            # Store as [label, centroid] pair
            center_list.append([label, centroid])
        return center_list
    
    def random_centers(self, data, lowDensity):
        """Randomly select center points from the data for each class.
        Exclude points in lowDensity."""
        center_list = []
        # Handle empty lowDensity input
        if not lowDensity:
            lowDensity_points = []
        else:
            # Convert to numpy array if not already
            lowDensity = np.array(lowDensity)
            # Extract point indices based on array shape
            if len(lowDensity.shape) > 1 and lowDensity.shape[0] > 1:
                lowDensity_points = lowDensity[:, 0]
            elif len(lowDensity.shape) > 1 and lowDensity.shape[0] == 1:
                lowDensity_points = [lowDensity[0][0]]
            else:
                lowDensity_points = []
        filtered_data = data.copy()
        # Remove low density points from consideration
        for index in lowDensity_points:
            if index in filtered_data[:, -1]:
                filtered_data = filtered_data[filtered_data[:, -1] != index]
        if len(filtered_data) == 0:
            return center_list
        # Get counts of each class label and sort by frequency
        label_counts = Counter(filtered_data[:, 0])
        sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
        # Select one random center for each class
        for label in sorted_labels:
            # Get all samples of the current class
            class_samples = filtered_data[filtered_data[:, 0] == label]
            # Skip if no samples for this class
            if len(class_samples) == 0:
                continue
            # Randomly select one sample as center
            center_index = np.random.choice(len(class_samples), size=1, replace=False)[0]
            center = class_samples[center_index:center_index+1]  # Keep as 2D array
            center_list.append(center)
        return center_list
    
    def detect_outliers(self, dataDis_sort, k, center_label):
        """Detect if a point is an outlier based on its k nearest neighbors.
    This method examines the k nearest points to determine if the center point
    is an outlier, normal point, or in a low-density region.
    Args:
        dataDis_sort: Sorted data array with points and distances
        k: Number of neighbors to consider
        center_label: The class label of the center point
    Returns:
        0: Point is an outlier (no homogeneous neighbors)
        1: Point is normal (all k neighbors are homogeneous)
        2: Point is in a low-density region (mixed neighborhood)"""
        # Get the k nearest points
        k_nearest = dataDis_sort[:k, :]
        
        # Count homogeneous neighbors (points with the same label)
        # Subtract 1 to exclude the center point itself
        homogeneous_count = np.sum(k_nearest[:, 0] == center_label) - 1
        
        # Determine point type based on homogeneous neighbor count
        if homogeneous_count == 0:
            return 0  # Outlier: No neighbors with the same label
        elif homogeneous_count == k - 1:
            return 1  # Normal: All neighbors have the same label
        else:
            return 2  # Low-density region: Mixed neighborhood
    
    def generate_granular_balls(self, ball_list, lowDensity, data, k):
        """Generate granular balls from data points.
    This method constructs granular balls by selecting random centers and determining
    appropriate radii to include homogeneous points while maintaining boundaries
    between different classes.
    
    Args:
        ball_list: List of existing granular balls
        lowDensity: List of points marked as low density
        data: Input data with format [label, features..., index]
        k: Parameter for neighborhood evaluation
        
    Returns:
        tuple: (new_balls, remaining_data, end_flag)
            - new_balls: List of newly created granular balls
            - remaining_data: Data points not included in any ball
            - end_flag: 1 if process should terminate, 0 otherwise"""
        
        # Initialize return values
        new_balls = []
        end_flag = 0
        tmp_ball_list = ball_list.copy() # Create a copy to avoid modifying original list
        undo_data = np.array(data)
        # Select random centers for potential new balls
        center_list = self.random_centers(data, lowDensity)

        # If no valid centers found, signal termination       
        if len(center_list) == 0:
            return new_balls, undo_data, 1
        # Process each center to create a granular ball            
        for center in center_list:
            min_dis_ball = math.inf
            tmp_radius = math.inf
            center_label = center[0][0]
            
            dis_list = np.array(self.calculate_dist(center[0][1:-1], undo_data[:,1:-1])) # Calculate distances from center to all remaining data points
            data_dis = np.concatenate((undo_data, dis_list.reshape(len(dis_list),1)), axis=1) # Attach distance values to data points
            dataDis_sort = data_dis[data_dis[:,-1].argsort()] # Sort data points by distance from center
            heter_indexlist = np.argwhere(dataDis_sort[:,0] != center_label) # Find points with different class label (heterogeneous points)
            # Determine radius based on distance to nearest heterogeneous point
            if len(heter_indexlist) != 0:
                heter_index = heter_indexlist[0][0]
                if heter_index == 1: # Special case: if nearest heterogeneous point is very close (index=1)
                    # Check if center point is outlier, normal, or in low density region
                    OOD_flag = self.detect_outliers(dataDis_sort, k, center_label) 
                    if OOD_flag == 0: # Outlier case
                        undo_data = undo_data[undo_data[:,-1] != center[0][-1]] # Remove center point from consideration and continue
                        continue
                    elif OOD_flag == 2: # Low density region
                        lowDensity.append([center[0][-1], 0]) # Mark as low density point and continue
                        continue
                    else:  # Normal point
                        dataDis_sort = np.delete(dataDis_sort, 1, 0) # Remove the first point (center point) from consideration
                        # Find the first heterogeneous point after the center point
                        heter_indexlist = np.argwhere(dataDis_sort[:,0] != center_label) # Find points with different class label (heterogeneous points)
                        # If no heterogeneous points found, use the last point's distance as radius
                        # Otherwise, use the distance to the first heterogeneous point as radius
                        if len(heter_indexlist) != 0: 
                            heter_index = heter_indexlist[0][0]
                            tmp_radius = dataDis_sort[heter_index-1][-1]
                        else:
                            tmp_radius = dataDis_sort[-1,-1]
                else:
                    # Regular case: use distance to last homogeneous point as radius
                    tmp_radius = dataDis_sort[heter_index-1][-1]
            else: # All points have same label, use max distance as radius
                tmp_radius = dataDis_sort[-1,-1]
            # Calculate minimum distance to existing balls    
            if len(tmp_ball_list) != 0:
                min_dis_ball = np.min(self.calculate_point_to_balls_dist(center, tmp_ball_list))
            else:
                min_dis_ball = math.inf
            # Determine final radius and data points to include in the ball    
            if tmp_radius <= min_dis_ball: # Case: nearest heterogeneous point is closer than existing balls
                radius = tmp_radius
                gb_data = dataDis_sort[:heter_index,:-1]
                undo_data = dataDis_sort[heter_index:,:-1]
            else: # Case: existing ball boundary constrains new ball size
                index_list = np.argwhere(dataDis_sort[:,-1] <= min_dis_ball)
                if len(index_list) <= 1: ## Only the center point is within the radius
                    # Too few points within allowed radius, mark as low density
                    lowDensity.append([center[0][-1], 1])
                    continue
                else:  # Use maximum allowed radius without overlapping existing balls
                    max_index = np.max(index_list)
                    radius = dataDis_sort[max_index][-1]
                    gb_data = dataDis_sort[:max_index+1,:-1]
                    undo_data = dataDis_sort[max_index+1:,:-1]
            # Create new granular ball and add to result list        
            new_ball = GranularBall(gb_data)
            new_ball.radius = radius
            new_ball.center = center[0][1:-1]
            new_ball.label = center_label
            new_balls.append(new_ball)
            tmp_ball_list.append(new_ball)  # Add new ball to temporary list for distance calculations
            
        return new_balls, undo_data, end_flag
    
    def generate_granular_ball_list(self, data, k):
        """Generate a list of granular balls through iterative construction.
    This method iteratively creates granular balls from the input data until
    all points are processed or no more valid balls can be created.
    
    Args:
        data: Input data array with format [label, features..., index]
        k: Parameter for neighborhood evaluation
        
    Returns:
        tuple: (ball_list, centerlist)
            - ball_list: List of all created granular balls
            - centerlist: List of centers of all balls"""
        ball_list = []
        undo_data = data
        lowDensity = []
        centerlist = []
        # Main iteration loop - continue until no more balls can be created
        while True:
            # Generate new granular balls from remaining data
            new_balls, undo_data, flag = self.generate_granular_balls(ball_list, lowDensity, undo_data, k)
            if flag == 1: # End loop if process signals termination
                break
            # Process newly created balls                
            if len(new_balls) != 0:
                for ball in new_balls:
                    if ball.num > 0: # Only add balls that contain data points
                        ball_list.append(ball)
                        centerlist.append(ball.center)
            # Check termination condition: each remaining data point belongs to a different class                        
            undo_category = np.unique(undo_data[:,0])
            if len(undo_data) == len(undo_category): ## All remaining points are unique classes
                break
        # Process remaining low density points that were marked for individual ball creation                
        lowDensity = np.array(lowDensity)
        if len(lowDensity) > 0 and len(undo_data) > 0:
            for sample_index in lowDensity[lowDensity[:,-1]==1][:,0]:
                if sample_index in undo_data[:,-1]:  
                    ball = GranularBall(data[data[:,-1]==sample_index])
                    ball.center = ball.data[0][1:-1]
                    ball.label = ball.data[0][0]
                    centerlist.append(ball.center)
                    ball_list.append(ball)
                    undo_data = undo_data[undo_data[:,-1]!=sample_index] # Remove processed point from consideration
                    
        self.balls = ball_list
        return ball_list, centerlist


# Main functions for external use
# These functions provide a simplified interface to the GranularBallManager class
def calculateDist(A, B, flag=0):
    manager = GranularBallManager()
    return manager.calculate_dist(A, B, flag)

def calPo2ballsDis(point, GB_List):
    manager = GranularBallManager()
    return manager.calculate_point_to_balls_dist(point, GB_List)

def calCenters(data):
    manager = GranularBallManager()
    return manager.calculate_centers(data)

def randCenters(data, lowDensity):
    manager = GranularBallManager()
    return manager.random_centers(data, lowDensity)

def OOD(dataDis_sort, k, center_label):
    manager = GranularBallManager()
    return manager.detect_outliers(dataDis_sort, k, center_label)

def generateGBs(GB_List, lowDensity, data, k):
    manager = GranularBallManager()
    return manager.generate_granular_balls(GB_List, lowDensity, data, k)

def generateGBList(data, k):
    manager = GranularBallManager()
    return manager.generate_granular_ball_list(data, k)