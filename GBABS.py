import numpy as np
import RD_GBG


class GBABS:
    """Granular Ball-based Approximate Borderline Sampling
    
    This class implements a method for sampling boundary points between different classes
    by identifying and analyzing granular balls at class boundaries.
    """
    def __init__(self, data, rho):
        self.data = data
        self.rho = rho  ## density threshold for GBs
        #self.bnd_gbs = [] # borderline GBs
        self.boundary_samples = []
        self.boundary_sample_indices = []

    def bound_sampling(self):
        """Sample boundary points between different classes
        
        This method identifies boundary points by:
        1. Generating granular balls
        2. Finding balls of different classes that are adjacent along feature dimensions
        3. Extracting samples from these boundary balls
        
        Returns:
            List of boundary samples (with labels, without indices)
        """
        # Generate granular balls and their centers
        granular_balls, center_positions = RD_GBG.generateGBList(self.data,self.rho)
        center_positions = np.array(center_positions)
        # Get number of features (excluding label and index)
        num_features = len(self.data[1]) - 2
        # Examine each feature dimension for boundary balls
        for feature_dim in range(num_features):
            # Sort ball centers by their feature value (ascending)
            sorted_indices = np.argsort(center_positions[:, feature_dim])

            # Compare adjacent balls in the sorted list
            for idx in range(len(sorted_indices) - 1):
                current_ball_idx = sorted_indices[idx]
                next_ball_idx = sorted_indices[idx + 1]

                # Check if adjacent balls belong to different classes
                if granular_balls[current_ball_idx].label != granular_balls[next_ball_idx].label:
                    '''if GB_list[j] not in self.bnd_gbs:
                        self.bnd_gbs.append(GB_list[j])
                    if GB_list[j+1] not in self.bnd_gbs:
                        self.bnd_gbs.append(GB_list[j+1])'''
                    # Sample boundary points from both balls
                    
                    # From first ball: samples with maximum value in the current feature dimension
                    current_ball_samples = self.extract_boundary_samples(
                        granular_balls[current_ball_idx], 
                        feature_dim, 
                        sample_max=True
                    )
                    
                    # From second ball: samples with minimum value in the current feature dimension
                    next_ball_samples = self.extract_boundary_samples(
                        granular_balls[next_ball_idx], 
                        feature_dim, 
                        sample_max=False
                    )
                    
                    # Add unique samples to the results
                    self._add_unique_samples(current_ball_samples)
                    self._add_unique_samples(next_ball_samples)
        
        return self.boundary_samples

    def extract_boundary_samples(self, ball, feature_dim, sample_max=True):
        """Extract boundary samples from a granular ball
        
        For balls at class boundaries, this method selects samples that are closest
        to the boundary in the specified feature dimension.
        
        Args:
            ball: The granular ball to sample from
            feature_dim: The feature dimension to consider
            sample_max: If True, select samples with maximum value, otherwise minimum
        
        Returns:
            Array of selected samples
        """
        # If the ball contains only one sample, return it
        if len(ball.data) == 1:
            return ball.data
        
        # Feature dimension in the data array is offset by 1 (label is at position 0)
        feature_col = feature_dim + 1
        
        # Select samples with maximum/minimum value in the specified dimension
        if sample_max:
            feature_value = np.max(ball.data[:, feature_col])
        else:
            feature_value = np.min(ball.data[:, feature_col])
        
        # Return all samples that match the selected feature value
        return ball.data[ball.data[:, feature_col] == feature_value]
    
    def _add_unique_samples(self, samples):
        """Add samples to the boundary sample list, avoiding duplicates
        
        Args:
            samples: Array of samples to add
        """
        for sample in samples:
            # Extract the sample index (last column)
            sample_index = sample[-1]
            
            # Only add if not already included
            if sample_index not in self.boundary_sample_indices:
                self.boundary_sample_indices.append(sample_index)
                # Store the sample with label but without index
                self.boundary_samples.append(sample[:-1])  # Keep label and features, remove only index
