import numpy as np  

def compute_unsafe_regions(obstacles, horizon=2, obs_dim=6):
    """
        Compute unsafe regions for the given obstacles.
        Returns a dictionary with the unsafe regions for each time step.
        Format: {time_step: [unsafe_region_1, unsafe_region_2, ...]}, where unsafe_region_i is a obs_dim x 2 array with the bounds of the unsafe region.
    """
    unsafe_regions = {}
    for i in range(horizon):
        unsafe_regions[i] = []
        for obs in obstacles[i]:
            set_bounds = np.zeros((obs_dim, 2))
            set_bounds[0] = [obs['x'] - obs['d'] / 2, obs['x'] + obs['d'] / 2]
            set_bounds[2] = [obs['y'] - obs['d'] / 2, obs['y'] + obs['d'] / 2]
            unsafe_regions[i].append(set_bounds)

    return unsafe_regions