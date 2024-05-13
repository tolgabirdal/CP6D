import numpy as np
from IPython import embed
# Example data
data = {'uncertainties': np.random.rand(1000)}  # Random uncertainties between 0 and 1

def get_mask_for_top_uncertainties(uncertainties, top_percentage):
    # Calculate the number of elements to include
    total_samples = len(uncertainties)
    top_count = int(total_samples * (top_percentage / 100))
    
    # Find the threshold value for the top 'top_percentage'%
    threshold = np.sort(uncertainties)[top_count - 1]
    embed()
    # Generate the mask
    mask = uncertainties <= threshold

    return mask

# Example usage:
top_percentage = 5  # Get the mask for the top 5% lowest uncertainties
mask = get_mask_for_top_uncertainties(data['uncertainties'], top_percentage)
print(f"Mask for the top {top_percentage}%: {mask}")
