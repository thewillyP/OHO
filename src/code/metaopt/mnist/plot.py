import os
import numpy as np

# Sample filename and data to save
filename = 'exp1/trial0'
script_dir = os.path.dirname(__file__)  # Get the directory of the current script
relative_path = os.path.join(script_dir, filename)  # Build the relative path

Wn_list = np.array([1, 2, 3, 4, 5])

# Save the array
np.save(relative_path + 'what', Wn_list)

# Load to confirm it saved correctly
loaded_Wn_list = np.load(relative_path + 'what.npy')
print("Saved and loaded data:", loaded_Wn_list)