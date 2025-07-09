import pickle

# Replace with your actual file path
file_path = '/Users/artemekimov/Desktop/Development/qd_grasp_6dof/runs/run0/details_export.pkl'

# Load the contents
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Display the content
print(data)
