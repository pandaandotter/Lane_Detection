import os
import json
import numpy as np

# Configuration
#TODO: make dynamic paths
input_dir = ""          # Change this to your input folder
output_dir = ""
image_center_x = 160                   # Assuming original images are 800px wide (adjust as needed)

os.makedirs(output_dir, exist_ok=True)

os.makedirs(output_dir, exist_ok=True)
def get_deepest_y(lane, h_samples):
    for x, y in reversed(list(zip(lane, h_samples))):  # start from bottom
        if x >= 0:
            return y
    return -1  # No valid points

def process_label_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    lanes = data["lanes"]
    h_samples = data["h_samples"]

    # Get max valid y for each lane
    lane_depths = []
    for lane in lanes:
        max_y = get_deepest_y(lane, h_samples)
        lane_depths.append((max_y, lane))

    # Take 2 lanes that go farthest down
    selected = sorted(lane_depths, key=lambda tup: -tup[0])[:2]
    selected_lanes = [lane for _, lane in selected]

    new_data = {
        "h_samples": h_samples,
        "lanes": selected_lanes
    }

    return new_data

# Process all files
for fname in os.listdir(input_dir):
    if fname.endswith(".json"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        filtered_data = process_label_file(input_path)

        with open(output_path, "w") as f_out:
            json.dump(filtered_data, f_out, indent=2)

print(f"Filtered labels saved to '{output_dir}'")