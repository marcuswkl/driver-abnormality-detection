import os
import csv

# Combine vectorised datasets into one master dataset

def combine_data(dataset_filenames):
    # Create master dataset
    with open('dataset.csv', 'w', newline='') as csv_output:
        # Initialise writer and write header row
        writer = csv.writer(csv_output, delimiter=',')
        writer.writerow(["time","speed","yaw","heading","location_x","location_y","gnss_latitude","gnss_longitude","accelerometer_x","accelerometer_y","accelerometer_z","gyroscope_x","gyroscope_y","gyroscope_z","height","throttle","steer","brake","reverse","hand_brake","manual_gear_shift","gear", "abnormality"])
        
        # Iterate through each vectorised dataset
        for filename in dataset_filenames:
            input_filestring = filename
            # Check if the vectorised dataset is a valid file
            if os.path.isfile(input_filestring):
                # Initialise reader and skip header row of vectorised dataset
                with open(input_filestring, newline='') as csv_input:
                    reader = csv.reader(csv_input, delimiter=',')
                    next(reader, None)
                    # Write each row into master dataset
                    for row in reader:
                        writer.writerow(row)