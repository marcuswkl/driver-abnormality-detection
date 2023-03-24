import os
import csv

# Vectorise all time series data to convert data files to data rows
# Combine data rows into one dataset based on class

def vectorise_data(directories):
    # Iterate through each class
    for class_number, directory in enumerate(directories):
        # Create vectorised dataset for the class
        with open(directory + '_vectorised.csv', 'w', newline='') as csv_output:
            # Initialise writer and write header row
            writer = csv.writer(csv_output, delimiter=',')
            writer.writerow(["time","speed","yaw","heading","location_x","location_y","gnss_latitude","gnss_longitude","accelerometer_x","accelerometer_y","accelerometer_z","gyroscope_x","gyroscope_y","gyroscope_z","height","throttle","steer","brake","reverse","hand_brake","manual_gear_shift","gear", "abnormality"])
            
            # Iterate through each class data file
            for filename in os.listdir(directory):
                input_filestring = os.path.join(directory, filename)
                # Check if the class data file is a valid file
                if os.path.isfile(input_filestring):
                    with open(input_filestring, newline='') as csv_input:
                        
                        # Create lists for each feature to store all 300 time series data values
                        time = []
                        speed = []
                        yaw = []
                        heading = []
                        location_x = []
                        location_y = []
                        gnss_latitude = []
                        gnss_longitude = []
                        accelerometer_x = []
                        accelerometer_y = []
                        accelerometer_z = []
                        gyroscope_x = []
                        gyroscope_y = []
                        gyroscope_z = []
                        height = []
                        throttle = []
                        steer = []
                        brake = []
                        reverse = []
                        hand_brake = []
                        manual_gear_shift = []
                        gear = []
                        
                        # Group all feature lists into a list
                        features_list = [time,speed,yaw,heading,location_x,location_y,gnss_latitude,gnss_longitude,accelerometer_x,accelerometer_y,accelerometer_z,gyroscope_x,gyroscope_y,gyroscope_z,height,throttle,steer,brake,reverse,hand_brake,manual_gear_shift,gear]
                        
                        # Initialise reader and skip header row of data file
                        reader = csv.reader(csv_input, delimiter=',')
                        next(reader, None)
                        
                        # Iterate through each row of the data file
                        for row in reader:
                            # Append each value in the row to their corresponding feature list based on index
                            for index, value in enumerate(row):
                                # Prevent index from going out of range
                                if index <= 21:
                                    features_list[index].append(value)
                    
                    # Append class number based on the current class
                    features_list.append(class_number)
                    # Write all feature lists as one row in the vectorised dataset
                    writer.writerow(features_list)