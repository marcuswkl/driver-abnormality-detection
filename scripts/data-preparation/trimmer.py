import os
import csv

# Trim all time series data to be exactly 5 minutes/300 seconds (t=0 to t=299).

def trim_data(directories):
    for directory in directories:
        # Create trimmed directories
        os.mkdir(directory + '_trimmed')
        
        for filename in os.listdir(directory):
            # Specify filestrings for I/O
            input_filestring = os.path.join(directory, filename)
            output_filestring = os.path.join(directory + '_trimmed', filename)
            
            # Checks if the input filestring is a valid file
            if os.path.isfile(input_filestring):
                # Initialise CSV file reader and writer
                with open(input_filestring, newline='') as csv_input, open(output_filestring, 'w', newline='') as csv_output:
                    reader = csv.reader(csv_input, delimiter=',')
                    writer = csv.writer(csv_output, delimiter=',')
                    
                    # Write data from reader only if reader data is header or time <= 299 seconds
                    for row in reader:
                        if row[0] == "time" or int(row[0]) <= 299:
                            writer.writerow(row)