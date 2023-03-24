# DATA PREPARATION

# Specify directories containing collected data

directories = ['normal', 'abnormal']

# Trim all time series data to be exactly 5 minutes/300 seconds (t=0 to t=299).
import trimmer
trimmer.trim_data(directories)

# Specify directories containing trimmed data

trimmed_directories = ['normal_trimmed', 'abnormal_trimmed']

# Vectorise all time series data to convert data files to data rows
# Combine data rows into one dataset based on class
import vectoriser
vectoriser.vectorise_data(trimmed_directories)

# Specify vectorised dataset filenames
dataset_filenames = ['normal_trimmed_vectorised.csv', 'abnormal_trimmed_vectorised.csv']

# Combine vectorised datasets into one master dataset
import combiner
combiner.combine_data(dataset_filenames)