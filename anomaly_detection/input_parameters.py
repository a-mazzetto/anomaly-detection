"""Anomaly detection parametrization"""
import os

# Folders
FILE_PATH = os.path.join(os.getcwd(), "data//auth.txt")
FILE_NAME = ".".join(os.path.basename(FILE_PATH).split(".")[:-1])
RESULTS_FOLDER = os.path.join(".//data", FILE_NAME)
if not os.path.exists(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)

# Number of nodes and type of process conditional on destination
N_NODES = 1000
DDCRP = False

# Preprocessing parametrization
PREPROCESSING_TSTART = 0
PREPROCESSING_TEND = 1e8
PREPROCESSING_THERSHOLD = 30
Y_PARAMETERS_FILENAME = "y_params.txt"
X_GIVEN_Y_PARAMETERS_FILENAME = "x_given_y_params.txt"

# Destination PY
DESTINATION_PVALUES_FILEPATH = os.path.join(RESULTS_FOLDER, "destination_py.txt")

# Source PY
SOURCE_GIVEN_DEST_PVALUES_FILEPATH_PY = os.path.join(RESULTS_FOLDER, "source_given_destination_py.txt")
