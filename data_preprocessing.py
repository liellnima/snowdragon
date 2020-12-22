# Data Preprocessing is done here
import numpy as np
import pandas as pd
import openpyxl
import os
import glob
from pathlib import Path # for os independent path handling
from snowmicropyn import Profile

# Set folder location of smp data
SMP_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/")
# Set file location of temperature data
T_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv")

# function to get all unlabelled data
def get_smp_data(smp, export=False):
    """ Gets unlabelled smp data
    Parameters:
    smp (Path): folder location of smp profiles
    Returns:
    DataFrame: complete smp data as pd.DataFrame
    """
    # create dir for csv exports
    if not os.path.exists("smp_csv"):
        os.mkdir("smp_csv")

    for root, dirs, files in os.walk(smp):
        for file in files:
            if file.endswith(".pnt"):
                # export pnt file as csv
                smp_raw = Profile.load(os.path.join(root, file))
                # create file name for export
                file_name = Path("smp_csv", Path(file).stem + ".csv")
                print(file_name)
                smp_raw.export_samples(file_name)

                # transform to pd DataFrame
                #smp_df = pd.read_csv(smp_csv, index_col=0)


    # unlabelled data has pnt file extension
    # convert to csv files
    # put it in pandas frame

# function to get all labelled data
    # labelled data has init file extension
    # convert to csv files
    # put it in pandas frame

# function to get temperature data
def get_temperature(temp):
    """ Gets temperature data from files
    Parameters:
    temp (Path): file location of temperature data
    Returns:
    DataFrame: complete data in pd.DataFrame format
    """
    temp_file = pd.read_csv(temp, index_col=0)
    return temp_file


def main():
    print("Hello, I am the start")
    tmp = get_temperature(temp=T_LOC)
    smp = get_smp_data(smp=SMP_LOC, export=True)
    print("Bye, I am the end")

# use the three functions to create one pandas dataframe
if __name__ == "__main__":
    main()
