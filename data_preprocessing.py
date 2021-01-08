# Data Preprocessing is done here
import numpy as np
import pandas as pd
import openpyxl
import os
import glob
import csv
import timeit
from pathlib import Path # for os independent path handling
from snowmicropyn import Profile

# Set folder location of smp data
SMP_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/")
# Set file location of temperature data
T_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv")
# Set folder name were export files get saved
EXP_LOC = Path("smp_csv_test")

# TODO pnt and csv counter? to check if all data has been retrived?

# function to get all unlabelled data
def get_smp_data(smp, export=False):
    """ Gets unlabelled smp data.
    Unlabelled data with pnt file extension is fetched and exported to csv files if export=True.
    csv files are subsequently transformed to pandas frame.

    Parameters:
    smp (Path): folder location of smp profiles
    export (Boolean): indicates if data should be exported from pnt to csv. Default is False
    Returns:
    DataFrame: complete smp data as pd.DataFrame
    """
    # create dir for csv exports
    if not os.path.exists(EXP_LOC):
        os.mkdir(EXP_LOC)

    if export:
        # walks through file system and exports pnt files to csv files
        for root, dirs, files in os.walk(smp):
            for file in files:
                if file.endswith(".pnt") and export:
                    # export pnt file as csv
                    smp_raw = Profile.load(os.path.join(root, file))
                    # create file name for export
                    file_name = Path(EXP_LOC, Path(file).stem + ".csv")
                    smp_raw.export_samples(file_name)

    # check if something has been exported in smp_csv
    if len(os.listdir(EXP_LOC)) == 0:
        print("Your dir for exporting files is empty. Consider setting export=True in the get_smp_data in order to export the pnt files to csv first.")

    # timing how long the transformtion from csv to dataframe takes
    start = timeit.timeit()

    # a list to save them all
    smp_all_rows = []

    # transform csv files to dict rows
    for root, dirs, files in os.walk(EXP_LOC):
        for file in files:
            if file.endswith(".csv"):
                # get smp datapoint name
                current_smp_idx = file.split(".")[0]
                # open csv file and write each row to  smp_all_rows
                with open(os.path.join(root, file)) as csv_file:
                    # skip the first two lines of the file (comments)
                    next(csv_file)
                    next(csv_file)
                    # read the rows of the current smp file (read as dictionaries!)
                    current_smp_rows = csv.DictReader(csv_file, fieldnames=["distance", "force", "smp_idx"])
                    # each dictionary row is appended to the shared row list
                    for row in current_smp_rows:
                        # row gets extended by its current smp index
                        row["smp_idx"] = current_smp_idx
                        smp_all_rows.append(row)
    # convert list of dicts to dataframe
    smp_all_df = pd.DataFrame(smp_all_rows)

    # convert default object datatype to numeric
    smp_all_df[["distance", "force"]] = smp_all_df[["distance", "force"]].apply(pd.to_numeric)

    end = timeit.timeit()
    print("Elapsed Time for creating data frame from csv files: ", start-end)
    return smp_all_df


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
    print("Starting to export and convert data")
    tmp = get_temperature(temp=T_LOC)
    smp = get_smp_data(smp=SMP_LOC, export=False)
    print("Overview of smp dataframe: \n", smp)
    print("Datapoints per SMP File: \n", smp["smp_idx"].value_counts())
    print("First row: \n", smp.iloc[0])
    print("Force at first row: ", smp["force"].iloc[0])
    print("Amount of datapoints with a force > 40: ", len(smp[smp["force"] > 40]))
    print("Only S31H0117 data: \n", smp[smp["smp_idx"] == "S31H0117"])
    print("Dataypes of columns: \n", smp.dtypes)
    print("Finished export, transformation and printing example features of data.")

# use the three functions to create one pandas dataframe
if __name__ == "__main__":
    main()
