# Data Preprocessing is done here
import data_generators

import numpy as np
import pandas as pd
import openpyxl
import os
import glob
import csv
import time
from pathlib import Path # for os independent path handling
from snowmicropyn import Profile

# Set folder location of smp data (pnt format)
SMP_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/")
# Set file location of temperature data
T_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv")
# Set folder name were export files get saved
EXP_LOC = Path("smp_csv")

# function to get all unlabelled data (meaning its converted, exported and put in one csv file)
def get_smp_data(smp, target_dir, filename="smp_all.csv", export=False, overwrite=False):
    """ Gets unlabelled smp data. Writes the complete data into one csv.
    Unlabelled data with pnt file extension is fetched and exported to csv files if export=True.
    csv files are subsequently transformed to pandas frame.

    Parameters:
    smp (Path): folder location of smp profiles
    target_dir (Path): folder name where converted csv files should get exported or were they have already been exported
    filename (String): Name for file, where all the smp data is written to. File extension must be csv!
    export (Boolean): indicates if data should be exported from pnt to csv. Default is False
    overwrite (Boolean): important during export; indicates if csv file should be overwriting if csv file already exists

    """
    # create dir for csv exports
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    if export:
        # match all files in the dir who end on .pnt recursively
        match_pnt = smp.as_posix() + "/**/*.pnt"
        # use generator to reduce memory usage
        file_generator = glob.iglob(match_pnt, recursive=True)
        # yields each matching file and exports it
        for file in file_generator:
            file_name = Path(target_dir, file.split("/")[-2] + ".csv")
            # exports file only if we want to overwrite it or it doesnt exist yet
            if overwrite or not file_name.is_file():
                Profile.load(file).export_samples(file_name)

    print("finished exporting")

    # check if something has been exported in smp_csv
    if len(os.listdir(target_dir)) == 0:
        print("Your export directory is empty. Consider setting export=True in the get_smp_data in order to export the pnt files to csv first.")

    # a list to save them all
    smp_all_rows = []
    # matching csv files recursively (recursively just to be safe)
    match_csv = target_dir.as_posix() + "/**/*.csv"
    # use generator to reduce ram usage
    file_generator = glob.iglob(match_csv, recursive=True)

    # column names in csv files
    col_names = ["distance", "force", "smp_idx"]

    # we will write all data in one csv file. file is cleared automatically if already existant
    with open(smp_all_filename, "w+") as smp_all:
        # writer for writing rows in our file
        writer = csv.writer(smp_all)
        # yields each matching csv file
        print("starting with csv generator")
        for file in file_generator:
            # get smp datapoint name
            current_smp_idx = file.split("/")[-1].split(".")[0]
            # open csv file and write each row to  smp_all_rows
            with open(Path(file)) as csv_file:
                # skip the first two lines of the file (comments)
                next(csv_file)
                next(csv_file)
                # read the rows of the current smp file (read as dictionaries!)
                current_smp_rows = csv.DictReader(csv_file, fieldnames=col_names)
                # each dictionary row is appended to the shared row list
                for row in current_smp_rows:
                     # row gets extended by its current smp index name
                     row["smp_idx"] = current_smp_idx
                     # write the row into the csv file
                     writer.writerow([row["distance"], row["force"], row["smp_idx"]])

    print("finished producing united csv file")

# TODO function to get all labelled data
    # labelled data has init file extension
    # convert to csv files
    # put it in pandas frame (or csv_all)

# function to get temperature data
def get_temperature(temp):
    """ Gets temperature data from files
    Parameters:
    temp (Path): file location of temperature data
    Returns:
    pd.DataFrame: complete data in pd.DataFrame format
    """
    temp_file = pd.read_csv(temp, index_col=0)
    return temp_file

# TODO update check_export, such that it works with a datagenerator instead!!!
# method to check if all pnt files have found their way into the united smp dataframe
def check_export(pnt_dir, smp_df, break_imm=True):
    """ Checks if all smp pnt files can be found in a dataframe
    Parameters:
    pnt_dir (Path): folder location where all the unlabelled pnt data is stored
    smp_df (pd.DataFrame): dataframe with column "smp_idx", where the complete smp data is collected
    break_imm (Boolean): indicates if search should be aborted immediately when a file was not found.
        This is faster and prints out which file has not been found. Default value is True.
    Returns:
    Boolean: True if all files can be found in the dataframe, False otherwise
    """
    # check if dir exists
    if not os.path.exists(pnt_dir):
        print("Targeted directory for pnt files does not exist.")
    # check if dir is empty
    if len(os.listdir(pnt_dir)) == 0:
        print("Warning: Targeted directory for pnt files is empty.")

    # stores whether file was found
    found_all = []

    # match all files in the dir who end on .pnt recursively
    match_pnt = smp.as_posix() + "/**/*.pnt"
    # use generator to reduce memory usage
    file_generator = glob.iglob(match_pnt, recursive=True)
    # yields each matching file and exports it
    for file in file_generator:
        smp_was_found = any(smp_df.smp_idx == file.split(".")[0]) # DATAGENERATOR
        found_all.append(smp_was_found)
        if break_imm and not smp_was_found:
            print("The following file was not found: ", file.split(".")[0])
            return False

    # if all values in found_all are True, return True
    if all(smp_found == True for smp_found in found_all):
        return True
    # in all other cases print how many where found and how many are missing
    print("Number of files found in the dataframe: ", found_all.count(True))
    print("Number of files NOT found in the dataframe: ", found_all.count(False))
    # and return False
    return False

def print_test_df(smp):
    """ Printing some features and information of smp DataFrame
    Paramters:
    smp (pd.DataFrame): dataframe from which the information is retrieved
    """
    print("Overview of smp dataframe: \n", smp)
    print("Datapoints per SMP File: \n", smp["smp_idx"].value_counts())
    print("First row: \n", smp.iloc[0])
    print("Force at first row: ", smp["force"].iloc[0])
    print("Amount of datapoints with a force > 40: ", len(smp[smp["force"] > 40]))
    print("Only S31H0117 data: \n", smp[smp["smp_idx"] == "S31H0117"])
    print("Dataypes of columns: \n", smp.dtypes)

def main():
    print("Starting to export and/or convert data")
    tmp = get_temperature(temp=T_LOC)
    start = time.time()
    # datagenerator still missing here
    smp = get_smp_data(smp=SMP_LOC, target_dir=EXP_LOC, export=False, overwrite=False)
    end = time.time()
    print("Elapsed time for export and dataframe creation: ", end-start)

    print("Number of files in export folder: ", len(os.listdir(EXP_LOC)))
    #print("All pnt files from source dir were also found in the given dataframe: ", check_export(SMP_LOC, smp))

    print_test_df(smp)
    print("Finished export, transformation and printing example features of data.")


# Middleterm TODO: make a new file to create different data generators  and find out how you can pipeline them
# Types of datagenerators: smp, temp, labelled and unlabelled, different window sizes
if __name__ == "__main__":
    main()
