# Data Preprocessing is done here
# imports from this project
import data_generators

# external imports
import numpy as np
import pandas as pd
import dask.dataframe as dd
import os
import glob
import csv
import time
import re
from pathlib import Path # for os independent path handling
from snowmicropyn import Profile

# Set folder location of smp data (pnt format)
SMP_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/")
# Set file location of temperature data
T_LOC = Path("/home/julia/Documents/University/BA/Data/Arctic/MOSAiC_ICE_Temperature.csv")
# Set folder name were export files get saved
EXP_LOC = Path("smp_csv")

# exports pnt files (our smp profiles!) to csv files in a target directory
def pnt_to_csv (pnt_dir, target_dir, overwrite=False):
    """ Exports all pnt files from a dir and its subdirs as csv files into a new dir.
    Parameters:
        pnt_dir (Path): folder location of pnt files (in our case the smp profiles)
        target_dir (Path): folder name where converted csv files should get exported or were they have already been exported
        overwrite (Boolean): indicates if csv file should be overwriting if csv file already exists
    """
    # create dir for csv exports
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    # match all files in the dir who end on .pnt recursively
    match_pnt = pnt_dir.as_posix() + "/**/*.pnt"
    # use generator to reduce memory usage
    file_generator = glob.iglob(match_pnt, recursive=True)
    # yields each matching file and exports it
    for file in file_generator:
        file_name = Path(target_dir, file.split("/")[-2] + ".csv")
        # exports file only if we want to overwrite it or it doesnt exist yet
        if overwrite or not file_name.is_file():
            Profile.load(file).export_samples(file_name)

    print("Finished exporting all pnt file as csv files in {}.".format(target_dir))

# unites the csv data into one csv and adds smp index to data
# and saves everything in pd.DataFrame that can be reloaded from the npz data
def get_smp_data(csv_dir, csv_filename="smp_all.csv", npz_filename=None, skip_unify=False, skip_npz=False):
    """ Gets unlabelled smp data. Indexes the data. Writes the complete data into one csv.
    csv files are subsequently transformed to pandas frame. Return the pd.DataFrame and stores it as npz.

    Parameters:
        csv_dir (Path): folder location of smp profiles as csv files
        csv_filename (String): Name for file, where all the smp data will be or is written to. File extension must be csv!
        npz_filename (String): File where npz data will be or is saved. If none (default) same name as csv_filename but with npz extension.
        skip_unify (Boolean): Default False. Set to True if csv_filename has already unified data in it.
        skip_npz (Bollean): Default False. Set to True if npz_filename has already stored data in it.
    """
    # check if something has been exported in smp_csv
    if len(os.listdir(csv_dir)) == 0:
        print("Your target directory is empty. Consider using pnt_to_csv first.")

    if not skip_unify:
        # unify all csv files into one and index the data with their SMP tag
        unify_and_index(csv_dir, csv_filename)

    if not skip_npz:
        # save csv file as npz
        save_csv_as_npz(csv_filename, npz_filename)

    # convert the npz to a pandas DataFrame and return it
    return npz_to_pd(npz_filename)

def unify_and_index(csv_dir, csv_filename):
    """ Gets unlabelled smp data. Indexes the data. Writes the complete data into one csv.
    Parameters:
        csv_dir (Path): folder location of smp profiles as csv files
        csv_filename (String): Name for file, where all the smp data is written to. File extension must be csv!
    """
    # dictionary to resolve smp indexing
    smp_idx_resolver = {}

    # a list to save them all
    smp_all_rows = []
    # matching csv files recursively (recursively just to be safe)
    match_csv = csv_dir.as_posix() + "/**/*.csv"
    # use generator to reduce ram usage
    file_generator = glob.iglob(match_csv, recursive=True)

    # column names in csv files
    col_names = ["distance", "force", "smp_idx"]

    # we will write all data in one csv file. file is cleared automatically if already existant
    with open(csv_filename, "w+") as smp_all:
        # writer for writing rows in our file
        writer = csv.writer(smp_all)
        # yields each matching csv file
        for file in file_generator:
            # get smp datapoint name and convert it to int
            current_smp_idx = idx_to_int(file.split("/")[-1].split(".")[0])
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

    print("\nExported united csv files to {}.".format(csv_filename))

def save_csv_as_npz(csv_filename, npz_filename):
    """ Exports a smp csv file produced by get_smp_data as npz.
    Paramters:
        csv_filename (String): The csv file with all smp profiles. Must have columns "distance", "force" and "smp_idx"
        npz_filename (String): Name of output npz file
    """
    # read each column as pd.DataFrame
    print("Progress 0/3: Start reading csv cols as pd.DataFrames")
    distance = pd.read_csv(csv_filename, usecols=[0], dtype=np.float32, sep=",", header=None, engine="c", low_memory=True)
    print("Progress 1/3: Read distance")
    force = pd.read_csv(csv_filename, usecols=[1], dtype=np.float32, sep=",", header=None, engine="c", low_memory=True)
    print("Progress 2/3: Read force")
    smp_idx = pd.read_csv(csv_filename, usecols=[2], dtype=np.int32, sep=",", header=None, engine="c", low_memory=True)
    print("Progress 3/3: Read index")

    # save columns as npz
    if npz_filename is None:
        npz_filename = csv_filename.split(".")[0] + ".npz"

    # save npz
    np.savez(npz_filename, distance=distance.values[:, 0], force=force.values[:, 0], smp_idx=smp_idx.values[:, 0])
    print("\nExported pd.DataFrame data as numpy arrays to {}.".format(npz_filename))



def npz_to_pd(npz_file):
    """ Converts a npz file to a pandas DataFrame.
    Paramters:
        npz_file (np.npz): A numpy npz file
    Returns:
        pd.DataFrame: the converted pandas Dataframe

    """
    smp_npz = np.load("test02.npz")
    return pd.DataFrame.from_dict({item: smp_npz[item] for item in smp_npz.files})

def idx_to_int(string_idx):
    """ Converts a string that indexes the smp profile to an int.
    Paramters:
        string_idx (String): the index that is converted
    Returns:
        int32: the index as int.
        For smp profiles starting with S31H, S43M, S49M [1, 2, 3, 4] + the last four digits are the int.
        For smp profiles starting with PS122, [0] + 1 digit Leg + 2 digit week + 3 digit id are the int.
        All other profiles are 0.
    """
    if "PS122" in string_idx:
        str_parts = re.split("_|-", string_idx)
        #     Mosaic + Leg          + week                  + id number
        return int("1" + str_parts[1] + str_parts[2].zfill(2) + str_parts[3].zfill(3))

    elif "S31H" in string_idx:
        return int("2" + string_idx[-4:].zfill(6))
    elif "S43M" in string_idx:
        return int("3" + string_idx[-4:].zfill(6))
    elif "S49M" in string_idx:
        return int("4" + string_idx[-4:].zfill(6))
    else:
        return 0


# function to get temperature data
def get_temperature(temp):
    """ Gets temperature data from files
    Parameters:
        temp (Path): file location of temperature data
    Returns:
        dd.DataFrame: complete data in pd.DataFrame format
    """
    return pd.read_csv(temp)

# method to check if all pnt files have found their way into the united smp dataframe
# ATTENTION: this method takes extremely long! Some minutes for checking the existence of one file.
# It works perfectly fine and fast for small dataframes. Searching large dataframes just takes time.
def check_export(pnt_dir, smp_df, break_imm=True):
    """ Checks if all smp pnt files can be found in a dataframe. Takes a lot of time.
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
    match_pnt = pnt_dir.as_posix() + "/**/*.pnt"
    # use generator to reduce memory usage
    file_generator = glob.iglob(match_pnt, recursive=True)
    # yields each matching file and exports it
    for file in file_generator:
        print("Determining if file {} is in the dataframe".format(file))
        smp_was_found = any(smp_df.smp_idx == idx_to_int(file.split("/")[-1].split(".")[0]))
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

# TODO function to get all labelled data
    # labelled data has init file extension
    # convert to csv files
    # put it in pandas frame

def print_test_df(smp):
    """ Printing some features and information of smp DataFrame.
    Paramters:
        smp (pd.DataFrame): dataframe from which the information is retrieved
    """
    print("Overview of smp dataframe: \n", smp.head())
    print("Dataypes of columns: \n", smp.dtypes)
    print("Datapoints per SMP File: \n", smp["smp_idx"].value_counts())
    print("First row: \n", smp.iloc[0])
    print("Force at first row: ", smp["force"].iloc[0])
    print("Amount of datapoints with a force > 40: ", len(smp[smp["force"] > 40]))
    print("Was S31H0117 found in the dataframe? ", any(smp.smp_idx == idx_to_int("S31H0117")))
    print("Only S31H0117 data: \n", smp[smp["smp_idx"] == idx_to_int("S31H0117")].head())

def main():

    print("Starting to export and/or convert data")
    # get temp data
    tmp = get_temperature(temp=T_LOC)
    print(tmp.head())

    # export, unite and label smp data
    start = time.time()
    # export data from pnt to csv
    pnt_to_csv(pnt_dir=SMP_LOC, target_dir=EXP_LOC, overwrite=False)
    # unite data in one csv file, index it, convert it to pandas (and save it as npz)
    smp = get_smp_data(csv_dir=EXP_LOC, csv_filename="smp_all.csv", npz_filename="smp_all.npz", skip_unify=True, skip_npz=True)

    end = time.time()
    print("Elapsed time for export and dataframe creation: ", end-start)

    print("Number of files in export folder: ", len(os.listdir(EXP_LOC)))
    print("All pnt files from source dir were also found in the given dataframe: ", check_export(SMP_LOC, smp))

    print_test_df(smp)
    print("Finished export, transformation and printing example features of data.")

# Middleterm TODO: labelling and windowing data -> do this on a pandas frame
# Types of dataframes we will need: smp, temp, labelled and unlabelled, different window sizes
if __name__ == "__main__":
    main()
