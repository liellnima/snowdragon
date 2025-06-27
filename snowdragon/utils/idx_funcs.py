import re
from snowdragon.utils.helper_funcs import load_configs

# load smp devices configs:
SMP_DEVICES = load_configs(
            config_subdir="smp_devices",
            config_name="smp_devices.yaml",
        )
SMP_DEVICE_VALUES = list(SMP_DEVICES["smp_devices"].values())

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
    # MOSAIC convention
    if "PS122" in string_idx:
        str_parts = re.split("_|-", string_idx)
        #     Mosaic + Leg          + week                  + id number
        return int("1" + str_parts[1] + str_parts[2].zfill(2) + str_parts[3].zfill(3))
    # SMP naming convention
    elif string_idx.startswith("S"):
        # check if the SMP device was added
        try:
            SMP_DEVICES["smp_devices"][string_idx[:4]]
        except KeyError:
            raise ValueError("Device {string_idx} was not found. It looks like you are using the traditional SMP file naming convention. If yes, please add your SMP device (start of filename) to the configs/smp_devices/smp_devices.yaml.")
        # try to convert the idx to int and return it
        try:
            return int(str(SMP_DEVICES["smp_devices"][string_idx[:4]]) + string_idx[-4:].zfill(6))
        except Exception:
            raise RuntimeError("Something is probably wrong with the code - this is not your fault (except if you are writing the code).")
    # ADD YOUR naming convention here with another elif line
    # Please make sure to add it to the configs/smp_devices/smp_devices.yaml
    else:
        raise ValueError("SMP naming convention is unknown. Please add another elif line in idx_to_int to handle your SMP naming convention.")


def int_to_idx(int_idx: int) -> str:
    """ Converts an int that indexes the smp profile to a string.
    Paramters:
        int_idx (int): the index that is converted
    Returns:
        str: the index as string
        For smp profiles starting with 200, 300 or 400, the last four digits are caught
            and added either to       S31H, S43M, S49M.
        For smp profiles starting with 1 throw an error, since no SMP profile should have a 1.
            PS122 is only used to describe event ids.
        Profiles with 0 throw an error.
        All other profiles get their int index returned as string (unchanged).
    """
    int_idx = str(int_idx)
    smp_device = int(int_idx[0])

    if (smp_device == 1) or (smp_device == 0):
        raise ValueError("SMP indices with 0 or 1 cannot be converted. Indices with 1 are reserved for event IDs. 0 means that no suitable match was found during index convertion.")
    # ADD YOUR naming convention here and decide how it should be handled
    # (you need to catch your specific numbers here, all other numbers will be treated as SMP naming convention or the default case)
    elif smp_device in SMP_DEVICE_VALUES:
        name_start_str = [key for key, val in SMP_DEVICES["smp_devices"].items() if val == smp_device]
        return name_start_str[0] + int_idx[3:7]
    else:
        return str(int_idx)