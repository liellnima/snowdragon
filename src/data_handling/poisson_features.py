import numpy as np
import pandas as pd
from snowmicropyn import Profile, loewe2012, windowing

def rolling_window(df, window_size, rolling_cols, window_type="gaussian", window_type_std=1, poisson_cols=None, **kwargs):
    """ Applies one or several rolling windows to a dataframe. Concatenates the different results to a new dataframe.
    Parameters:
        df (pd.DataFrame): Original dataframe over whom we roll.
        window_size (list): List of window sizes that should be applied. e.g. [4]
        rolling_cols (list): list of columns over which should be rolled
        window_type (String): E.g. Gaussian (default). None is a normal window. Accepts any window types listed here:
            https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows
        window_type_std (int): std used for window type
        poisson_cols (list): List with names what should be retrieved from the poisson shot model. Default None (nothing included).
            List can include: "distance", "median_force", "lambda", "f0", "delta", "L"
        **kwargs: are catched in case more arguments are given on the commandline (see data_loader)
    Returns:
        pd.DataFrame: concatenated dataframes (original and new rolled ones)
    """
    all_dfs = [df]
    # for the poisson shot model calculations: df can only have cols force and distance
    poisson_df = pd.DataFrame(df[["distance", "mean_force"]])
    poisson_df.columns = ["distance", "force"]
    poisson_all_cols = ["distance", "median_force", "lambda", "f0", "delta", "L"]

    # roll over columns with different window sizes
    for window in window_size:
        # Roll window over 1mm summarizes statistics
        try:
            # window has the current distance point as center
            # for the first data points, only a few future datapoints will be used (min_periods=1 -> no na values)
            df_rolled = df[rolling_cols].rolling(window, win_type=window_type, center=True, min_periods=1).mean(std=window_type_std)
        except KeyError:
            print("The dataframe given does not have the columns indicated in rolling_cols.")
        # rename columns of rolled dataframe for distinction
        df_rolled.columns = [col + "_" + str(window) for col in rolling_cols]
        all_dfs.append(df_rolled)

        # Roll window for a poisson shot model to get lambda and delta
        if poisson_cols is not None:
            try:
                # calculate lambda and delta and media of poisson shot model
                overlap = (((window - 1) / window) * 100) + 0.0001 # add epsilon to round up 0.0001
                poisson_rolled = calc(poisson_df, window=window, overlap=overlap) #essentially the loewe2012.calc function
                poisson_rolled.columns = poisson_all_cols
                poisson_rolled = poisson_rolled[poisson_cols]
            except KeyError:
                print("You can only use a (sub)list of the following features for poisson_cols: distance, median_force, lambda, f0, delta, L")
            # add the poisson data to the all_dfs list and rename columns for distinction
            poisson_rolled.columns = [col + "_" + str(window) for col in poisson_cols]
            all_dfs.append(poisson_rolled)

    return pd.concat(all_dfs, axis=1)

# author Henning Loewe (2012) -> only difference: I am preventing zero divisions
def calc(samples, window, overlap):
    """Calculation of shot noise model parameters.
    :param samples: A pandas dataframe with columns called 'distance' and 'force'.
    :param window: Size of moving window.
    :param overlap: Overlap factor in percent.
    :return: Pandas dataframe with the columns 'distance', 'force_median',
             'L2012_lambda', 'L2012_f0', 'L2012_delta', 'L2012_L'.
    """
    # Calculate spatial resolution of the distance samples as median of all step sizes.
    spatial_res = np.median(np.diff(samples.distance.values))

    # Split dataframe into chunks
    chunks = windowing.chunkup(samples, window, overlap)
    result = []
    for center, chunk in chunks:
        f_median = np.median(chunk.force)
        # check if all elements are zero -> if yes, replace results with 0
        if all(item == 0 for item in chunk.force):
            sn = (0, 0, 0, 0) # a tuple containing four zeros (lamda, f0, delta, L)
        else:
            sn = loewe2012.calc_step(spatial_res, chunk.force)
        result.append((center, f_median) + sn)

    return pd.DataFrame(result, columns=['distance', 'force_median', 'L2012_lambda', 'L2012_f0',
                                         'L2012_delta', 'L2012_L'])
