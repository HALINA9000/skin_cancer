import os  # Operating system functions
import pandas as pd  # Datasets processing, CSV files handling


def load_data(csv_dir, csv_file):
    """
    Load meta data of images with lesions from CSV file.

    Parameters
    ----------
        csv_dir : str
            Full path of destination directory with CSV files.
        csv_file : str
            CSV file name with image meta data.

    Returns
    -------
        pandas.DataFrame
            Meta data of lesion images.
    """
    return pd.read_csv(os.path.join(csv_dir, csv_file))


def save_data(df, csv_dir, csv_file):
    """
    Save meta data of images with lesions to CSV file.

    Parameters
    ----------
        df : pandas.DataFrame
            Meta data of lesion images.
        csv_dir : str
            Full path of destination directory with CSV files.
        csv_file : str
            CSV file name with image meta data.
    """
    df.to_csv(os.path.join(csv_dir, csv_file), index=False)
