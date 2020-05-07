import os  # Operating system functions
import pandas as pd  # Datasets processing, CSV files handling
import numpy as np  # Linear algebra
from utils import load_data, save_data  # Project utilities

DATA_PATH = 'Data'  # Main data dir
STRIPES_DIR = os.path.join(DATA_PATH, 'stripes')  # Pre-augmented images
CSV_DIR = os.path.join(DATA_PATH, 'csv')  # CSV files dir
CSV_FILE = 'main.csv'
TRAIN_CSV_FILE = 'train.csv'
VAL_CSV_FILE = 'val.csv'
TEST_CSV_FILE = 'test.csv'


def group_data(main_df, group_by):
    """
    Group meta data of images according criterion in 'group_by'.

    Parameters
    ----------
        main_df : pandas.DataFrame
            Meta data of lesion image meta data.
        group_by : list[str]
            List of columns - grouping criterion.

    Returns
    -------
        pandas.DataFrame
            Meta data grouped according list of 'group_by' criterion.
    """
    return (main_df.groupby(group_by)  # Grouping
            .size()  # Counting
            .reset_index()  # Series to DataFrame
            .rename(columns={0: 'amount'}))  # For better readability


def split_sub_dataset(main_df, diagnosis, resolution, dataset_ratio):
    """
    Divide selected subset of main dataset into train, val and test.

    Select subset based on provided diagnosis and resolution and then
    split it into train, val and test according to ratios provided in
    dataset_ratio.

    Parameters
    ----------
        main_df : pandas.DataFrame
            Meta data of lesion image meta data.
        diagnosis : str
            Diagnosis of lesion.
        resolution : str
            Resolution of image: 'hi' (high) or 'lo' (low).
        dataset_ratio : dict
            Defines sizes of val and test datasets as float numbers < 1.
            where 1. is size of whole (train + val + test) dataset.
    """
    mask = ((main_df['diagnosis'] == diagnosis)
            & (main_df['resolution'] == resolution))
    records_num = len(main_df[mask])
    if records_num >= 3:  # If < 3 leave 'dataset' as is, i.e. 'train'.
        limit_val = int(np.ceil(records_num * dataset_ratio['val']))
        limit_test = int(np.ceil(records_num * dataset_ratio['test']))
        main_df.loc[mask
                    & (mask.cumsum() <= limit_val),
                    'dataset'] = 'val'
        main_df.loc[mask
                    & (mask.cumsum() > limit_val)
                    & (mask.cumsum() <= (limit_val + limit_test)),
                    'dataset'] = 'test'


def split_dataset(main_df):
    dataset_ratio = {'val': 0.15, 'test': 0.15}
    main_df['dataset'] = 'train'  # Setting default 'dataset' value.
    # Shuffle dataset and reindex
    main_df = main_df.sample(frac=1).reset_index(drop=True)
    group_by = ['lesion_type', 'diagnosis', 'resolution']
    df = group_data(main_df, group_by)
    for index, row in df[['diagnosis', 'resolution']].iterrows():
        split_sub_dataset(main_df,
                          row['diagnosis'],
                          row['resolution'],
                          dataset_ratio)
    return main_df


def angles_and_flips(lesion_type):
    """
    Generate lists of angles and flipping used during augmentation.

    Parameters
    ----------
        lesion_type : str
            Type of lesion: benign or malignant.

    Returns
    -------
        angles_str_lst : lst[str]
            List of angles (* 10) by which augmented images were rotated.
        flips_lst : lst[str]
            List of flipping done during images augmentation.
    """
    if lesion_type == 'benign':
        angles_str_lst = [str(int(10 * i * 45.)).zfill(4) for i in range(4)]
        flips_lst = ['f0']
    else:
        angles_str_lst = [str(int(10 * i * 22.5)).zfill(4) for i in range(7)]
        flips_lst = ['f' + str(i) for i in range(4)]
    return angles_str_lst, flips_lst


def expand_subset(df, lesion_type):
    """
    Assign pre-augmented images.

    Parameters
    ----------
        df : pandas.DataFrame
            Subset of train, val or test dataset which contains only lesions
            of given type (benign or malignant).
        lesion_type : str
            Type of lesion: benign or malignant.

    Returns
    -------
        df : pandas.DataFrame
            All relevant augmented images with lesion type.
    """
    angles_lst, flips_lst = angles_and_flips(lesion_type)
    aug_factor = len(angles_lst) * len(flips_lst)
    records_num = len(df)
    df = pd.concat([df] * aug_factor).reset_index(drop=True)
    angles_and_flips_lst = [[angle, flip]
                            for angle in angles_lst
                            for flip in flips_lst]
    for i, [angle, flip] in enumerate(angles_and_flips_lst):
        df.loc[df.isin(df.iloc[i * records_num:(i + 1) * records_num])
               .dsc_file, 'dsc_file'] = \
            df['dsc_file'] + df['lesion_type'].str[0] + angle + flip + \
            '.jpg'
    return df.rename(columns={'dsc_file': 'stripe_file'})


def extract_and_expand_subset(main_df, subset_type):
    """
    Extract subset from main_df and assign pre-augmented files.

    Parameters
    ----------
        main_df : pandas.DataFrame
            Meta data of lesion images.
        subset_type : str
            Type of dataset: train, val or test.

    Returns
    -------
        pandas.DataFrame
            File names with binary lesion type (0.0: benign, 1.0: malignant)
    """
    df = main_df[['dsc_file', 'lesion_type']][main_df['dataset']
                                              == subset_type]
    benign_df = df[df['lesion_type'] == 'benign']
    malignant_df = df[df['lesion_type'] == 'malignant']
    benign_aug_df = expand_subset(benign_df, 'benign')
    benign_aug_df['binary_lesion_type'] = 0
    benign_aug_df.drop('lesion_type', axis=1, inplace=True)
    malignant_aug_df = expand_subset(malignant_df, 'malignant')
    malignant_aug_df['binary_lesion_type'] = 1
    malignant_aug_df.drop('lesion_type', axis=1, inplace=True)
    return (pd.concat([benign_aug_df, malignant_aug_df])
            .sample(frac=1)
            .reset_index(drop=True)
            .rename(columns={'dsc_file': 'stripe_file'}))


def main():
    main_df = load_data(CSV_DIR, CSV_FILE)
    main_df = split_dataset(main_df)
    train_df = extract_and_expand_subset(main_df, 'train')
    save_data(train_df, CSV_DIR, TRAIN_CSV_FILE)
    val_df = extract_and_expand_subset(main_df, 'val')
    save_data(val_df, CSV_DIR, VAL_CSV_FILE)
    test_df = extract_and_expand_subset(main_df, 'test')
    save_data(test_df, CSV_DIR, TEST_CSV_FILE)


if __name__ == "__main__":
    main()
