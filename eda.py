# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:35:23 2019

@author: Tom S. tom dot s at halina9000 dot com

Exploratory Data Analysis of dermoscopic images from ISIC Archive.
"""
import os  # Operating system functions
import json  # JSON files handling
from tqdm import tqdm  # Progress bar
import pandas as pd  # Datasets processing, CSV files handling
import seaborn as sns  # Charts
import matplotlib.pyplot as plt  # Charts
from utils import save_data  # Project utilities

DATA_PATH = 'Data'  # Main data dir
DSC_DIR = os.path.join(DATA_PATH, 'Descriptions')  # Description files dir
IMG_DIR = os.path.join(DATA_PATH, 'Images')  # Original images dir
CSV_DIR = os.path.join(DATA_PATH, 'csv')  # CSV files dir
CSV_FILE = 'main.csv'  # CSV file with image meta data
CHARTS_DIR = os.path.join(DATA_PATH, 'charts')  # PDA charts dir

if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)
if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)


def size_and_type(dsc_file, dsc_dir):
    """
    Read from description file size of image and type of lesion.

    Parameters
    ----------
        dsc_file : str
            Name of json description file.
        dsc_dir : str
            Full path of directory with description files.

    Returns
    -------
        src_size : tuple(int, int)
            Size of source image as a tuple(width, height).
        lesion_type : str
            Type of lesion or empty string if type not present.
        diagnosis : str
            Detailed diagnosis or empty string if diagnosis not present.
        diagnosis_confirm_type : str
            Type of diagnosis confirmation.
    """
    dsc_file_object = open(os.path.join(dsc_dir, dsc_file), 'r')
    dsc = json.load(dsc_file_object)
    src_size = (dsc['meta']['acquisition']['pixelsX'],
                dsc['meta']['acquisition']['pixelsY'])
    try:
        lesion_type = dsc['meta']['clinical']['benign_malignant']
    except KeyError:
        lesion_type = 'unknown'
    try:
        diagnosis = dsc['meta']['clinical']['diagnosis']
    except KeyError:
        diagnosis = 'unknown'
    try:
        diagnosis_confirm_type = \
            dsc['meta']['clinical']['diagnosis_confirm_type']
    except KeyError:
        diagnosis_confirm_type = 'unknown'
    dsc_file_object.close()
    if lesion_type is None:
        lesion_type = 'unknown'
    if diagnosis is None:
        diagnosis = 'unknown'
    if diagnosis_confirm_type is None:
        diagnosis_confirm_type = 'unknown'
    return src_size, lesion_type, diagnosis, diagnosis_confirm_type


def img_file_extension(src_img_dir, img_file_name):
    """
    Check if path for jpeg or png file exists.

    Parameters
    ----------
        src_img_dir : str
            Full path of source directory with image files.
        img_file_name : str
            Original name of image file with no file extension.

    Returns
    -------
        str or None
            Extension of image file or None if no file in known format.
    """
    path_jpeg = os.path.join(src_img_dir, img_file_name + '.jpeg')
    path_jpg = os.path.join(src_img_dir, img_file_name + '.jpg')
    path_png = os.path.join(src_img_dir, img_file_name + '.png')
    if os.path.exists(path_jpeg):
        return '.jpeg'
    elif os.path.exists(path_jpg):
        return '.jpg'
    elif os.path.exists(path_png):
        return '.png'
    else:
        return None


def make_main_df(dsc_dir):
    """
    Make DataFrame containing meta data of dermatoscopic images.

    Parameters
    ----------
        dsc_dir : str
            Full path of directory with description files.

    Returns
    -------
        pandas.DataFrame
            Meta data of dermatoscopic images.
    """
    dsc_files = os.listdir(dsc_dir)
    dsc_files_num = len(dsc_files)
    img_meta_data_lst = []
    for i in tqdm(range(dsc_files_num)):
        dsc_file = dsc_files[i]
        img_file_ext = img_file_extension(IMG_DIR, dsc_file)
        if img_file_ext:
            src_size, lesion_type, diagnosis, diagnosis_confirm_type = \
                size_and_type(dsc_file, dsc_dir)
            img_meta_data_lst.append([dsc_file,
                                      img_file_ext,
                                      src_size[0],
                                      src_size[1],
                                      lesion_type,
                                      diagnosis,
                                      diagnosis_confirm_type])

    cols = ['dsc_file',
            'img_ext',
            'width',
            'height',
            'lesion_type',
            'diagnosis',
            'diagnosis_confirm_type']
    return pd.DataFrame(img_meta_data_lst, columns=cols)


def bar_chart(img_file, charts_dir, size, axes, df, title, palette,
              hue=None, order=None, dodge=False, scale='log'):
    """
    Plot bar chart with amount of cases in logarithmic scale.

    Parameters
    ----------
        img_file : str
            Name of file (with file extension) where chart should be saved.
        charts_dir: str
            Directory where chart should be saved.
        size : tuple(int, int)
            Size of chart.
        axes: tuple(str, str)
            Name of x and y axis (column names in DataFrame df).
        df : pandas.DataFrame
            Chart data.
        title : tuple(str, str)
            Chart title and subtitle.
        palette : list[str] or None
            Custom palette.
        hue : str, optional
            Higher level grouping.
            Default: None
        order : list[str], optional
            Custom y-ticks order.
            Default: None
        dodge : bool
            When 'hue' nesting is used, whether elements should be shifted
            along the categorical axis.
        scale : str
            X-axis scale: 'log' or 'linear'.
            Default: 'log'
    """
    plt.ioff()
    base_color = '#555555'
    sns.set(rc={'figure.figsize': size,
                'figure.constrained_layout.use': True,
                'axes.titlesize': 'x-small',
                'axes.titleweight': 'bold',
                'axes.labelsize': 'xx-small',
                'text.color': base_color,
                'xtick.labelsize': 'x-small',
                'ytick.labelsize': 'xx-small',
                'font.family': 'monospace',
                'legend.facecolor': 'white'})
    ax = sns.barplot(x=axes[0],
                     y=axes[1],
                     hue=hue,
                     data=df,
                     order=order,
                     dodge=dodge,
                     palette=palette)
    ax.set_title('\n' + ' '.join(title[0]) + '\n' + title[1] + '\n',
                 color=base_color)
    ax.set_xscale(scale)
    ax.set_xlabel('amount of cases')
    ax.set_ylabel('')
    if scale == 'log':
        ax.axes.set_xlim(0.5, 1e5)
        horizontalalignment = 'right'
        color = 'white'
    else:
        ax.axes.set_xlim(0, 14000)
        horizontalalignment = 'left'
        color = base_color
    if hue:
        ax.legend(title=None,
                  fontsize='x-small',
                  loc='upper right',
                  frameon=True)
    for patch in ax.patches:
        x = patch.get_width()
        if not pd.isna(x):
            label = '  ' + str(int(x)) + '  '
            ax.text(x,
                    patch.get_y() + patch.get_height() / 2,
                    label,
                    horizontalalignment=horizontalalignment,
                    verticalalignment='center',
                    fontsize='xx-small',
                    weight='bold',
                    color=color)
    ax.get_figure().savefig(os.path.join(charts_dir, img_file))
    plt.close('all')


def group_data(main_df, group_by):
    """
    Load meta data of images with lesions.

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


def update_lesion_type(main_df, lesion_type_wrong, lesion_type_correct,
                       diagnosis):
    """
    Correct lesion type.

    Parameters
    ----------
        main_df : pandas.DataFrame
            Meta data of lesion image meta data.
        lesion_type_wrong : str
            Incorrect type of lesion.
        lesion_type_correct : str
            Correct type of lesion.
        diagnosis : str
            Diagnosis of lesion.
    """
    main_df.loc[(main_df['lesion_type'] == lesion_type_wrong)
                & (main_df['diagnosis'] == diagnosis),
                'lesion_type'] \
        = lesion_type_correct


def data_cleaning(main_df):
    """
    Perform data cleaning.

    Correction of values of lesion_type, if possible. Then removing rows
    with unknown lesion_type. Finally split cases with unknown diagnosis
    into 'unknown benign' and 'unknown malignant'.

    Parameters
    ----------
        main_df : pandas.DataFrame
            Meta data of lesion image meta data.
    """
    updates_lst = [
        ['malignant', 'benign', 'seborrheic keratosis'],
        ['benign', 'malignant', 'basal cell carcinoma'],
        ['unknown', 'benign', 'actinic keratosis'],
        ['unknown', 'malignant', 'basal cell carcinoma'],
        ['unknown', 'benign', 'dermatofibroma'],
        ['unknown', 'benign', 'pigmented benign keratosis'],
        ['unknown', 'malignant', 'squamous cell carcinoma'],
        ['unknown', 'benign', 'vascular lesion'],
        ['indeterminate', 'benign', 'nevus'],
        ['indeterminate/benign', 'benign', 'nevus'],
        ['indeterminate', 'benign', 'atypical melanocytic proliferation'],
        ['indeterminate/malignant', 'benign', 'atypical melanocytic '
                                              'proliferation']
    ]
    for lesion_type_wrong, lesion_type_correct, diagnosis in updates_lst:
        update_lesion_type(main_df,
                           lesion_type_wrong,
                           lesion_type_correct,
                           diagnosis)
    # Remove rest of unknown lesion types
    main_df.drop(main_df[main_df['lesion_type'] == 'unknown'].index,
                 axis=0,
                 inplace=True)
    # Distinguish benign unknown and malignant unknown diagnosis
    main_df.loc[main_df['diagnosis'] == 'unknown', 'diagnosis'] = \
        main_df['diagnosis'] + ' ' + main_df['lesion_type']


def dim_into_bins(main_df, dim, bin_size=500):
    """
    Aggregate given dimension of images into bins with defined range.

    Parameters
    ----------
        main_df : pandas.DataFrame
            Meta data of lesion image meta data.
        dim : str
            Column name in DataFrame storing values of given dimension.
        bin_size : int
            Size of each bin.
            Default: 500.

    Returns
    -------
        pandas.DataFrame
            Aggregated results - amount of images in dimension ranges.
    """
    dim_range = main_df[dim].agg([min, max])
    bins_min = bin_size * (dim_range['min'] // bin_size)
    bins_max = bin_size * (dim_range['max'] // bin_size + 2)
    bins = [i for i in range(bins_min, bins_max, bin_size)]
    return (pd.cut(main_df[dim], bins)
            .reset_index()
            .groupby(dim)
            .size()
            .reset_index()
            .rename(columns={dim: 'range', 0: 'amount'})
            .astype({'range': str}))


def main():
    main_df = make_main_df(DSC_DIR)

    # 1. Lesion types before cleaning.
    # 1.1 Query
    group_by = ['lesion_type']
    df = group_data(main_df, group_by)
    # 1.2 Chart
    title = ('TYPES OF LESIONS', 'before data cleaning')
    order = ['benign', 'indeterminate/benign', 'indeterminate', 'unknown',
             'indeterminate/malignant', 'malignant']
    palette = ['#024fc2', '#3e5980', '#666666', '#666666', '#823f3e',
               '#c70300']
    axes = ('amount', 'lesion_type')
    size = (6, 2.5)
    img_file = 'lesion_types_before_cleaning.svg'
    bar_chart(img_file, CHARTS_DIR, size, axes, df, title, palette,
              order=order)

    # 3. Data cleaning
    data_cleaning(main_df)

    # 2. Lesion types after cleaning.
    # 2.1 Query
    group_by = ['lesion_type']
    df = group_data(main_df, group_by)
    # 2.2 Chart
    title = ('TYPES OF LESIONS', 'after data cleaning')
    order = ['benign', 'malignant']
    palette = ['#024fc2', '#c70300']
    axes = ('amount', 'lesion_type')
    size = (6, 1.5)
    img_file = 'lesion_types_after_cleaning.svg'
    bar_chart(img_file, CHARTS_DIR, size, axes, df, title, palette,
              order=order)

    # 3. Diagnosis types.
    # 3.1 Query
    group_by = ['lesion_type', 'diagnosis']
    df = group_data(main_df, group_by)
    # 3.2 Chart
    title = ('TYPES OF DIAGNOSIS', 'after data cleaning')
    palette = ['#024fc2', '#c70300']
    axes = ('amount', 'diagnosis')
    hue = 'lesion_type'
    size = (6, 6)
    img_file = 'diagnosis_types.svg'
    bar_chart(img_file, CHARTS_DIR, size, axes, df, title, palette, hue=hue)

    # 4. Diagnosis confirmation type
    # 4.1 Query
    group_by = ['diagnosis_confirm_type']
    df = group_data(main_df, group_by)
    # 4.2 Chart
    title = ('DIAGNOSIS CONFIRMATION', 'after data cleaning')
    palette = ['#024fc2']
    axes = ('amount', 'diagnosis_confirm_type')
    size = (6, 2.25)
    img_file = 'diagnosis_confirmation_types.svg'
    bar_chart(img_file, CHARTS_DIR, size, axes, df, title, palette)

    # 5. Image sizes
    # 5.1 Height and width of landscape oriented images.
    main_df['height_landscape'] = main_df[['width', 'height']].min(axis=1)
    main_df['width_landscape'] = main_df[['width', 'height']].max(axis=1)
    # 5.2 Grouping by high and low resolution
    threshold = 2000
    main_df['resolution'] = 'lo'
    main_df.loc[main_df['height_landscape'] >= threshold, 'resolution'] = 'hi'
    # 5.3 Square root of image area - sqrt(width * height)
    main_df['sqrt_area'] = ((main_df['width_landscape']
                             * main_df['height_landscape'])
                            .pow(1. / 2)
                            .astype(int))
    # 6. Image area
    # 6.1 Query
    df = dim_into_bins(main_df, 'sqrt_area')
    # 6.2 Chart
    title = ('IMAGE SIZE', 'square root of area')
    palette = ['#666666']
    axes = ('amount', 'range')
    scale = 'linear'
    size = (6, 3.5)
    img_file = 'image_sizes_sqrt_area.svg'
    bar_chart(img_file, CHARTS_DIR, size, axes, df, title, palette,
              scale=scale)

    # 7. Image height
    # 7.1 Query
    df = dim_into_bins(main_df, 'height_landscape')
    # 7.2 Chart
    title = ('IMAGE SIZE', 'height')
    palette = ['#666666']
    axes = ('amount', 'range')
    size = (6, 3.2)
    img_file = 'image_sizes_height.svg'
    bar_chart(img_file, CHARTS_DIR, size, axes, df, title, palette,
              scale=scale)

    # 8. Save main_df as CSV file.
    save_data(main_df, CSV_DIR, CSV_FILE)


if __name__ == "__main__":
    main()
