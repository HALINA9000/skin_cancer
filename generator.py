import os  # Operating system functions
from tqdm import tqdm  # Progress bar
from io import BytesIO  # Buffered I/O implementation
from PIL import Image  # Image processing
import numpy as np  # Linear algebra
from keras.utils import Sequence


class StripeGenerator(Sequence):
    """
    Mini-batches generator.

    Load images stored on disk to memory, decompress them and then return
    as numpy arrays together with binary lesions type.
    """
    def __init__(self, df, stripes_dir, batch_size=8, dims=(1000, 2000),
                 channels=3, shuffle=True):
        """
        Initialization.

        Setting variables, loading content of files to memory and (optionally)
        shuffle dataset.

        Parameters
        ----------
            df : pandas.DataFrame
                Dataset containing file names of images (stripes) and lesions
                type in binary form (0: benign, 1: malignant)
            stripes_dir : str
                Full path of destination directory where images are stored.
            batch_size : int
                Size of mini-batch.
                Default: 8.
            dims : tuple(int, int)
                Width and height of images.
                Default: (1000, 2000).
            channels : int
                Number of channels.
                Default: 3 (RGB image).
            shuffle : bool
                Do or do not dataset shuffling.
                Default: True
        """
        self.df = df
        self.stripes_dir = stripes_dir
        self.batch_size = batch_size
        self.dims = dims
        self.channels = channels
        self.shuffle = shuffle

        self.file_content_dict = self.__load_files_to_memory()
        self.on_epoch_end()

    def __len__(self):
        """
        Calculate the number of mini-batches per epoch.

        Returns
        _______
            int
                Number of mini-batches.
        """
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        """
        Generate mini-batch with records number determined by batch_size.

        Returns
        _______
            batch_x_np : numpy.array
                Normalized images.
            batch_y_np : numpy.array(int)
                Binary lesion type (0: benign, 1: malignant).
        """
        batch_x_np = np.empty((self.batch_size, *self.dims, self.channels))
        batch_y_np = np.empty(self.batch_size, dtype=int)
        record_lo = index * self.batch_size
        record_hi = (index + 1) * self.batch_size
        for i, record_num in enumerate(range(record_lo, record_hi)):
            # Get file name
            stripe_file = self.df['stripe_file'][record_num]
            # Get content of file (compressed)
            stripe_comp_binary = self.file_content_dict[stripe_file]
            # Read as image
            stripe_obj = Image.open(BytesIO(stripe_comp_binary))
            # Convert to np.array (height, width, channels) and normalize
            stripe_np = np.asarray(stripe_obj) / 255.
            # Get lesion type
            binary_lesion_type = self.df['binary_lesion_type'][record_num]
            batch_x_np[i, ] = stripe_np.reshape(*self.dims, self.channels)
            batch_y_np[i] = binary_lesion_type
        return batch_x_np, batch_y_np

    def on_epoch_end(self):
        """
        If shuffle is set to True: shuffle dataset.
        """
        if self.shuffle:
            self.df = self.df.sample(frac=1)

    def __load_files_to_memory(self):
        """
        Pseudo RAM disk - load images as binary files to memory.

        Returns
        -------
            file_content_dict : dict[str, bytes]
                File names and their binary contents.
        """
        records_num = len(self.df)
        file_content_dict = {}
        for i in tqdm(range(records_num)):
            file = self.df['stripe_file'].iloc[i]
            file_path = os.path.join(self.stripes_dir, file)
            img_com_obj = open(file_path, 'rb')
            file_content_dict[file] = img_com_obj.read()
            img_com_obj.close()
        return file_content_dict
