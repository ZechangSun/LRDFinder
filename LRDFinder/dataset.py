import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from astropy.io import fits

from .utils import *



class LRDDataset(Dataset):
    def __init__(self, 
                pos_fits_files: List[str], 
                neg_fits_files: List[str], 
                npixels: int=255,
                random_mask_ratio: float=0.1,
                random_shift: Tuple[int]=(-25, 75),
                noise_std_max: float=0.05,
                is_train: bool=True):
        """Initialize the LRDDataset with spectral data from FITS files.
        
        Creates a PyTorch Dataset for handling spectral data, loading both positive and negative
        class samples, processing them, and preparing them for training or evaluation.
        
        Args:
            pos_fits_files (List[str]): List of file paths to FITS files containing positive class spectra
            neg_fits_files (List[str]): List of file paths to FITS files containing negative class spectra
            npixels (int, optional): Number of pixels per spectrum after padding/truncation. Defaults to 255.
            random_mask_ratio (float, optional): Ratio of pixels to randomly mask during training (0.0-1.0). Defaults to 0.1.
            random_shift (Tuple[int], optional): Range of random shifts to apply during training (min, max). Defaults to (-25, 75).
            noise_std_max (float, optional): Maximum standard deviation for random noise added during training. Defaults to 0.05.
            is_train (bool, optional): Flag indicating if the dataset is for training (True) or evaluation (False). Defaults to True.
            
        Attributes:
            pos_fits_files (List[str]): Positive class FITS file paths
            neg_fits_files (List[str]): Negative class FITS file paths
            npixels (int): Target number of pixels per spectrum
            is_train (bool): Training/Evaluation mode flag
            k (float): Random mask percentage (random_mask_ratio * 100)
            xl (int): Minimum random shift value
            xr (int): Maximum random shift value
            noise_std_max (float): Maximum noise standard deviation
            specs (numpy.ndarray): Concatenated spectral data of shape (n_samples, npixels)
            errors (numpy.ndarray): Concatenated error data of shape (n_samples, npixels)
            labels (numpy.ndarray): Concatenated labels of shape (n_samples,) (1 for positive, 0 for negative)
            radecs (numpy.ndarray): Concatenated RA/Dec coordinates of shape (n_samples, 2)
            
        Notes:
            - The dataset combines positive and negative samples into unified arrays
            - All data is loaded and processed during initialization using _load_all_specs
            - Training mode parameters (mask, shift, noise) are only used in __getitem__ during training
        """
        self.pos_fits_files = pos_fits_files
        self.neg_fits_files = neg_fits_files
        self.npixels = npixels
        self.is_train = is_train
        
        self.k = random_mask_ratio * 100
        self.xl, self.xr = random_shift
        self.noise_std_max = noise_std_max

        pos_specs, pos_errors, pos_radecs = self._load_all_specs(self.pos_fits_files)
        neg_specs, neg_errors, neg_radecs = self._load_all_specs(self.neg_fits_files)
        
        pos_labels = np.ones(pos_specs.shape[0], dtype=np.float32)
        neg_labels = np.zeros(neg_specs.shape[0], dtype=np.float32)
        
        if len(self.neg_fits_files) != 0 and len(self.pos_fits_files) != 0:
            self.specs = np.concatenate([pos_specs, neg_specs], axis=0).astype(np.float32)
            self.errors = np.concatenate([pos_errors, neg_errors], axis=0).astype(np.float32)
            self.labels = np.concatenate([pos_labels, neg_labels], axis=0).astype(np.float32)
            self.radecs =  np.concatenate([pos_radecs, neg_radecs], axis=0).astype(np.float32)
        elif len(self.neg_fits_files) != 0 and len(self.pos_fits_files) == 0:
            self.specs = neg_specs.astype(np.float32)
            self.errors = neg_errors.astype(np.float32)
            self.labels = neg_labels.astype(np.float32)
            self.radecs = neg_radecs.astype(np.float32)
        elif len(self.neg_fits_files) == 0 and len(self.pos_fits_files) != 0:
            self.specs = pos_specs.astype(np.float32)
            self.errors = pos_errors.astype(np.float32)
            self.labels = pos_labels.astype(np.float32)
            self.radecs = pos_radecs.astype(np.float32)
        else:
            raise ValueError("At least one of pos_fits_files or neg_fits_files must be non-empty.")
    
    def _load_all_specs(self, file_paths: List[str]) -> NumpyArray:
        """Load and concatenate spectral data from multiple FITS files.
        
        Reads spectral data, error data, and RA/Dec coordinates from a list of FITS files
        and combines them into unified arrays for use in the dataset.
        
        Args:
            file_paths (List[str]): List of file paths to FITS files containing spectral data
            
        Returns:
            tuple: Three concatenated numpy arrays:
                - specs (NumpyArray): Combined spectral data with shape (total_samples, npixels)
                - errors (NumpyArray): Combined error data with shape (total_samples, npixels)
                - radecs (NumpyArray): Combined RA/Dec coordinates with shape (total_samples, 2)
            
        Notes:
            - Returns an empty numpy array if no file paths are provided
            - Uses `_load_from_fits_file` to process individual files
            - Concatenates arrays along axis=0 (sample dimension)
            - Typically called twice during initialization: once for positive samples and once for negative samples
        """
        if len(file_paths) == 0:
            return np.array([]), np.array([]), np.array([])
        all_specs, all_errors, all_radecs = [], [], []
        for path in file_paths:
            spec, error, radec = self._load_from_fits_file(path)
            all_specs.append(spec)
            all_errors.append(error)
            all_radecs.append(radec)
        return np.concatenate(all_specs, axis=0), np.concatenate(all_errors, axis=0), np.concatenate(all_radecs, axis=0)
    
    def _load_from_fits_file(self, path: str) -> NumpyArray:
        """
        Load and preprocess spectral data from a FITS file.
        
        Extracts spectrum, error, and RA/Dec coordinates from a FITS file,
        then applies preprocessing steps to prepare the data for use in the model.
        
        Args:
            path (str): Path to the FITS file containing spectral data.
            
        Returns:
            tuple: A tuple containing three numpy arrays:
                - spec (NumpyArray): Preprocessed spectral data with shape (n_samples, npixels)
                - error (NumpyArray): Preprocessed error data with shape (n_samples, npixels)
                - radec (NumpyArray): RA/Dec coordinates with shape (n_samples, 2)
            
        Notes:
            - The FITS file is expected to have a specific structure:
              * HDU 1: Contains table data with optional 'RA' and 'DEC' columns
              * HDU 2: Contains spectral data
              * HDU 3: Contains error data
            - If RA/Dec columns are not present, zeros are used as placeholder coordinates.
            - Error data is processed using safe_reciprocal_divide to avoid division by zero.
            - NaN values in both spectrum and error are filled with 0.0.
            - Data is padded or truncated to match self.npixels length.
            - Spectrum values are clipped to the range [-3.0, 2.0].
            - Error values are clipped to the range [0.0, 2.0].
        """
        with fits.open(path) as hdul:
            if 'ra' in hdul['PARAMETERS'].data.columns.names and 'dec' in hdul['PARAMETERS'].data.columns.names:
                ra, dec = hdul['PARAMETERS'].data['RA'], hdul['PARAMETERS'].data['DEC']
            else:
                ra, dec = np.zeros(hdul['PARAMETERS'].data.shape[0]), np.zeros(hdul['PARAMETERS'].data.shape[0])
            radec = np.vstack([ra, dec]).T
            spec = hdul['FLUX_NOISY'].data
            error = safe_reciprocal_divide(hdul['IVAR'].data)**0.5
        spec = fillna(spec, val=0.)
        error = fillna(error, val=0.)
        spec = pad_spec(spec, self.npixels)
        error = pad_spec(error, self.npixels)
        
        spec = np.clip(spec, a_min=-2., a_max=2.)
        error = np.clip(error, a_min=0., a_max=2.)
        return spec, error, radec
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Implements the PyTorch Dataset interface method to get the dataset size.
        
        Returns:
            int: Total number of samples in the dataset, equal to the length of self.labels.
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Implements the PyTorch Dataset interface method to retrieve individual samples.
        Applies data augmentation during training and returns relevant data fields.
        
        Args:
            idx (int): Index of the data sample to retrieve.
            
        Returns:
            dict: Dictionary containing the data sample with the following keys:
                - spec (torch.Tensor): Spectral data tensor with shape matching input spec.
                - error (torch.Tensor): Error data tensor with shape matching input error.
                - label (torch.Tensor): Label tensor for the sample.
                - radec (torch.Tensor, optional): RA/Dec coordinates tensor, only returned during evaluation.
        
        Notes:
            - During training (`self.is_train=True`), applies random_shift and mask_random_pixels augmentations.
            - Always applies a mask to remove pixels where either spec or error is 0.
            - All returned tensors are converted to torch.float32 dtype.
            - RA/Dec coordinates are only included in the returned dictionary during evaluation.
        """
        spec, label, error, radec = self.specs[idx], self.labels[idx], self.errors[idx], self.radecs[idx]
        if self.is_train:
            spec = random_shift(spec, self.xl, self.xr)
            spec = mask_random_pixels(spec, self.k)
            mask = (spec != 0.) & (error != 0.)
            spec = spec * mask
            error = error * mask
            return {'spec': torch.tensor(spec, dtype=torch.float32), 
                    'error': torch.tensor(error, dtype=torch.float32), 
                    'label': torch.tensor(label, dtype=torch.float32)}
        else:
            mask = (spec != 0.) & (error != 0.)
            spec = spec * mask
            error = error * mask
            return {"spec": torch.tensor(spec, dtype=torch.float32), 
                    "error": torch.tensor(error, dtype=torch.float32), 
                    "label": torch.tensor(label, dtype=torch.float32), 
                    "radec": torch.tensor(radec, dtype=torch.float32)}


