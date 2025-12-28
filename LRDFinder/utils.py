import numpy as np
from typing import TypeAlias


NumpyArray: TypeAlias = np.ndarray


def fillna(array: NumpyArray, val: int|float=0.):
    """
    Replace NaN values in the input array with a specified value.
    
    Args:
        array (NumpyArray): Input numpy array of any dimension.
        val (int|float, optional): Value to replace NaN with. Defaults to 0.0.
    
    Returns:
        NumpyArray: Array with NaN values replaced.
        
    Note:
        This function modifies the input array in-place and also returns the modified array.
    """
    mask = np.isnan(array)
    array[mask] = val
    return array


def safe_reciprocal_divide(matrix: NumpyArray):
    """
    Safely compute the reciprocal of each element in the matrix, avoiding division by zero.
    
    Args:
        matrix (NumpyArray): Input numpy array of any dimension.
        
    Returns:
        NumpyArray: Array where each non-zero element is replaced by its reciprocal (1/element).
                    Zero elements remain zero in the output.
                    
    Note:
        - This function handles division by zero by returning 0 for any zero input elements.
        - The output array has dtype=np.float64 regardless of the input dtype.
        - The function does not modify the input array.
    """
    result = np.zeros_like(matrix, dtype=np.float64)
    np.divide(1, matrix, where=matrix!=0, out=result)
    return result


def pad_spec(spec: NumpyArray, length: int):
    """
    Pad or truncate a spectrum (or array of spectra) to a specified length.
    
    Args:
        spec (NumpyArray): Input spectrum array, can be 1D (single spectrum) or 2D (multiple spectra, 
                           where each row represents a spectrum).
        length (int): Target length to pad or truncate the spectrum(s) to. Must be a positive integer.
        
    Returns:
        NumpyArray: Padded or truncated spectrum array with the specified length.
                    - If input is 1D, returns a 1D array of length `length`.
                    - If input is 2D, returns a 2D array with shape (N, `length`), where N is the number 
                      of spectra in the input.
                    
    Raises:
        ValueError: If `length` is not a positive integer.
        ValueError: If the input `spec` has more than 2 dimensions.
        
    Notes:
        - If the input spectrum length is greater than `length`, the spectrum is truncated to the first `length` elements.
        - If the input spectrum length is less than `length`, the spectrum is padded with zeros to reach `length` elements.
        - The output array preserves the data type of the input array.
        - The function creates a new array and does not modify the input array.
    """
    if not isinstance(length, int) or length < 0:
        raise ValueError('length must be positive integer')

    if spec.ndim == 1:
        n_pixels = len(spec)
        if n_pixels >= length:
            paded_spec = spec[:length]
        else:
            paded_spec = np.zeros(length, dtype=spec.dtype)
            paded_spec[:n_pixels] = spec.copy()
    elif spec.ndim == 2:
        N , n_pixels = spec.shape
        if n_pixels >= length:
            paded_spec = spec.copy()
            paded_spec = paded_spec[:, :length]
        else:
            paded_spec = np.zeros((N, length), dtype=spec.dtype)
            paded_spec[:, :n_pixels] = spec.copy()
    else:
        raise ValueError('shape of input spec must be 1D or 2D.')
    return paded_spec


def random_shift(spec: NumpyArray, xl: int, xr: int):
    """
    Apply a random shift to a spectrum (or array of spectra) within a specified range.
    
    Args:
        spec (NumpyArray): Input spectrum array, can be 1D (single spectrum) or 2D (multiple spectra, 
                           where each row represents a spectrum).
        xl (int): Minimum shift amount (in pixels). Can be negative for leftward shifts.
        xr (int): Maximum shift amount (in pixels). Can be negative for leftward shifts.
        
    Returns:
        NumpyArray: Shifted spectrum array with the same shape as input.
                    - Pixels shifted out of bounds are lost
                    - Empty spaces created by shifting are filled with zeros
                    - Each spectrum in a 2D array receives an independent random shift
                    
    Raises:
        ValueError: If either xl or xr is not an integer.
        ValueError: If the input spec has more than 2 dimensions.
        
    Notes:
        - The shift amount dx is randomly chosen from the range [xl, xr] (inclusive).
        - For positive dx: spectrum is shifted left by dx pixels, right side padded with zeros.
        - For negative dx: spectrum is shifted right by |dx| pixels, left side padded with zeros.
        - The function creates a new array and does not modify the input array.
    """
    if not (isinstance(xl, int) and isinstance(xr, int)):
        raise ValueError(f'xr, xl must be int type!')
    
    if spec.ndim == 1:
        n_pixels = len(spec)
        shifted_spec = np.zeros_like(spec, dtype=spec.dtype)
        dx = np.random.randint(xl, xr + 1)
        
        if dx == 0:
            shifted_spec = spec.copy()
        elif dx > 0:
            valid_len = n_pixels - dx
            if valid_len > 0:
                shifted_spec[dx:] = spec[:valid_len]
        else:
            dx_abs = -dx
            valid_len = n_pixels - dx_abs
            if valid_len > 0:
                shifted_spec[:valid_len] = spec[dx_abs:]
    elif spec.ndim == 2:
        shifted_spec = np.zeros_like(spec)
        for i in range(spec.shape[0]):
            shifted_spec[i] = random_shift(spec[i], xl, xr)
    else:
        raise ValueError('spec must be 1D or 2D array')
    
    return shifted_spec


def add_noise(spec: NumpyArray, noise_std_max: float):
    """
    Add random Gaussian noise to non-zero elements of a spectrum (or array of spectra).
    
    Args:
        spec (NumpyArray): Input spectrum array, can be 1D (single spectrum) or 2D (multiple spectra, 
                           where each row represents a spectrum).
        noise_std_max (float): Maximum standard deviation for the Gaussian noise. 
                               Must be a positive float.
                               
    Returns:
        NumpyArray: Spectrum array with added noise, same shape and dtype as input.
                    - Noise is only added to non-zero elements of the input
                    - Zero elements remain zero in the output
                    - Each spectrum/element receives independent noise
                    
    Raises:
        ValueError: If noise_std_max is not a positive float.
        
    Notes:
        - Noise is generated as Gaussian (normal) distribution with zero mean
        - The standard deviation for the noise is randomly chosen between 0 and noise_std_max
          for each call to the function
        - The function creates a new array and does not modify the input array
    """
    if not isinstance(noise_std_max, float) or noise_std_max < 0:
        raise ValueError('noise_std_max must be positive float')
    mask = ~(spec == 0.)
    noise = np.random.randn(*spec.shape) * np.random.uniform(0, noise_std_max)
    return (spec + noise) * mask


def mask_random_pixels(spec: NumpyArray, k: float):
    """
    Randomly mask (set to 0) a specified percentage of pixels in a spectrum (or array of spectra).
    
    Args:
        spec (NumpyArray): Input spectrum array, can be 1D (single spectrum) or 2D (multiple spectra, 
                           where each row represents a spectrum).
        k (float): Percentage of pixels to mask, must be between 0 and 100 (inclusive).
        
    Returns:
        NumpyArray: Spectrum array with randomly masked pixels, same shape and dtype as input.
                    - Each spectrum in a 2D array has its own independent random masking
                    - Masked pixels are set to 0
                    - The number of masked pixels is approximately k% of total pixels
                    
    Raises:
        ValueError: If k is not between 0 and 100.
        ValueError: If the input spec has more than 2 dimensions.
        
    Notes:
        - The exact number of masked pixels is rounded to the nearest integer
        - Masking is done randomly without replacement (each pixel can be masked at most once)
        - The function creates a new array and does not modify the input array
    """
    if not (0 <= k <= 100):
        raise ValueError("k must be within 0 and 100")
    
    if spec.ndim == 1:
        n_pixels = len(spec)
        n_mask = int(round(n_pixels * k / 100))
        masked_spec = spec.copy()
        if n_mask > 0:
            mask_indices = np.random.choice(n_pixels, size=n_mask, replace=False)
            masked_spec[mask_indices] = 0.
    elif spec.ndim == 2:
        masked_spec = spec.copy()
        for i in range(spec.shape[0]):
            masked_spec[i] = mask_random_pixels(spec[i], k)
    else:
        raise ValueError('spec must be 1D or 2D array')
    
    return masked_spec



