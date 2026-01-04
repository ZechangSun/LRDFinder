"""
MockProfileGenerator - Optimized Mock Emission Line Profile Generator

A high-performance class for generating mock emission line profiles with various components.
Optimized for large-scale spectroscopic analysis (>10k spectra) with efficient memory usage,
batch processing, and thread-safe operations.

Features:
- Lazy loading with intelligent caching
- Batch processing for multiple spectra
- Memory-efficient data structures
- Multi-spectrum FITS output format
- Support for broad+narrow and broad+absorption profiles
"""

import os
import threading
from typing import Optional, List, Dict, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from astropy.io import fits


# Constants
DEFAULT_VEL_RANGE = 2000  # km/s
DEFAULT_CACHE_SIZE = 1000
DEFAULT_NORM_RANGE = 100  # km/s
DEFAULT_BROAD_RATIO_RANGE = (2, 10)
DEFAULT_BROAD_FWHM_RANGE = (900, 1500)  # km/s
DEFAULT_ABS_EW_RANGE = (2, 10)  # Angstrom
DEFAULT_ABS_VEL_RANGE = (-300, 100)  # km/s
DEFAULT_ABS_FWHM_RANGE = (80, 250)  # km/s

# Rest wavelengths
REST_WAVELENGTHS = {
    'Ha': 6564.614,  # H-alpha
    'Hb': 4862.683,  # H-beta
}



class MockProfileGenerator:
    """
    High-performance mock emission line profile generator.

    Generates realistic spectroscopic profiles with:
    - Broad emission components with configurable parameters
    - Absorption features with variable strength and width
    - Multi-spectrum FITS output compatible with DESI format
    """

    # ============================================================================
    # Initialization and Core Data Management
    # ============================================================================

    def __init__(self, halpha_file: Optional[str] = None, hbeta_file: Optional[str] = None,
                 max_cache_size: int = DEFAULT_CACHE_SIZE) -> None:
        """
        Initialize the MockProfileGenerator with spectra data.

        Parameters:
        -----------
        halpha_file : str or None
            Path to H-alpha narrow spectra FITS file (optional)
        hbeta_file : str or None
            Path to H-beta narrow spectra FITS file (optional)
        max_cache_size : int
            Maximum number of processed spectra to cache (default: 1000)
        """
        # Core attributes
        self.halpha_file = halpha_file
        self.hbeta_file = hbeta_file
        self.max_cache_size = max_cache_size

        # Data storage
        self.narrow_spec_ha: Optional[fits.HDUList] = None
        self.narrow_spec_hb: Optional[fits.HDUList] = None

        # Lazy loading state
        self._ha_loaded = False
        self._hb_loaded = False

        # File information (pre-loaded for validation)
        self._ha_info: Optional[Dict] = None
        self._hb_info: Optional[Dict] = None

        # Thread-safe caching
        self._cache_lock = threading.Lock()
        self._processed_cache: Dict[str, Tuple] = {}

        # Performance monitoring
        self._perf_stats = {
            'total_spectra_processed': 0,
            'total_processing_time': 0.0,
            'avg_spectra_per_second': 0.0,
            'last_batch_size': 0,
            'last_batch_time': 0.0
        }

        # Initialize file information
        self._init_file_info()

    def _init_file_info(self) -> None:
        """Initialize file information for validation."""
        if self.halpha_file and os.path.exists(self.halpha_file):
            self._ha_info = self._get_fits_info(self.halpha_file)
        if self.hbeta_file and os.path.exists(self.hbeta_file):
            self._hb_info = self._get_fits_info(self.hbeta_file)

    def _get_fits_info(self, filename: str) -> Dict:
        """Get basic info about FITS file without loading full data."""
        try:
            with fits.open(filename) as hdul:
                n_spectra = len(hdul['PARAMETERS'].data)
                return {'n_spectra': n_spectra, 'filename': filename}
        except Exception as e:
            raise RuntimeError(f"Error reading FITS file {filename}: {e}")

    def _ensure_loaded(self, line: str) -> None:
        """Ensure the appropriate spectra file is loaded."""
        if line == 'Ha' and not self._ha_loaded and self.halpha_file:
            if not os.path.exists(self.halpha_file):
                raise FileNotFoundError(f"H-alpha spectra file not found: {self.halpha_file}")
            self.narrow_spec_ha = fits.open(self.halpha_file)
            self._ha_loaded = True
        elif line == 'Hb' and not self._hb_loaded and self.hbeta_file:
            if not os.path.exists(self.hbeta_file):
                raise FileNotFoundError(f"H-beta spectra file not found: {self.hbeta_file}")
            self.narrow_spec_hb = fits.open(self.hbeta_file)
            self._hb_loaded = True

    def _get_cache_key(self, idx: int, line: str) -> str:
        """Generate cache key for processed spectra."""
        return f"{line}_{idx}"

    def _get_spectra_count(self, line: str) -> int:
        """Get the number of available spectra for a given line."""
        info = self._ha_info if line == 'Ha' else self._hb_info
        return info['n_spectra'] if info else 0

    # ============================================================================
    # Data Processing and Transformation Methods
    # ============================================================================

    def _process_spectra_data(self, idx: int, line: str = 'Ha', use_cache: bool = True) -> Tuple:
        """
        Process spectra data for a given index with caching.

        Parameters:
        -----------
        idx : int
            Index of the spectrum to process
        line : str
            Emission line ('Ha' for H-alpha, 'Hb' for H-beta)
        use_cache : bool
            Whether to use cached results

        Returns:
        --------
        wave : ndarray
            Wavelength array
        flux : ndarray
            Flux array
        err : ndarray
            Error array
        z : float
            Redshift
        """
        cache_key = self._get_cache_key(idx, line)

        # Check cache first
        if use_cache:
            with self._cache_lock:
                if cache_key in self._processed_cache:
                    return self._processed_cache[cache_key]

        # Ensure spectra are loaded
        self._ensure_loaded(line)
        narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb

        if narrow_spec is None:
            raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Process the data
        params = narrow_spec['PARAMETERS'].data[idx]
        flux = narrow_spec['FLUX_NOISY'].data[idx]
        ivar = narrow_spec['IVAR'].data[idx]

        z = params['REDSHIFT']
        wav_min = params['WAV_MIN']
        dwave = params['DWAVE']
        n_pix = params['N_PIXELS']
        wave = wav_min + np.arange(n_pix) * dwave

        flux = flux[:n_pix]
        # ivar is inverse variance, so error is 1/sqrt(ivar)
        err = np.where(ivar[:n_pix] > 0, 1.0 / np.sqrt(ivar[:n_pix]), np.inf)

        result = (wave, flux, err, z)

        # Cache the result
        if use_cache:
            with self._cache_lock:
                if len(self._processed_cache) >= self.max_cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._processed_cache))
                    del self._processed_cache[oldest_key]
                self._processed_cache[cache_key] = result

        return result

    def _process_batch_spectra_data(self, indices: List[int], line: str = 'Ha') -> Tuple[List, List, List, np.ndarray]:
        """
        Process multiple spectra efficiently in batch.

        Parameters:
        -----------
        indices : array-like
            Indices of spectra to process
        line : str
            Emission line ('Ha' for H-alpha, 'Hb' for H-beta)

        Returns:
        --------
        waves : list
            List of wavelength arrays
        fluxes : list
            List of flux arrays
        errs : list
            List of error arrays
        zs : ndarray
            Array of redshifts
        """
        indices = np.asarray(indices)
        self._ensure_loaded(line)
        narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb

        if narrow_spec is None:
            raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Pre-allocate arrays for efficiency
        n_spectra = len(indices)
        waves, fluxes, errs = [], [], []
        zs = np.zeros(n_spectra)

        # Process all spectra at once where possible
        params_all = narrow_spec['PARAMETERS'].data[indices]
        flux_all = narrow_spec['FLUX_NOISY'].data[indices]
        ivar_all = narrow_spec['IVAR'].data[indices]

        zs = params_all['REDSHIFT']
        wav_min = params_all['WAV_MIN']
        dwave = params_all['DWAVE']
        n_pix = params_all['N_PIXELS']

        for i, idx in enumerate(indices):
            wave = wav_min[i] + np.arange(n_pix[i]) * dwave[i]
            flux = flux_all[i][:n_pix[i]]
            ivar = ivar_all[i][:n_pix[i]]

            # ivar is inverse variance, so error is 1/sqrt(ivar)
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.inf)

            waves.append(wave)
            fluxes.append(flux)
            errs.append(err)

            # Cache individual results
            cache_key = self._get_cache_key(idx, line)
            with self._cache_lock:
                if len(self._processed_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self._processed_cache))
                    del self._processed_cache[oldest_key]
                self._processed_cache[cache_key] = (wave, flux, err, zs[i])

        return waves, fluxes, errs, zs

    # ============================================================================
    # Single Spectrum Profile Generation
    # ============================================================================


    def _create_multi_spectrum_hdul(self, all_params: List, all_fluxes: List[np.ndarray],
                                   all_ivars: List[np.ndarray], profile_type: str,
                                   line: str, original_indices: Optional[List[int]] = None) -> fits.HDUList:
        """Create a multi-spectrum HDUList from processed data."""
        if not all_params:
            raise ValueError("No spectra data provided")

        # Find the maximum length to pad all arrays
        max_length = max(len(flux) for flux in all_fluxes)

        # Pad arrays to maximum length
        padded_fluxes = []
        padded_ivars = []
        for flux, ivar in zip(all_fluxes, all_ivars):
            flux_padded = np.pad(flux, (0, max_length - len(flux)), 'constant', constant_values=0.0)
            ivar_padded = np.pad(ivar, (0, max_length - len(ivar)), 'constant', constant_values=0.0)
            padded_fluxes.append(flux_padded)
            padded_ivars.append(ivar_padded)

        # Convert to numpy arrays
        flux_array = np.array(padded_fluxes)
        ivar_array = np.array(padded_ivars)

        # Create PARAMETERS table
        if all_params:
            # Get column definitions from the original spectra
            narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
            original_hdu = narrow_spec['PARAMETERS']
            columns = []

            # Create columns for each field
            for col_name in original_hdu.data.dtype.names:
                col_data = np.array([params_row[col_name] for params_row in all_params])
                col_format = original_hdu.columns[col_name].format
                columns.append(fits.Column(name=col_name, format=col_format, array=col_data))

            # Update spectrum_id to preserve original indices or be sequential
            if 'spectrum_id' in original_hdu.data.dtype.names:
                spectrum_id_col = columns[original_hdu.data.dtype.names.index('spectrum_id')]
                if original_indices is not None:
                    spectrum_id_col.array = np.array(original_indices)
                else:
                    spectrum_id_col.array = np.arange(len(all_params))

            params_hdu = fits.BinTableHDU.from_columns(columns, name='PARAMETERS')
        else:
            params_hdu = fits.BinTableHDU(data=None, name='PARAMETERS')

        # Create HDUs
        primary_hdu = fits.PrimaryHDU()
        flux_hdu = fits.ImageHDU(data=flux_array, name='FLUX_NOISY')
        ivar_hdu = fits.ImageHDU(data=ivar_array, name='IVAR')

        # Add metadata
        primary_hdu.header['PROFILE'] = profile_type
        primary_hdu.header['LINE'] = line
        primary_hdu.header['N_SPEC'] = len(all_params)
        primary_hdu.header['MAX_PIX'] = max_length

        return fits.HDUList([primary_hdu, params_hdu, flux_hdu, ivar_hdu])


    def _generate_broad_plus_narrow_batch_parallel(self, indices: List[int], line: str,
                                                  broad_to_narrow_ratio_range: Tuple[float, float],
                                                  broad_fwhm_range: Tuple[float, float],
                                                  n_processes: Optional[int] = None,
                                                  vel_range: float = DEFAULT_VEL_RANGE) -> Tuple:
        """
        Generate broad + narrow mock line profiles using parallel processing for both data processing and mock generation.

        Parameters:
        -----------
        indices : List[int]
            Indices of spectra to process
        line : str
            Emission line ('Ha' or 'Hb')
        broad_to_narrow_ratio_range : Tuple[float, float]
            Range for random broad/narrow ratio
        broad_fwhm_range : Tuple[float, float]
            Range for random broad FWHM in km/s
        n_processes : int or None
            Number of processes to use. If None, uses CPU count.
        vel_range : float
            Velocity range in km/s

        Returns:
        --------
        Tuple of processed flux, error, and parameter arrays
        """
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(indices))

        if n_processes == 1:
            print(f"🔄 Using sequential processing for broad+narrow generation (single process)")
            # For single process, use sequential processing
            return self._generate_broad_plus_narrow_batch_sequential(indices, line, broad_to_narrow_ratio_range,
                                                                   broad_fwhm_range, vel_range)

        import time
        start_time = time.time()

        print(f"🚀 Starting broad+narrow generation: {n_processes} processes, {len(indices)} spectra")

        # Load data in main process with minimal memory usage
        print(f"📂 Loading FITS data efficiently...")
        self._ensure_loaded(line)
        narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
        if narrow_spec is None:
            raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Get rest wavelength
        rest_wave = REST_WAVELENGTHS[line]

        # Extract only the data for requested indices - no unnecessary copying
        print(f"📊 Extracting data for {len(indices)} spectra...")
        indices_array = np.array(indices)
        flux_data = narrow_spec['FLUX_NOISY'].data[indices_array]  # Direct indexing
        ivar_data = narrow_spec['IVAR'].data[indices_array]
        params_data = narrow_spec['PARAMETERS'].data[indices_array]

        # Pre-compute all wavelength arrays to avoid reconstruction in workers
        print(f"🔢 Pre-computing wavelength arrays...")
        raw_data = []
        for i, idx in enumerate(indices):
            params = params_data[i]
            z = float(params['REDSHIFT'])

            # Pre-compute wavelength array once
            wav_min = float(params['WAV_MIN'])
            dwave = float(params['DWAVE'])
            n_pix = int(params['N_PIXELS'])
            wave = wav_min + np.arange(n_pix, dtype=np.float64) * dwave

            # Get flux and error data
            flux = flux_data[i][:n_pix].astype(np.float64)
            ivar_truncated = ivar_data[i][:n_pix]
            err = np.where(ivar_truncated > 0, 1.0 / np.sqrt(ivar_truncated), np.inf).astype(np.float64)

            # Pass pre-computed arrays: (wave, flux, err, z, idx)
            raw_data.append((wave, flux, err, z, idx))

        # Split data into chunks for multiprocessing
        chunk_size = max(1, len(raw_data) // n_processes)
        data_chunks = [raw_data[i:i + chunk_size] for i in range(0, len(raw_data), chunk_size)]
        print(f"📦 Split into {len(data_chunks)} chunks ({chunk_size} spectra per chunk)")

        # Launch parallel processing
        print(f"⚡ Starting broad+narrow parallel processing...")
        chunk_results = {}
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            # Submit tasks for each chunk with index
            future_to_index = {}
            for i, chunk in enumerate(data_chunks):
                # Use different random seed for each chunk to ensure reproducibility
                chunk_seed = 42 + i
                future = executor.submit(_generate_broad_plus_narrow_chunk_worker, chunk, rest_wave,
                                       broad_to_narrow_ratio_range, broad_fwhm_range, chunk_seed)
                future_to_index[future] = i

            # Collect results and store by index
            completed_chunks = 0
            for future in as_completed(future_to_index):
                try:
                    chunk_idx = future_to_index[future]
                    result = future.result()
                    chunk_results[chunk_idx] = result
                    completed_chunks += 1
                    print(f"✅ Completed broad+narrow chunk {completed_chunks}/{len(data_chunks)} (index {chunk_idx})")
                except Exception as exc:
                    raise RuntimeError(f'Broad+narrow generation failed: {exc}')

        print(f"🎯 All {len(data_chunks)} broad+narrow chunks completed")

        # Combine results in original order using pre-allocated arrays for speed
        if not chunk_results:
            return [], [], []

        print(f"🔄 Fast recombination...")
        total_spectra = len(indices)
        all_fluxes = [None] * total_spectra
        all_ivars = [None] * total_spectra
        all_indices = [None] * total_spectra

        # Fill results in correct order
        spectrum_idx = 0
        for chunk_idx in range(len(data_chunks)):
            if chunk_idx in chunk_results:
                fluxes, ivars, indices_chunk = chunk_results[chunk_idx]
                chunk_size_actual = len(fluxes)

                for j in range(chunk_size_actual):
                    all_fluxes[spectrum_idx + j] = fluxes[j]
                    all_ivars[spectrum_idx + j] = ivars[j]
                    all_indices[spectrum_idx + j] = indices_chunk[j]

                spectrum_idx += chunk_size_actual

        end_time = time.time()
        processing_time = end_time - start_time

        # Update performance statistics
        self._perf_stats['total_spectra_processed'] += len(indices)
        self._perf_stats['total_processing_time'] += processing_time
        self._perf_stats['avg_spectra_per_second'] = self._perf_stats['total_spectra_processed'] / self._perf_stats['total_processing_time']
        self._perf_stats['last_batch_size'] = len(indices)
        self._perf_stats['last_batch_time'] = processing_time

        print(f"✨ Broad+narrow generation complete: {len(all_fluxes)} spectra processed in {processing_time:.2f}s ({len(all_fluxes)/processing_time:.1f} spectra/s)")
        return all_fluxes, all_ivars, all_indices

    def _generate_broad_plus_narrow_batch_sequential(self, indices: List[int], line: str,
                                                    broad_to_narrow_ratio_range: Tuple[float, float],
                                                    broad_fwhm_range: Tuple[float, float],
                                                    vel_range: float = DEFAULT_VEL_RANGE) -> Tuple:
        """
        Sequential version of broad + narrow generation for small batches or fallback.
        """
        waves_batch, fluxes_batch, errs_batch, zs = self._process_batch_spectra_data(indices, line)
        rest_wave = self._get_rest_wavelength(line)

        # Convert to velocity space and normalize
        line_waves, line_fluxes, line_flux_errs, vel_to_lines = self._convert_to_velocity_space_batch(
            waves_batch, fluxes_batch, errs_batch, zs, rest_wave, vel_range)
        norm_fluxes, norm_flux_errs, _ = self._normalize_spectra_batch(
            line_fluxes, line_flux_errs, vel_to_lines)

        # Set random seed for reproducibility
        np.random.seed(42)

        all_fluxes = []
        all_ivars = []

        for line_wave, norm_flux, norm_flux_err, z in zip(line_waves, norm_fluxes, norm_flux_errs, zs):
            # Calculate normalized narrow component integrated flux
            norm_range_flag = np.abs((line_wave / (1 + z) - rest_wave) / rest_wave * 299792.458) < DEFAULT_NORM_RANGE
            norm_narrow_val = np.trapz(norm_flux[norm_range_flag & ~np.isnan(norm_flux)],
                                       line_wave[norm_range_flag & ~np.isnan(norm_flux)])

            # Generate random broad component parameters
            broad_to_narrow_ratio = np.random.uniform(*broad_to_narrow_ratio_range)
            broad_fwhm = np.random.uniform(*broad_fwhm_range)

            # Generate broad component
            broad_gaussian = self._generate_broad_component(line_wave, z, rest_wave, broad_to_narrow_ratio, broad_fwhm, norm_narrow_val)

            # Combine components
            composite = norm_flux + broad_gaussian

            # Renormalize
            rest_wave_range = line_wave / (1 + z)
            norm_range_flag = (rest_wave_range > rest_wave - 2) & (rest_wave_range < rest_wave + 2)
            if np.any(norm_range_flag):
                renorm = np.nanmax(composite[norm_range_flag])
            else:
                renorm = np.nanmax(composite)
            norm_composite = composite / renorm
            norm_composite_err = norm_flux_err / renorm

            # Convert to inverse variance
            ivar = np.where(norm_composite_err > 0, 1.0 / (norm_composite_err ** 2), 0.0)

            all_fluxes.append(norm_composite)
            all_ivars.append(ivar)

        return all_fluxes, all_ivars, indices

    def generate_broad_plus_narrow_batch(self, indices: List[int], line: str = 'Ha',
                                        vel_range: float = DEFAULT_VEL_RANGE,
                                        broad_to_narrow_ratio_range: Tuple[float, float] = DEFAULT_BROAD_RATIO_RANGE,
                                        broad_fwhm_range: Tuple[float, float] = DEFAULT_BROAD_FWHM_RANGE,
                                        use_multiprocessing: bool = True,
                                        n_processes: Optional[int] = None) -> fits.HDUList:
        """
        Generate broad + narrow mock line profiles for multiple spectra efficiently.
        Uses parallel processing for both data processing and mock spectrum generation.

        Parameters:
        -----------
        indices : List[int]
            Indices of spectra to process
        line : str
            Emission line ('Ha' or 'Hb')
        vel_range : float
            Velocity range in km/s
        broad_to_narrow_ratio_range : Tuple[float, float]
            Range for random broad/narrow ratio
        broad_fwhm_range : Tuple[float, float]
            Range for random broad FWHM in km/s
        use_multiprocessing : bool
            Whether to use multiprocessing for parallel processing
        n_processes : int or None
            Number of processes to use. If None, uses CPU count.

        Returns:
        --------
        hdul : HDUList
            Single HDUList object containing multiple spectra
        """
        indices = np.asarray(indices)

        # Choose processing method based on multiprocessing option
        if use_multiprocessing:
            fluxes_batch, ivars_batch, processed_indices = self._generate_broad_plus_narrow_batch_parallel(
                indices.tolist(), line, broad_to_narrow_ratio_range, broad_fwhm_range, n_processes, vel_range)
        else:
            fluxes_batch, ivars_batch, processed_indices = self._generate_broad_plus_narrow_batch_sequential(
                indices, line, broad_to_narrow_ratio_range, broad_fwhm_range, vel_range)

        # Ensure spectra are loaded for parameter extraction
        self._ensure_loaded(line)
        narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
        if narrow_spec is None:
            raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Extract all parameters at once using vectorized indexing (much faster than looping)
        # Convert to list of numpy records (preserves field access like params['REDSHIFT'])
        all_params = [row for row in narrow_spec['PARAMETERS'].data[processed_indices]]

        # fluxes_batch and ivars_batch are already in the correct order
        all_fluxes = fluxes_batch
        all_ivars = ivars_batch

        # Create multi-spectrum HDUList
        return self._create_multi_spectrum_hdul(all_params, all_fluxes, all_ivars, 'broad_plus_narrow', line, processed_indices)

    def generate_all_profiles_batch(self, n_spectra: int, line: str = 'Ha', 
                                  broad: bool = True, broad_abs: bool = True,
                                  vel_range: float = DEFAULT_VEL_RANGE,
                                  broad_to_narrow_ratio_range: Tuple[float, float] = DEFAULT_BROAD_RATIO_RANGE,
                                  broad_fwhm_range: Tuple[float, float] = DEFAULT_BROAD_FWHM_RANGE,
                                  abs_ew_range: Tuple[float, float] = DEFAULT_ABS_EW_RANGE,
                                  abs_vel_range: Tuple[float, float] = DEFAULT_ABS_VEL_RANGE,
                                  abs_fwhm_range: Tuple[float, float] = DEFAULT_ABS_FWHM_RANGE,
                                  output_dir: Optional[str] = None,
                                  save_broad: Union[bool, str] = True, save_absorption: Union[bool, str] = True,
                                  random_seed: Optional[int] = None, use_multiprocessing: bool = True,
                                  n_processes: Optional[int] = None) -> Dict[str, Union[fits.HDUList, Dict]]:
        """
        Generate batches of selected profile types with randomly selected spectra.

        Parameters:
        -----------
        n_spectra : int
            Number of spectra to generate (randomly selected from available spectra, with replacement if needed)
        line : str
            Emission line ('Ha' or 'Hb')
        vel_range : float
            Velocity range in km/s
        broad_to_narrow_ratio_range : tuple
            Range for random broad/narrow ratio
        broad_fwhm_range : tuple
            Range for random broad FWHM in km/s
        abs_ew_range : tuple
            Range for random absorption equivalent width
        abs_vel_range : tuple
            Range for random absorption velocity offset
        abs_fwhm_range : tuple
            Range for random absorption FWHM
        output_dir : str or None
            Directory to save FITS files (if None, files are not saved)
        broad : bool
            Whether to generate broad+narrow profiles
        broad_abs : bool
            Whether to generate broad+absorption+narrow profiles
        save_broad : bool or str
            Whether to save broad+narrow profiles to FITS. If True, uses default filename with parameters.
            If str, uses the string as the filename.
        save_absorption : bool or str
            Whether to save broad+absorption+narrow profiles to FITS. If True, uses default filename with parameters.
            If str, uses the string as the filename.
        random_seed : int or None
            Random seed for reproducible results
        use_multiprocessing : bool
            Whether to use multiprocessing for parallel processing
        n_processes : int or None
            Number of processes to use. If None, uses CPU count.

        Returns:
        --------
        results : dict
            Dictionary containing requested generated profiles:
            - 'broad': HDUList object containing multiple broad+narrow profiles (if broad=True)
            - 'absorption': HDUList object containing multiple broad+absorption+narrow profiles (if broad_abs=True)
            Each HDUList has the same format as input FITS files with 2D arrays
            - 'metadata': dict with generation parameters and file paths
        """
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Get available spectra count and randomly select indices
        available_info = self._ha_info if line == 'Ha' else self._hb_info
        if available_info is None:
            raise ValueError(f"No {line} spectra information available. Please provide the corresponding FITS file.")

        max_index = available_info['n_spectra'] - 1
        # Allow replacement if n_spectra > available spectra
        replace = n_spectra > available_info['n_spectra']

        indices = np.random.choice(max_index + 1, size=n_spectra, replace=replace)
        indices = np.sort(indices)  # Sort for consistent ordering (may contain duplicates if replacement used)

        # Initialize variables
        waves_broad = None
        abs_hduls = None

        # Generate broad + narrow profiles if requested
        if broad:
            print(f"Generating {n_spectra} broad+narrow profiles...")
            waves_broad = self.generate_broad_plus_narrow_batch(
                indices, line, vel_range, broad_to_narrow_ratio_range, broad_fwhm_range, use_multiprocessing, n_processes)

        # Generate broad + absorption + narrow profiles if requested
        if broad_abs:
            print(f"Generating {n_spectra} broad+absorption+narrow profiles...")

            # Process absorption profiles individually for unique random parameters
            all_abs_wav_mins = []
            all_abs_dwaves = []
            all_abs_npixels = []
            all_abs_fluxes = []
            all_abs_errs = []

            # Ensure spectra are loaded for parameter extraction
            self._ensure_loaded(line)
            narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
            if narrow_spec is None:
                raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

            for idx in indices:
                # Generate absorption profile (returns arrays)
                wav_min, dwave, npixels, flux, err = self.generate_broad_with_absorption_plus_narrow(
                    idx, line, vel_range, broad_to_narrow_ratio_range, broad_fwhm_range,
                    abs_ew_range, abs_vel_range, abs_fwhm_range)

                all_abs_wav_mins.append(wav_min)
                all_abs_dwaves.append(dwave)
                all_abs_npixels.append(npixels)
                all_abs_fluxes.append(flux)
                all_abs_errs.append(err)

            # Create parameters from the collected data (vectorized extraction - much faster)
            # Convert to list of numpy records (preserves field access like params['REDSHIFT'])
            all_abs_params = [row for row in narrow_spec['PARAMETERS'].data[indices]]

            # Convert errors to inverse variance
            all_abs_ivars = []
            for err in all_abs_errs:
                ivar = np.where(err > 0, 1.0 / (err ** 2), 0.0)
                all_abs_ivars.append(ivar)

            # Create multi-spectrum HDUList for absorption profiles
            abs_hduls = self._create_multi_spectrum_hdul(all_abs_params, all_abs_fluxes, all_abs_ivars,
                                                        'broad_with_absorption_plus_narrow', line, indices.tolist())
        else:
            # Ensure spectra are loaded for parameter extraction (needed for metadata even if not generating absorption)
            self._ensure_loaded(line)
            narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
            if narrow_spec is None:
                raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Save to FITS format if requested
        saved_files = {}
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            # Save broad + narrow profiles
            if broad and save_broad and hasattr(waves_broad, 'writeto'):
                if save_broad is True:
                    # Use default filename with parameters
                    seed_str = str(random_seed) if random_seed is not None else "none"
                    broad_ratio_str = f"{broad_to_narrow_ratio_range[0]}-{broad_to_narrow_ratio_range[1]}"
                    broad_fwhm_str = f"{broad_fwhm_range[0]}-{broad_fwhm_range[1]}"
                    broad_file = os.path.join(output_dir, f'broad_narrow_{line}_n{n_spectra}_ratio{broad_ratio_str}_fwhm{broad_fwhm_str}_seed{seed_str}.fits')
                else:
                    # Use provided filename
                    broad_file = os.path.join(output_dir, save_broad)
                waves_broad.writeto(broad_file, overwrite=True)
                saved_files['broad'] = broad_file
                print(f"Saved broad+narrow profiles to: {broad_file}")

            # Save broad + absorption + narrow profiles
            if broad_abs and save_absorption and hasattr(abs_hduls, 'writeto'):
                if save_absorption is True:
                    # Use default filename with parameters
                    seed_str = str(random_seed) if random_seed is not None else "none"
                    broad_ratio_str = f"{broad_to_narrow_ratio_range[0]}-{broad_to_narrow_ratio_range[1]}"
                    broad_fwhm_str = f"{broad_fwhm_range[0]}-{broad_fwhm_range[1]}"
                    abs_ew_str = f"{abs_ew_range[0]}-{abs_ew_range[1]}"
                    abs_vel_str = f"{abs_vel_range[0]}-{abs_vel_range[1]}"
                    abs_fwhm_str = f"{abs_fwhm_range[0]}-{abs_fwhm_range[1]}"
                    abs_file = os.path.join(output_dir, f'broad_absorption_narrow_{line}_n{n_spectra}_ratio{broad_ratio_str}_fwhm{broad_fwhm_str}_ew{abs_ew_str}_vel{abs_vel_str}_fwhm{abs_fwhm_str}_seed{seed_str}.fits')
                else:
                    # Use provided filename
                    abs_file = os.path.join(output_dir, save_absorption)
                abs_hduls.writeto(abs_file, overwrite=True)
                saved_files['absorption'] = abs_file
                print(f"Saved broad+absorption+narrow profiles to: {abs_file}")

        # Prepare results dictionary
        results = {}
        if broad:
            results['broad'] = waves_broad          # Single HDUList with multiple spectra
        if broad_abs:
            results['absorption'] = abs_hduls       # Single HDUList with multiple spectra
        results['metadata'] = {
            'selected_indices': indices.tolist(),
            'n_spectra_requested': n_spectra,
            'n_spectra_available': available_info['n_spectra'],
            'used_replacement': replace,
            'line': line,
            'vel_range': vel_range,
            'broad_to_narrow_ratio_range': broad_to_narrow_ratio_range,
            'broad_fwhm_range': broad_fwhm_range,
            'abs_ew_range': abs_ew_range,
            'abs_vel_range': abs_vel_range,
            'abs_fwhm_range': abs_fwhm_range,
            'random_seed': random_seed,
            'available_spectra': available_info['n_spectra'],
            'saved_files': saved_files
        }

        print(f"Successfully generated {n_spectra} broad and absorption profiles")
        return results

    def _save_profiles_to_fits(self, filename, params_data, fluxes, errs, line, profile_type):
        """
        Save generated profiles to FITS format matching input file structure.

        Parameters:
        -----------
        filename : str
            Output FITS filename
        params_data : ndarray
            Parameter data from original spectra
        fluxes : list
            List of flux arrays
        errs : list
            List of error arrays
        line : str
            Emission line type
        profile_type : str
            Type of profile ('broad_narrow' or 'broad_absorption_narrow')
        """
        from astropy.io import fits
        import numpy as np

        # Find the maximum length to pad all arrays to the same size
        max_length = max(len(flux) for flux in fluxes)

        # Convert fluxes and errors to proper format with padding
        flux_arrays = []
        ivar_arrays = []

        for i, (flux, err) in enumerate(zip(fluxes, errs)):
            # Use normalized flux directly
            flux_scaled = flux

            # Convert normalized errors to inverse variance
            ivar = np.where(err > 0, 1.0 / (err ** 2), 0.0)

            # Pad arrays to maximum length with zeros
            flux_padded = np.pad(flux_scaled, (0, max_length - len(flux_scaled)), 'constant', constant_values=0.0)
            ivar_padded = np.pad(ivar, (0, max_length - len(ivar)), 'constant', constant_values=0.0)

            flux_arrays.append(flux_padded)
            ivar_arrays.append(ivar_padded)

            # Update N_PIXELS in parameters to reflect actual data length (not padded length)
            params_data['N_PIXELS'][i] = len(flux)

        # Convert to numpy arrays
        flux_array = np.array(flux_arrays)
        ivar_array = np.array(ivar_arrays)

        # Create FITS HDUs
        primary_hdu = fits.PrimaryHDU()
        params_hdu = fits.BinTableHDU(data=params_data, name='PARAMETERS')
        flux_hdu = fits.ImageHDU(data=flux_array, name='FLUX_NOISY')
        ivar_hdu = fits.ImageHDU(data=ivar_array, name='IVAR')

        # Add metadata
        primary_hdu.header['PROFILE'] = profile_type
        primary_hdu.header['LINE'] = line
        primary_hdu.header['N_SPEC'] = len(fluxes)
        primary_hdu.header['MAX_PIX'] = max_length
        primary_hdu.header['CREATED'] = 'MockProfileGenerator'

        # Create HDU list and write
        hdul = fits.HDUList([primary_hdu, params_hdu, flux_hdu, ivar_hdu])
        hdul.writeto(filename, overwrite=True)
        hdul.close()

    def clear_cache(self):
        """Clear the processed spectra cache to free memory."""
        with self._cache_lock:
            self._processed_cache.clear()

    def get_cache_stats(self):
        """Get cache statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._processed_cache),
                'max_cache_size': self.max_cache_size,
                'cache_hit_ratio': None  # Could be implemented with counters
            }

    def get_performance_stats(self):
        """Get performance statistics."""
        return self._perf_stats.copy()


    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._perf_stats = {
            'total_spectra_processed': 0,
            'total_processing_time': 0.0,
            'avg_spectra_per_second': 0.0,
            'last_batch_size': 0,
            'last_batch_time': 0.0
        }

    def preload_spectra_info(self):
        """
        Preload basic spectra information for all available files.

        Returns:
        --------
        info : dict
            Dictionary with spectra information
        """
        info = {}
        if self._ha_info:
            info['Ha'] = self._ha_info
        if self._hb_info:
            info['Hb'] = self._hb_info
        return info

    def _convert_to_velocity_space(self, wave, flux, err, z, rest_wave, vel_range=2000):
        """
        Convert spectra to velocity space centered on emission line.

        Parameters:
        -----------
        wave : ndarray
            Wavelength array
        flux : ndarray
            Flux array
        err : ndarray
            Error array
        z : float
            Redshift
        rest_wave : float
            Rest wavelength of emission line
        vel_range : float
            Velocity range in km/s

        Returns:
        --------
        line_wave : ndarray
            Wavelength array in velocity range
        line_flux : ndarray
            Flux array in velocity range
        line_flux_err : ndarray
            Error array in velocity range
        vel_to_line : ndarray
            Velocity array relative to line center
        """
        # Convert to rest frame first (following notebook approach)
        rest_wave_obs = wave / (1 + z)
        rest_flux = flux * (1 + z)
        rest_flux_err = err * (1 + z)

        c = 299792.458  # speed of light in km/s
        vel_to_line = (rest_wave_obs - rest_wave) / rest_wave * c

        flag = np.abs(vel_to_line) < vel_range
        line_wave = wave[flag]
        line_flux = rest_flux[flag]  # Use rest-frame flux
        line_flux_err = rest_flux_err[flag]  # Use rest-frame error

        return line_wave, line_flux, line_flux_err, vel_to_line[flag]

    def _convert_to_velocity_space_batch(self, waves, fluxes, errs, zs, rest_wave, vel_range=2000):
        """
        Vectorized version of _convert_to_velocity_space for multiple spectra.

        Parameters:
        -----------
        waves : list of ndarray
            List of wavelength arrays
        fluxes : list of ndarray
            List of flux arrays
        errs : list of ndarray
            List of error arrays
        zs : ndarray
            Array of redshifts
        rest_wave : float
            Rest wavelength of emission line
        vel_range : float
            Velocity range in km/s

        Returns:
        --------
        line_waves : list of ndarray
            List of wavelength arrays in velocity range
        line_fluxes : list of ndarray
            List of flux arrays in velocity range
        line_flux_errs : list of ndarray
            List of error arrays in velocity range
        vel_to_lines : list of ndarray
            List of velocity arrays relative to line center
        """
        line_waves, line_fluxes, line_flux_errs, vel_to_lines = [], [], [], []

        for wave, flux, err, z in zip(waves, fluxes, errs, zs):
            result = self._convert_to_velocity_space(wave, flux, err, z, rest_wave, vel_range)
            line_waves.append(result[0])
            line_fluxes.append(result[1])
            line_flux_errs.append(result[2])
            vel_to_lines.append(result[3])

        return line_waves, line_fluxes, line_flux_errs, vel_to_lines

    def _normalize_spectra_batch(self, fluxes, errs, vel_to_lines, norm_range=200):
        """
        Vectorized version of _normalize_spectra for multiple spectra.

        Parameters:
        -----------
        fluxes : list of ndarray
            List of flux arrays
        errs : list of ndarray
            List of error arrays
        vel_to_lines : list of ndarray
            List of velocity arrays
        norm_range : float
            Normalization range in km/s

        Returns:
        --------
        norm_fluxes : list of ndarray
            List of normalized flux arrays
        norm_flux_errs : list of ndarray
            List of normalized error arrays
        peak_fluxes : ndarray
            Array of peak flux values
        """
        norm_fluxes, norm_flux_errs, peak_fluxes = [], [], []

        for flux, err, vel_to_line in zip(fluxes, errs, vel_to_lines):
            norm_flux, norm_flux_err, peak_flux = self._normalize_spectra(flux, err, vel_to_line, norm_range)
            norm_fluxes.append(norm_flux)
            norm_flux_errs.append(norm_flux_err)
            peak_fluxes.append(peak_flux)

        return norm_fluxes, norm_flux_errs, np.array(peak_fluxes)

    def _normalize_spectra(self, flux, err, vel_to_line, norm_range=200):
        """
        Normalize spectra by peak flux in central region.

        Parameters:
        -----------
        flux : ndarray
            Flux array
        err : ndarray
            Error array
        vel_to_line : ndarray
            Velocity array
        norm_range : float
            Velocity range for normalization in km/s

        Returns:
        --------
        norm_flux : ndarray
            Normalized flux
        norm_flux_err : ndarray
            Normalized error
        peak_flux : float
            Peak flux used for normalization
        """
        norm_range_flag = np.abs(vel_to_line) < norm_range

        # Use flux in the normalization range, but if empty, use the overall peak
        if np.any(norm_range_flag):
            peak_flux = np.nanmax(flux[norm_range_flag])
        else:
            peak_flux = np.nanmax(flux)

        # Ensure peak_flux is not zero or nan
        if peak_flux <= 0 or np.isnan(peak_flux):
            peak_flux = 1.0

        norm_flux = flux / peak_flux
        norm_flux_err = err / peak_flux

        return norm_flux, norm_flux_err, peak_flux

    def _process_spectra_batch_parallel(self, indices: List[int], line: str, n_processes: Optional[int] = None, vel_range: float = DEFAULT_VEL_RANGE) -> Tuple:
        """
        Process a batch of spectra using multiprocessing for improved performance.

        Parameters:
        -----------
        indices : List[int]
            Indices of spectra to process
        line : str
            Emission line ('Ha' or 'Hb')
        n_processes : int or None
            Number of processes to use. If None, uses CPU count.

        Returns:
        --------
        Tuple of processed data arrays
        """
        if n_processes is None:
            n_processes = min(mp.cpu_count(), len(indices))

        if n_processes == 1:
            print(f"🔄 Using sequential processing (single process)")
            # For single process, use sequential processing
            return self._process_batch_spectra_data(indices, line)

        import time
        start_time = time.time()

        print(f"🚀 Starting multiprocessing: {n_processes} processes, {len(indices)} spectra")

        # Load data in main process with minimal memory usage
        print(f"📂 Loading FITS data efficiently...")
        self._ensure_loaded(line)
        narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
        if narrow_spec is None:
            raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Get rest wavelength
        rest_wave = REST_WAVELENGTHS[line]

        # Extract only the data for requested indices - no unnecessary copying
        print(f"📊 Extracting data for {len(indices)} spectra...")
        indices_array = np.array(indices)
        flux_data = narrow_spec['FLUX_NOISY'].data[indices_array]  # Direct indexing
        ivar_data = narrow_spec['IVAR'].data[indices_array]
        params_data = narrow_spec['PARAMETERS'].data[indices_array]

        # Pre-compute all wavelength arrays to avoid reconstruction in workers
        print(f"🔢 Pre-computing wavelength arrays...")
        raw_data = []
        for i, idx in enumerate(indices):
            params = params_data[i]
            z = float(params['REDSHIFT'])

            # Pre-compute wavelength array once
            wav_min = float(params['WAV_MIN'])
            dwave = float(params['DWAVE'])
            n_pix = int(params['N_PIXELS'])
            wave = wav_min + np.arange(n_pix, dtype=np.float64) * dwave

            # Get flux and error data
            flux = flux_data[i][:n_pix].astype(np.float64)
            ivar_truncated = ivar_data[i][:n_pix]
            err = np.where(ivar_truncated > 0, 1.0 / np.sqrt(ivar_truncated), np.inf).astype(np.float64)

            # Pass pre-computed arrays: (wave, flux, err, z)
            raw_data.append((wave, flux, err, z))

        # Split data into optimally-sized chunks
        chunk_size = max(1, len(raw_data) // n_processes)
        data_chunks = [raw_data[i:i + chunk_size] for i in range(0, len(raw_data), chunk_size)]
        print(f"📦 Split into {len(data_chunks)} chunks ({chunk_size} spectra per chunk)")

        # Launch parallel processing
        print(f"⚡ Starting parallel processing...")
        chunk_results = {}
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            future_to_index = {}
            for i, chunk in enumerate(data_chunks):
                future = executor.submit(_process_spectra_chunk_worker, chunk, rest_wave, vel_range)
                future_to_index[future] = i

            # Collect results with progress tracking
            completed_chunks = 0
            for future in as_completed(future_to_index):
                try:
                    chunk_idx = future_to_index[future]
                    result = future.result()
                    chunk_results[chunk_idx] = result
                    completed_chunks += 1
                    print(f"✅ Completed chunk {completed_chunks}/{len(data_chunks)} (index {chunk_idx})")
                except Exception as exc:
                    raise RuntimeError(f'Chunk processing failed: {exc}')

        print(f"🎯 All {len(data_chunks)} chunks completed")

        # Combine results in original order using pre-allocated arrays for speed
        if not chunk_results:
            return [], [], [], []

        print(f"🔄 Fast recombination...")
        total_spectra = len(indices)
        all_waves = [None] * total_spectra
        all_fluxes = [None] * total_spectra
        all_errs = [None] * total_spectra
        all_zs = np.empty(total_spectra, dtype=np.float64)

        # Fill results in correct order
        spectrum_idx = 0
        for chunk_idx in range(len(data_chunks)):
            if chunk_idx in chunk_results:
                waves, fluxes, errs, zs = chunk_results[chunk_idx]
                chunk_size_actual = len(waves)

                for j in range(chunk_size_actual):
                    all_waves[spectrum_idx + j] = waves[j]
                    all_fluxes[spectrum_idx + j] = fluxes[j]
                    all_errs[spectrum_idx + j] = errs[j]
                    all_zs[spectrum_idx + j] = zs[j]

                spectrum_idx += chunk_size_actual

        end_time = time.time()
        processing_time = end_time - start_time

        # Update performance statistics
        self._perf_stats['total_spectra_processed'] += len(indices)
        self._perf_stats['total_processing_time'] += processing_time
        self._perf_stats['avg_spectra_per_second'] = self._perf_stats['total_spectra_processed'] / self._perf_stats['total_processing_time']
        self._perf_stats['last_batch_size'] = len(indices)
        self._perf_stats['last_batch_time'] = processing_time

        print(f"✨ Multiprocessing complete: {len(all_fluxes)} spectra processed in {processing_time:.2f}s ({len(all_fluxes)/processing_time:.1f} spectra/s)")
        return all_waves, all_fluxes, all_errs, all_zs

    


    def _generate_broad_component(self, wave, z, rest_wave, broad_to_narrow_ratio, broad_fwhm, norm_narrow_val=1.0):
        """
        Generate broad emission component.

        Parameters:
        -----------
        wave : ndarray
            Wavelength array
        z : float
            Redshift
        rest_wave : float
            Rest wavelength of emission line
        broad_to_narrow_ratio : float
            Ratio of broad to narrow component flux
        broad_fwhm : float
            FWHM of broad component in km/s
        norm_narrow_val : float
            Normalized integrated flux of narrow component (default: 1.0)

        Returns:
        --------
        broad_gaussian : ndarray
            Broad component flux
        """
        c = 299792.458  # speed of light in km/s
        norm_broad_val = norm_narrow_val * broad_to_narrow_ratio

        # Compute sigma in wavelength space
        broad_sigma_lambda = (broad_fwhm / (2 * np.sqrt(2 * np.log(2)))) / c * rest_wave

        # Area under Gaussian = amplitude * sigma * sqrt(2*pi)
        amplitude_lambda = norm_broad_val / (broad_sigma_lambda * np.sqrt(2 * np.pi))

        broad_gaussian = amplitude_lambda * np.exp(-0.5 * ((wave - rest_wave * (1+z)) / broad_sigma_lambda)**2)

        return broad_gaussian

    def _generate_absorption_component(self, wave, z, rest_wave, abs_ew, abs_vel, abs_fwhm):
        """
        Generate absorption component.

        Parameters:
        -----------
        wave : ndarray
            Wavelength array
        z : float
            Redshift
        rest_wave : float
            Rest wavelength of emission line
        abs_ew : float
            Equivalent width of absorption
        abs_vel : float
            Velocity offset of absorption from line center
        abs_fwhm : float
            FWHM of absorption feature

        Returns:
        --------
        abs_gaussian : ndarray
            Absorption component (values between 0 and 1)
        """
        c = 299792.458  # speed of light in km/s

        abs_center = rest_wave + abs_vel * rest_wave / c
        abs_center = abs_center * (1+z)
        abs_sigma = (abs_fwhm/c * abs_center) / (2 * np.sqrt(2 * np.log(2)))
        abs_amplitude = abs_ew / (abs_sigma * np.sqrt(2 * np.pi))
        abs_gaussian = abs_amplitude * np.exp(-0.5 * ((wave - abs_center) / abs_sigma)**2)
        abs_gaussian[abs_gaussian > 1] = 1  # Cap at 1

        return abs_gaussian

    def _get_rest_wavelength(self, line):
        """Get rest wavelength for emission line."""
        if line == 'Ha':
            return 6564.614
        elif line == 'Hb':
            return 4862.683
        else:
            raise ValueError("Line must be 'Ha' or 'Hb'")


    def generate_broad_plus_narrow(self, idx: int, line: str = 'Ha', vel_range: float = DEFAULT_VEL_RANGE,
                                  broad_to_narrow_ratio_range: Tuple[float, float] = DEFAULT_BROAD_RATIO_RANGE,
                                  broad_fwhm_range: Tuple[float, float] = DEFAULT_BROAD_FWHM_RANGE,
                                  use_cache: bool = True) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        """
        Generate broad + narrow mock line profile.

        Parameters:
        -----------
        idx : int
            Index of spectrum to use
        line : str
            Emission line ('Ha' or 'Hb')
        vel_range : float
            Velocity range in km/s
        broad_to_narrow_ratio_range : tuple
            Range for random broad/narrow ratio
        broad_fwhm_range : tuple
            Range for random broad FWHM in km/s
        use_cache : bool
            Whether to use cached processed spectra

        Returns:
        --------
        wav_min : float
            Minimum wavelength
        dwave : float
            Wavelength spacing
        npixels : int
            Number of pixels
        flux : ndarray
            Normalized flux array
        err : ndarray
            Normalized error array
        """
        rest_wave = self._get_rest_wavelength(line)

        # Process narrow component data
        wave, flux, err, z = self._process_spectra_data(idx, line, use_cache)

        # Convert to velocity space
        line_wave, line_flux, line_flux_err, vel_to_line = self._convert_to_velocity_space(
            wave, flux, err, z, rest_wave, vel_range)

        # Normalize
        norm_flux, norm_flux_err, _ = self._normalize_spectra(line_flux, line_flux_err, vel_to_line)

        # Calculate normalized narrow component integrated flux
        norm_range_flag = np.abs(vel_to_line) < DEFAULT_NORM_RANGE  # normalization range
        norm_narrow_val = np.trapz(norm_flux[norm_range_flag & ~np.isnan(norm_flux)],
                                   line_wave[norm_range_flag & ~np.isnan(norm_flux)])

        rest_wave = self._get_rest_wavelength(line)

        # Ensure spectra are loaded and get narrow_spec
        self._ensure_loaded(line)
        narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
        if narrow_spec is None:
            raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Get redshift from cached data
        cache_key = self._get_cache_key(idx, line)
        with self._cache_lock:
            if cache_key in self._processed_cache:
                _, _, _, z = self._processed_cache[cache_key]
            else:
                # Fallback: get redshift directly
                params = narrow_spec['PARAMETERS'].data[idx]
                z = params['REDSHIFT']

        # Generate random broad component parameters
        broad_to_narrow_ratio = np.random.uniform(*broad_to_narrow_ratio_range)
        broad_fwhm = np.random.uniform(*broad_fwhm_range)

        # Generate broad component using the same wavelength range as narrow component
        broad_gaussian = self._generate_broad_component(line_wave, z, rest_wave, broad_to_narrow_ratio, broad_fwhm, norm_narrow_val)

        # Combine components
        composite = norm_flux + broad_gaussian

        # Renormalize
        rest_wave_range = line_wave / (1 + z)
        norm_range_flag = (rest_wave_range > rest_wave - 2) & (rest_wave_range < rest_wave + 2)
        if np.any(norm_range_flag):
            renorm = np.nanmax(composite[norm_range_flag])
        else:
            renorm = np.nanmax(composite)
        if renorm > 0:
            norm_composite = composite / renorm
            norm_composite_err = norm_flux_err / renorm
        else:
            norm_composite = composite
            norm_composite_err = norm_flux_err

        # Get original parameters for wavelength information
        params = narrow_spec['PARAMETERS'].data[idx]

        # Return arrays: wav_min, dwave, npixels, flux, err
        return params['WAV_MIN'], params['DWAVE'], params['N_PIXELS'], norm_composite, norm_composite_err

    def generate_broad_with_absorption_plus_narrow(self, idx: int, line: str = 'Ha', vel_range: float = DEFAULT_VEL_RANGE,
                                                 broad_to_narrow_ratio_range: Tuple[float, float] = DEFAULT_BROAD_RATIO_RANGE,
                                                 broad_fwhm_range: Tuple[float, float] = DEFAULT_BROAD_FWHM_RANGE,
                                                 abs_ew_range: Tuple[float, float] = DEFAULT_ABS_EW_RANGE,
                                                 abs_vel_range: Tuple[float, float] = DEFAULT_ABS_VEL_RANGE,
                                                 abs_fwhm_range: Tuple[float, float] = DEFAULT_ABS_FWHM_RANGE,
                                                 use_cache: bool = True) -> Tuple[float, float, int, np.ndarray, np.ndarray]:
        """
        Generate broad * (1-abs) + narrow mock line profile.

        Parameters:
        -----------
        idx : int
            Index of spectrum to use
        line : str
            Emission line ('Ha' or 'Hb')
        vel_range : float
            Velocity range in km/s
        broad_to_narrow_ratio_range : tuple
            Range for random broad/narrow ratio
        broad_fwhm_range : tuple
            Range for random broad FWHM in km/s
        abs_ew_range : tuple
            Range for random absorption equivalent width
        abs_vel_range : tuple
            Range for random absorption velocity offset
        abs_fwhm_range : tuple
            Range for random absorption FWHM
        use_cache : bool
            Whether to use cached processed spectra

        Returns:
        --------
        wav_min : float
            Minimum wavelength
        dwave : float
            Wavelength spacing
        npixels : int
            Number of pixels
        flux : ndarray
            Combined normalized flux array
        err : ndarray
            Normalized error array
        """
        rest_wave = self._get_rest_wavelength(line)

        # Process narrow component data
        wave, flux, err, z = self._process_spectra_data(idx, line, use_cache)

        # Convert to velocity space
        line_wave, line_flux, line_flux_err, vel_to_line = self._convert_to_velocity_space(
            wave, flux, err, z, rest_wave, vel_range)

        # Normalize
        norm_flux, norm_flux_err, _ = self._normalize_spectra(line_flux, line_flux_err, vel_to_line)

        # Calculate normalized narrow component integrated flux
        norm_range_flag = np.abs(vel_to_line) < DEFAULT_NORM_RANGE  # normalization range
        norm_narrow_val = np.trapz(norm_flux[norm_range_flag & ~np.isnan(norm_flux)],
                                   line_wave[norm_range_flag & ~np.isnan(norm_flux)])

        # Ensure spectra are loaded and get narrow_spec
        self._ensure_loaded(line)
        narrow_spec = self.narrow_spec_ha if line == 'Ha' else self.narrow_spec_hb
        if narrow_spec is None:
            raise ValueError(f"No {line} spectra data loaded. Please provide the corresponding FITS file.")

        # Get redshift from cached data
        cache_key = self._get_cache_key(idx, line)
        with self._cache_lock:
            if cache_key in self._processed_cache:
                _, _, _, z = self._processed_cache[cache_key]
            else:
                # Fallback: get redshift directly
                params = narrow_spec['PARAMETERS'].data[idx]
                z = params['REDSHIFT']

        # Generate random parameters
        broad_to_narrow_ratio = np.random.uniform(*broad_to_narrow_ratio_range)
        broad_fwhm = np.random.uniform(*broad_fwhm_range)
        abs_ew = np.random.uniform(*abs_ew_range)
        abs_vel = np.random.uniform(*abs_vel_range)
        abs_fwhm = np.random.uniform(*abs_fwhm_range)

        # Generate components using the same wavelength range as narrow component
        broad_gaussian = self._generate_broad_component(line_wave, z, rest_wave, broad_to_narrow_ratio, broad_fwhm, norm_narrow_val)
        abs_gaussian = self._generate_absorption_component(line_wave, z, rest_wave, abs_ew, abs_vel, abs_fwhm)

        # Apply absorption to broad component: broad * (1 - abs)
        broad_with_abs = broad_gaussian * (1 - abs_gaussian)

        # Combine with narrow
        composite = norm_flux + broad_with_abs

        # Renormalize
        rest_wave_range = line_wave / (1 + z)
        norm_range_flag = (rest_wave_range > rest_wave - 2) & (rest_wave_range < rest_wave + 2)
        if np.any(norm_range_flag):
            renorm = np.nanmax(composite[norm_range_flag])
        else:
            renorm = np.nanmax(composite)
        if renorm > 0:
            norm_composite = composite / renorm
            norm_composite_err = norm_flux_err / renorm
        else:
            norm_composite = composite
            norm_composite_err = norm_flux_err

        # Get original parameters for wavelength information
        params = narrow_spec['PARAMETERS'].data[idx]

        # Return arrays: wav_min, dwave, npixels, flux, err
        return params['WAV_MIN'], params['DWAVE'], params['N_PIXELS'], norm_composite, norm_composite_err
 
 

def _process_spectra_chunk_worker(chunk_data: List[Tuple], rest_wave: float, vel_range: float) -> Tuple:
    """
    Worker function using pre-computed wavelength arrays.
    Optimized for minimal memory allocation and maximum speed.

    Parameters:
    -----------
    chunk_data : List[Tuple]
        List of (wave_array, flux_array, err_array, z) tuples - wave_array is pre-computed
    rest_wave : float
        Rest wavelength of emission line
    vel_range : float
        Velocity range in km/s

    Returns:
    --------
    Tuple of processed data arrays for this chunk
    """
    print(f"🔧 Worker processing {len(chunk_data)} spectra")

    # Pre-allocate result lists with known size
    n_spectra = len(chunk_data)
    processed_waves = [None] * n_spectra
    processed_fluxes = [None] * n_spectra
    processed_errs = [None] * n_spectra
    processed_zs = np.empty(n_spectra, dtype=np.float64)

    c = 299792.458  # speed of light in km/s

    for i, (wave, flux, err, z) in enumerate(chunk_data):
        # Convert to velocity space (vectorized)
        rest_wave_obs = wave / (1 + z)
        rest_flux = flux * (1 + z)
        rest_flux_err = err * (1 + z)

        vel_to_line = (rest_wave_obs - rest_wave) / rest_wave * c
        flag = np.abs(vel_to_line) < vel_range

        # Extract only the data we need
        line_wave = wave[flag]
        line_flux = rest_flux[flag]
        line_flux_err = rest_flux_err[flag]
        vel_mask = vel_to_line[flag]

        # Normalize using vectorized operations
        norm_range_flag = np.abs(vel_mask) < DEFAULT_NORM_RANGE  # normalization range
        if np.any(norm_range_flag):
            peak_flux = np.nanmax(line_flux[norm_range_flag])
        else:
            peak_flux = np.nanmax(line_flux)

        if peak_flux <= 0 or np.isnan(peak_flux):
            peak_flux = 1.0

        # Store results directly (no intermediate variables)
        processed_waves[i] = line_wave
        processed_fluxes[i] = line_flux / peak_flux
        processed_errs[i] = line_flux_err / peak_flux
        processed_zs[i] = z

    return processed_waves, processed_fluxes, processed_errs, processed_zs


def _generate_broad_plus_narrow_chunk_worker(chunk_data: List[Tuple], rest_wave: float,
                                                        broad_to_narrow_ratio_range: Tuple[float, float],
                                                        broad_fwhm_range: Tuple[float, float],
                                                        random_seed: int = 42) -> Tuple:
    """
    Broad+narrow generation with minimal memory allocation.
    Processes complete mock generation pipeline in one pass.

    Parameters:
    -----------
    chunk_data : List[Tuple]
        List of (wave_array, flux_array, err_array, z, idx) tuples
    rest_wave : float
        Rest wavelength of emission line
    broad_to_narrow_ratio_range : Tuple[float, float]
        Range for random broad/narrow ratio
    broad_fwhm_range : Tuple[float, float]
        Range for random broad FWHM in km/s
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    Tuple of processed flux, error, and parameter arrays
    """
    print(f"🔧 Broad+narrow worker processing {len(chunk_data)} spectra")

    # Set random seed and pre-allocate results
    np.random.seed(random_seed)
    n_spectra = len(chunk_data)
    processed_fluxes = [None] * n_spectra
    processed_errs = [None] * n_spectra
    processed_params = [None] * n_spectra

    c = 299792.458  # speed of light in km/s

    for i, (wave, flux, err, z, idx) in enumerate(chunk_data):
        # Data processing (velocity space conversion)
        rest_wave_obs = wave / (1 + z)
        rest_flux = flux * (1 + z)
        rest_flux_err = err * (1 + z)

        vel_to_line = (rest_wave_obs - rest_wave) / rest_wave * c
        flag = np.abs(vel_to_line) < 2000  # Default vel_range

        line_wave = wave[flag]
        line_flux = rest_flux[flag]
        line_flux_err = rest_flux_err[flag]
        vel_mask = vel_to_line[flag]

        # Normalization
        norm_range_flag = np.abs(vel_mask) < DEFAULT_NORM_RANGE
        peak_flux = np.nanmax(line_flux[norm_range_flag]) if np.any(norm_range_flag) else np.nanmax(line_flux)
        if peak_flux <= 0 or np.isnan(peak_flux):
            peak_flux = 1.0

        norm_flux = line_flux / peak_flux
        norm_flux_err = line_flux_err / peak_flux
        norm_narrow_val = np.trapz(norm_flux[norm_range_flag & ~np.isnan(norm_flux)], line_wave[norm_range_flag & ~np.isnan(norm_flux)])

        # Generate broad component parameters
        broad_to_narrow_ratio = np.random.uniform(*broad_to_narrow_ratio_range)
        broad_fwhm = np.random.uniform(*broad_fwhm_range)

        # Generate broad component (inline for speed)
        norm_broad_val = broad_to_narrow_ratio * norm_narrow_val
        broad_sigma_lambda = (broad_fwhm / (2 * np.sqrt(2 * np.log(2)))) / c * rest_wave
        amplitude_lambda = norm_broad_val / (broad_sigma_lambda * np.sqrt(2 * np.pi))
        broad_gaussian = amplitude_lambda * np.exp(-0.5 * ((line_wave - rest_wave * (1+z)) / broad_sigma_lambda)**2)

        # Combine and renormalize
        composite = norm_flux + broad_gaussian
        rest_wave_range = line_wave / (1 + z)
        renorm_range = (rest_wave_range > rest_wave - 2) & (rest_wave_range < rest_wave + 2)

        renorm = np.nanmax(composite[renorm_range]) if np.any(renorm_range) else np.nanmax(composite)
        if renorm > 0:
            norm_composite = composite / renorm
            norm_composite_err = norm_flux_err / renorm
        else:
            norm_composite = composite
            norm_composite_err = norm_flux_err

        # Convert to inverse variance
        ivar = np.where(norm_composite_err > 0, 1.0 / (norm_composite_err ** 2), 0.0)

        # Store results
        processed_fluxes[i] = norm_composite
        processed_errs[i] = ivar
        processed_params[i] = idx

    return processed_fluxes, processed_errs, processed_params
