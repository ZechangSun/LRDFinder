"""
DESI Mock Emission Line Profile Generation Script

This script generates comprehensive mock spectroscopic datasets for training machine learning
models to classify DESI galaxies. It creates three types of emission line profiles for both
H-alpha and H-beta lines:

1. Narrow-only profiles: Clean emission lines from input templates
2. Broad + narrow profiles: AGN-like spectra with broad components
3. Broad + absorption + narrow profile
"""

# ============================================================================
# Dependencies and Configuration
# ============================================================================

from astropy.io import fits
import os
import numpy as np

# Matplotlib configuration (for potential plotting/analysis)
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["ytick.left"] = True
plt.rcParams['font.size'] = 15

# Suppress warnings for cleaner batch processing output
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Main Generation Logic
# ============================================================================

from LRDFinder.mock import MockProfileGenerator
import numpy as np

# Define data directory for input templates and output files
mock_data_dir = '/home/lxj/Project/DESI_LRD_ML/data_mock'

# Initialize mock profile generator with narrow emission line templates
# These template spectra serve as the "narrow" component for all generated profiles
generator = MockProfileGenerator(
    halpha_file=os.path.join(mock_data_dir, 'narrow_spectra_Ha.fits'),
    hbeta_file=os.path.join(mock_data_dir, 'narrow_spectra_Hb.fits')
)

# ============================================================================
# Generate H-alpha Mock Profiles (100k spectra total)
# ============================================================================

print("🚀 Generating H-alpha mock emission line profiles...")
Ha_results = generator.generate_all_profiles_batch(
    n_spectra=10000,                    # Total spectra for H-alpha
    line='Ha',                          # H-alpha emission line (6564.614 Å)
    broad_to_narrow_ratio_range=(2, 10), # Broad component strength relative to narrow
    abs_ew_range=(2, 10),               # Absorption equivalent width range (Å)
    output_dir=mock_data_dir,           # Save FITS files to this directory
    save_broad=True,                    # Generate broad+narrow composite profiles
    save_absorption=True,               # Generate full profiles with absorption
    random_seed=2024,                   # Ensures reproducible results
    use_multiprocessing=True,           # Enable parallel processing for speed
    n_processes=128                     # Max CPU cores (adjust for your system)
)

# ============================================================================
# Generate H-beta Mock Profiles (100k spectra total)
# ============================================================================

print("🚀 Generating H-beta mock emission line profiles...")
Hb_results = generator.generate_all_profiles_batch(
    n_spectra=10000,                    # Total spectra for H-beta
    line='Hb',                          # H-beta emission line (4862.683 Å)
    broad_to_narrow_ratio_range=(2, 10), # Broad component strength relative to narrow
    abs_ew_range=(2, 10),               # Absorption equivalent width range (Å)
    output_dir=mock_data_dir,           # Save FITS files to this directory
    save_broad=True,                    # Generate broad+narrow composite profiles
    save_absorption=True,               # Generate full profiles with absorption
    random_seed=2024,                   # Same seed for consistent results
    use_multiprocessing=True,           # Enable parallel processing
    n_processes=128                     # Max CPU cores (adjust for your system)
)

 