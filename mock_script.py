"""
DESI Mock Emission Line Profile Generation Script

Generates mock spectroscopic datasets for training ML models on DESI galaxy classification.
Creates emission line profiles: broad+narrow, and broad+absorption+narrow.
"""
from astropy.io import fits
import os
import numpy as np
 
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

 
import warnings
warnings.filterwarnings('ignore')

from LRDFinder.mock import MockProfileGenerator

# Set data directory paths
mock_data_dir = '/home/lxj/Project/DESI_LRD_ML/data_mock'

# Create generator with H-alpha and H-beta narrow line templates
# These templates provide the base narrow emission lines for all mock profiles
generator = MockProfileGenerator(
    halpha_file=os.path.join(mock_data_dir, 'narrow_spectra_Ha.fits'),
    hbeta_file=os.path.join(mock_data_dir, 'narrow_spectra_Hb.fits')
)

# ============================================================================
# Generate H-alpha mock profiles  
# ============================================================================

print("🚀 Generating H-alpha mock emission line profiles...")

# Generate both broad+narrow and broad+absorption+narrow profiles for H-alpha
Ha_results = generator.generate_all_profiles_batch(
    n_spectra=10000,                     
    line='Ha',                           
    broad_to_narrow_ratio_range=(3, 10), # Broad component 4-15x stronger than narrow (integrated flux within +-1000 km/s)
    abs_ew_range=(2, 10),               # Absorption equivalent width 2-10 Å
    output_dir=mock_data_dir,           # Save outputs to data directory
    save_broad='mock_Ha_broad_only.fits',    # Filename for broad+narrow profiles
    save_absorption='mock_Ha_broad_absorption.fits',    # Filename for profiles with absorption
    random_seed=2024,                   # Fixed seed for reproducible results
    use_multiprocessing=True,           # Use parallel processing
    n_processes=128                     # Use all available CPU cores
)

# ============================================================================
# Generate H-beta mock profiles
# ============================================================================

print("🚀 Generating H-beta mock emission line profiles...")

# Generate both broad+narrow and broad+absorption+narrow profiles for H-beta
Hb_results = generator.generate_all_profiles_batch(
    n_spectra=10000,                    
    line='Hb',                           
    broad_to_narrow_ratio_range=(3, 10), 
    abs_ew_range=(2, 10),               
    output_dir=mock_data_dir,           
    save_broad='mock_Hb_broad_only.fits',    
    save_absorption='mock_Hb_broad_absorption.fits',    
    random_seed=2024,                   
    use_multiprocessing=True,           
    n_processes=128                     
)
# ============================================================================
# Generate additional H-alpha + strong NII profiles with only broad components
# ============================================================================

generator = MockProfileGenerator(
    halpha_file=os.path.join(mock_data_dir, 'narrow_spectra_Ha.fits'),
    hbeta_file=None  # Skip H-beta to save memory
)

Ha_narrow_results = generator.generate_all_profiles_batch(
    n_spectra=10000,                    
    line='Ha',                           
    broad=True, broad_abs=False,       # Disable broad+absorption mock
    broad_to_narrow_ratio_range=(3, 10),
    output_dir=mock_data_dir,           
    save_broad='mock_Ha_strongNII_broad_only.fits',     
    random_seed=2024,                   
    use_multiprocessing=True,           
    n_processes=128                     
)
