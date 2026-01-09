import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from astropy.io import fits

# Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

from LRDFinder.mock import MockProfileGenerator


def generate_and_visualize_large_batch():
    """
    Generate 100 spectra using generate_all_profiles_batch and visualize them
    """
    print("=== Generating and Visualizing 100 Spectra ===")
    
    # Specify the data files
    data_dir = '/Users/bytedance/Desktop/code/LRDFinder/data/'
    ha_file = os.path.join(data_dir, 'narrow_spectra_Ha.fits')
    hb_file = os.path.join(data_dir, 'narrow_spectra_Hb.fits')
    
    # Create a dedicated folder for visualizations
    vis_dir = '/Users/bytedance/Desktop/code/LRDFinder/tests/large_batch_visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📁 Creating visualization folder: {vis_dir}")
    
    try:
        # Initialize the generator with both Ha and Hb files
        print("🔧 Initializing MockProfileGenerator...")
        generator = MockProfileGenerator(halpha_file=ha_file, hbeta_file=hb_file)
        
        # Generate 100 spectra for Ha line with and without absorption
        n_spectra = 100
        print(f"🎯 Generating {n_spectra} Ha spectra with and without absorption...")
        
        # Generate the spectra with generate_all_profiles_batch
        results = generator.generate_all_profiles_batch(
            n_spectra=n_spectra,
            line='Ha',
            broad=True,
            broad_abs=True,
            broad_to_narrow_ratio_range=(0.2, 1),
            abs_ew_range=(2, 10),
            abs_vel_range=(-300, 100),
            abs_fwhm_range=(80, 250),
            random_seed=42,
            use_multiprocessing=True,
            n_processes=4
        )
        
        print(f"✅ Spectra generated successfully!")
        print(f"📋 Result keys: {list(results.keys())}")
        
        # Get the rest wavelength for Ha
        rest_wave = 6564.614  # H-alpha rest wavelength
        
        # Visualize both broad-only and absorption spectra
        if 'broad' in results:
            print(f"📈 Visualizing {n_spectra} broad+narrow spectra...")
            visualize_batch_spectra(results['broad'], results['metadata'], vis_dir, "broad_only")
        
        if 'absorption' in results:
            print(f"📈 Visualizing {n_spectra} broad+absorption+narrow spectra...")
            visualize_batch_spectra(results['absorption'], results['metadata'], vis_dir, "with_absorption")
        
        print("\n=== All Visualizations Completed! ===")
        print(f"📊 Spectra generated: {n_spectra} for Ha line")
        print(f"📁 Visualizations saved to: {vis_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_batch_spectra(result_hdul, metadata, output_dir, prefix="spectra"):
    """
    Visualize a batch of spectra from an HDUList
    """
    # Get the data
    params = result_hdul['PARAMETERS'].data
    fluxes = result_hdul['FLUX_NOISY'].data
    ivars = result_hdul['IVAR'].data
    
    # Calculate errors
    errors = np.sqrt(1.0 / ivars)
    errors[ivars == 0] = 0  # Set error to 0 where IVAR is 0
    
    # Get rest wavelength from metadata
    rest_wave = 6564.614  # Ha rest wavelength
    c = 299792.458  # speed of light in km/s
    
    n_spectra = len(params)
    spectra_per_figure = 10
    n_figures = (n_spectra + spectra_per_figure - 1) // spectra_per_figure
    
    print(f"   Creating {n_figures} figures with {spectra_per_figure} spectra each...")
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * spectra_per_figure
        end_idx = min((fig_idx + 1) * spectra_per_figure, n_spectra)
        current_spectra = end_idx - start_idx
        
        # Create figure
        fig, axes = plt.subplots(current_spectra, 1, figsize=(12, 2*current_spectra), sharex=True)
        if current_spectra == 1:
            axes = [axes]
        
        for i in range(current_spectra):
            spectrum_idx = start_idx + i
            flux = fluxes[spectrum_idx]
            error = errors[spectrum_idx]
            param = params[spectrum_idx]
            
            # Calculate wavelength and velocity
            wave_min = param['WAV_MIN']
            dwave = param['DWAVE']
            z = param['REDSHIFT']
            actual_length = len(flux)
            
            wave = wave_min + np.arange(actual_length) * dwave
            rest_wave_obs = wave / (1 + z)
            vel_to_line = (rest_wave_obs - rest_wave) / rest_wave * c
            
            # Mask zero values
            mask = (flux != 0.) & (error != 0.)
            valid_vel = vel_to_line
            valid_flux = flux
            valid_error = error
            
            # Plot
            ax = axes[i]
            ax.plot(valid_vel, valid_flux, linewidth=1.2, color='darkgreen')
            ax.plot(valid_vel, valid_error, linewidth=1.2, color='red')
            ax.fill_between(valid_vel, valid_flux - valid_error, valid_flux + valid_error, alpha=0.3, color='lightgreen')
            
            # Add title and labels
            ax.set_title(f"Spectrum {spectrum_idx+1}, z={z:.4f}", fontsize=10)
            ax.set_ylabel('Normalized Flux', fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis='both', labelsize=8)
            

            ax.set_ylim(-1, 1)
        
        # Set x-label for the last plot
        axes[-1].set_xlabel('Velocity relative to line center (km/s)', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{prefix}_fig{fig_idx+1}.png")
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"   💾 Saved {fig_path}")
        plt.close(fig)


if __name__ == "__main__":
    generate_and_visualize_large_batch()
