import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from astropy.io import fits

# Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

from LRDFinder.mock import MockProfileGenerator


def visualize_large_batch_spectra():
    """
    Generate and visualize 100 spectra with flux and error bands
    """
    print("=== Generating and Visualizing 100 Spectra with Errors ===")
    print("📁 Using file: /Users/bytedance/Desktop/code/LRDFinder/data/narrow_spectra_Ha.fits")
    
    # Specify the data file
    ha_file = '/Users/bytedance/Desktop/code/LRDFinder/data/narrow_spectra_Ha.fits'
    
    try:
        # Initialize the generator
        print("🔧 Initializing MockProfileGenerator...")
        generator = MockProfileGenerator(halpha_file=ha_file)
        
        # Get available spectra count
        ha_info = generator._ha_info
        if not ha_info:
            print("❌ Failed to get spectrum information")
            return False
        
        n_spectra_available = ha_info['n_spectra']
        print(f"📊 Available spectra: {n_spectra_available}")
        
        # Generate 100 spectra
        batch_size = 100
        indices = np.random.choice(n_spectra_available, size=batch_size, replace=False)
        print(f"🎯 Generating {batch_size} spectra, indices: {indices[:5]}...{indices[-5:]}")
        
        # Generate the spectra
        print("🚀 Running generate_broad_plus_narrow_batch...")
        result_hdul = generator.generate_broad_plus_narrow_batch(
            indices=indices,
            line='Ha',
            vel_range=2000,
            broad_to_narrow_ratio_range=(0.2, 1),
            broad_fwhm_range=(900, 1500),
            use_multiprocessing=True,
            n_processes=4
        )
        
        print(f"✅ generate_broad_plus_narrow_batch completed successfully!")
        
        # Extract data
        params = result_hdul['PARAMETERS'].data
        fluxes = result_hdul['FLUX_NOISY'].data
        ivars = result_hdul['IVAR'].data
        
        print(f"📊 Generated {len(params)} spectra")
        print(f"📏 Flux data shape: {fluxes.shape}")
        print(f"📏 IVAR data shape: {ivars.shape}")
        
        # Calculate errors from IVAR
        errors = np.sqrt(1.0 / ivars)
        errors[ivars == 0] = 0  # Set error to 0 where IVAR is 0
        
        # Visualize in batches of 10 spectra per figure
        print("📈 Visualizing spectra with error bands...")
        spectra_per_figure = 10
        total_figures = (batch_size + spectra_per_figure - 1) // spectra_per_figure
        
        rest_wave = 6564.614  # Ha rest wavelength
        c = 299792.458  # speed of light in km/s
        
        for fig_idx in range(total_figures):
            # Calculate the range of spectra for this figure
            start_idx = fig_idx * spectra_per_figure
            end_idx = min((fig_idx + 1) * spectra_per_figure, batch_size)
            current_batch = end_idx - start_idx
            
            print(f"   Creating figure {fig_idx+1}/{total_figures} with {current_batch} spectra...")
            
            # Create figure with subplots
            fig, axes = plt.subplots(current_batch, 1, figsize=(12, 2*current_batch), sharex=True)
            if current_batch == 1:
                axes = [axes]
            
            for i in range(current_batch):
                spectrum_idx = start_idx + i
                flux = fluxes[spectrum_idx]
                error = errors[spectrum_idx]
                ivar = ivars[spectrum_idx]
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
                valid_vel = vel_to_line
                valid_flux = flux
                valid_error = error
                
                # Plot
                ax = axes[i]
                ax.plot(valid_vel, valid_flux, linewidth=1.2, color='darkgreen', label='Flux')
                ax.plot()
                ax.fill_between(valid_vel, 
                              valid_flux - valid_error, 
                              valid_flux + valid_error, 
                              alpha=0.3, color='lightgreen', label='Error')
                ax.plot(valid_vel, valid_error, linestyle='--', color='red', label='Error')
                
                # Add title and labels
                ax.set_title(f"Spectrum {indices[spectrum_idx]}, z={z:.4f}", fontsize=10)
                ax.set_ylabel('Normalized Flux', fontsize=9)
                ax.grid(True, alpha=0.2)
                ax.tick_params(axis='both', labelsize=8)
                

                ax.set_ylim(-1, 1)
            
            # Set x-label for the last plot
            axes[-1].set_xlabel('Velocity relative to line center (km/s)', fontsize=10)
            
            # Add a legend to the first subplot
            axes[0].legend(fontsize=8, loc='upper right')
            
            # Adjust layout and save
            plt.tight_layout()
            output_path = f'tests/large_batch_spectra_fig{fig_idx+1}.png'
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            print(f"   💾 Saved to: {output_path}")
            plt.close(fig)
        
        print("\n=== Visualization Summary ===")
        print(f"✅ Generated {batch_size} spectra")
        print(f"✅ Created {total_figures} figures, {spectra_per_figure} spectra per figure")
        print(f"✅ Each spectrum shows flux (dark green) and error bands (light green)")
        print(f"✅ All figures saved to tests/ directory")
        print(f"\n📋 Files generated:")
        for fig_idx in range(total_figures):
            print(f"   tests/large_batch_spectra_fig{fig_idx+1}.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    visualize_large_batch_spectra()
