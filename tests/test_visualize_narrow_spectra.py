import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from astropy.io import fits

# Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))


def visualize_narrow_spectra():
    """
    Visualize the spectra from narrow_spectra_Ha.fits
    """
    print("=== Visualizing Spectra from narrow_spectra_Ha.fits ===")
    
    # Specify the file path
    file_path = '/Users/bytedance/Desktop/code/LRDFinder/data/narrow_spectra_Ha.fits'
    print(f"📁 Using file: {file_path}")
    
    # Create a folder for visualizations
    vis_dir = '/Users/bytedance/Desktop/code/LRDFinder/tests/narrow_spectra_visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    print(f"📁 Creating visualization folder: {vis_dir}")
    
    try:
        # Open the FITS file
        print("🔍 Opening FITS file...")
        with fits.open(file_path) as hdul:
            print(f"📊 FITS file structure: {[hdu.name for hdu in hdul]}")
            
            # Get the PARAMETERS, FLUX_NOISY, and IVAR data
            params = hdul['PARAMETERS'].data
            flux_noisy = hdul['FLUX_NOISY'].data
            ivar = hdul['IVAR'].data
            
            n_spectra = len(params)
            print(f"📊 Total spectra: {n_spectra}")
            print(f"📏 Flux data shape: {flux_noisy.shape}")
            print(f"📏 IVAR data shape: {ivar.shape}")
            
            # Calculate errors from IVAR
            errors = np.sqrt(1.0 / ivar)
            errors[ivar == 0] = 0  # Set error to 0 where IVAR is 0
            
            # Get rest wavelength for Ha
            rest_wave = 6564.614  # H-alpha rest wavelength
            c = 299792.458  # speed of light in km/s
            
            # Visualize a sample of spectra
            sample_size = 20  # Visualize 20 spectra
            sample_indices = np.random.choice(n_spectra, size=sample_size, replace=False)
            print(f"🎯 Visualizing {sample_size} spectra from indices: {sample_indices[:5]}...{sample_indices[-5:]}")
            
            # Create figures with 5 spectra each
            spectra_per_figure = 5
            n_figures = (sample_size + spectra_per_figure - 1) // spectra_per_figure
            
            for fig_idx in range(n_figures):
                start_idx = fig_idx * spectra_per_figure
                end_idx = min((fig_idx + 1) * spectra_per_figure, sample_size)
                current_spectra = end_idx - start_idx
                
                # Create figure
                fig, axes = plt.subplots(current_spectra, 1, figsize=(12, 2*current_spectra), sharex=True)
                if current_spectra == 1:
                    axes = [axes]
                
                for i in range(current_spectra):
                    sample_idx = sample_indices[start_idx + i]
                    flux = flux_noisy[sample_idx]
                    error = errors[sample_idx]
                    param = params[sample_idx]
                    
                    # Calculate wavelength and velocity
                    wave_min = param['WAV_MIN']
                    dwave = param['DWAVE']
                    z = param['REDSHIFT']
                    
                    # Use actual flux array length instead of metadata N_PIXELS
                    actual_pixels = len(flux)
                    
                    # Create wavelength array
                    wave = wave_min + np.arange(actual_pixels) * dwave
                    
                    # Convert to velocity space
                    rest_wave_obs = wave / (1 + z)
                    vel_to_line = (rest_wave_obs - rest_wave) / rest_wave * c
                    
                    # Mask zero values
                    mask = (flux != 0.) & (error != 0.)
                    valid_vel = vel_to_line[mask]
                    valid_flux = flux[mask]
                    valid_error = error[mask]
                    
                    # Plot
                    ax = axes[i]
                    ax.plot(valid_vel, valid_flux, linewidth=1.2, color='darkblue')
                    ax.fill_between(valid_vel, valid_flux - valid_error, 
                                  valid_flux + valid_error, alpha=0.3, color='lightblue')
                    
                    # Add title and labels
                    ax.set_title(f"Narrow Spectrum {sample_idx}, z={z:.4f}", fontsize=10)
                    ax.set_ylabel('Flux', fontsize=9)
                    ax.grid(True, alpha=0.2)
                    ax.tick_params(axis='both', labelsize=8)
                    
                    # Set y-limits
                    if len(valid_flux) > 0:
                        # Calculate flux ranges and handle NaN/Inf values
                        flux_minus_error = valid_flux - valid_error
                        flux_plus_error = valid_flux + valid_error
                        
                        y_min = np.min(flux_minus_error)
                        y_max = np.max(flux_plus_error)
                        
                        # Check if y_min or y_max is NaN/Inf
                        if np.isfinite(y_min) and np.isfinite(y_max):
                            ax.set_ylim(y_min * 1.1, y_max * 1.1)
                        else:
                            # Use default limits if values are invalid
                            ax.set_ylim(-1, 1)
                    else:
                        ax.set_ylim(-1, 1)
                
                # Set x-label for the last plot
                axes[-1].set_xlabel('Velocity relative to line center (km/s)', fontsize=10)
                
                # Add a general title
                fig.suptitle(f"Narrow H-alpha Spectra (Sample {fig_idx+1}/{n_figures})")
                
                # Adjust layout and save
                plt.tight_layout()
                fig_path = os.path.join(vis_dir, f"narrow_spectra_fig{fig_idx+1}.png")
                plt.savefig(fig_path, dpi=200, bbox_inches='tight')
                print(f"   💾 Saved {fig_path}")
                plt.close(fig)
            
            # Also create a histogram of redshifts for the entire dataset
            print("📊 Creating redshift histogram...")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(params['REDSHIFT'], bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax.set_title('Redshift Distribution of Narrow Spectra')
            ax.set_xlabel('Redshift (z)')
            ax.set_ylabel('Number of Spectra')
            ax.grid(True, alpha=0.3)
            redshift_hist_path = os.path.join(vis_dir, "redshift_distribution.png")
            plt.savefig(redshift_hist_path, dpi=200, bbox_inches='tight')
            print(f"   💾 Saved redshift histogram: {redshift_hist_path}")
            plt.close(fig)
            
            print("\n=== All Visualizations Completed! ===")
            print(f"📊 Spectra visualized: {sample_size} out of {n_spectra}")
            print(f"📁 Visualizations saved to: {vis_dir}")
            return True
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    visualize_narrow_spectra()
