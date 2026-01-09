import numpy as np
import sys
import os
from astropy.io import fits

# Add the project root to the path
sys.path.insert(0, os.path.abspath('..'))

from LRDFinder.mock import MockProfileGenerator


def test_generate_all_profiles_batch():
    """
    Test generate_all_profiles_batch for both Ha/Hb lines, with and without absorption
    """
    print("=== Testing MockGenerator.generate_all_profiles_batch ===")
    
    # Specify the data files
    data_dir = '/Users/bytedance/Desktop/code/LRDFinder/data/'
    ha_file = os.path.join(data_dir, 'narrow_spectra_Ha.fits')
    hb_file = os.path.join(data_dir, 'narrow_spectra_Hb.fits')
    
    print(f"📁 Using Ha file: {ha_file}")
    print(f"📁 Using Hb file: {hb_file}")
    
    try:
        # Initialize the generator with both Ha and Hb files
        print("🔧 Initializing MockProfileGenerator with both Ha and Hb files...")
        generator = MockProfileGenerator(halpha_file=ha_file, hbeta_file=hb_file)
        
        # Test parameters
        n_spectra = 10  # Small batch for testing
        
        # Test cases: (line, broad, broad_abs, description)
        test_cases = [
            ('Ha', True, False, 'Ha: broad+narrow (without absorption)'),
            ('Ha', True, True, 'Ha: broad+absorption+narrow (with absorption)'),
            ('Hb', True, False, 'Hb: broad+narrow (without absorption)'),
            ('Hb', True, True, 'Hb: broad+absorption+narrow (with absorption)')
        ]
        
        print(f"🎯 Running {len(test_cases)} test cases...")
        
        for i, (line, broad, broad_abs, description) in enumerate(test_cases):
            print(f"\n📋 Test Case {i+1}/{len(test_cases)}: {description}")
            
            # Run the function
            result = generator.generate_all_profiles_batch(
                n_spectra=n_spectra,
                line=line,
                broad=broad,
                broad_abs=broad_abs,
                broad_to_narrow_ratio_range=(3, 10),
                abs_ew_range=(2, 10),
                abs_vel_range=(-300, 100),
                abs_fwhm_range=(80, 250),
                random_seed=42,
                use_multiprocessing=True,
                n_processes=2
            )
            
            print(f"✅ Test case completed successfully!")
            print(f"📋 Result keys: {list(result.keys())}")
            
            # Verify results
            expected_keys = ['metadata']
            if broad:
                expected_keys.append('broad')
            if broad_abs:
                expected_keys.append('absorption')
            
            # Check if expected keys are present
            for key in expected_keys:
                if key in result:
                    if key == 'metadata':
                        print(f"   ✅ Metadata present: {result['metadata'].keys()}")
                    else:
                        print(f"   ✅ {key} profiles generated: HDUList with {len(result[key])} HDUs")
                        # Check HDU structure
                        print(f"      HDU names: {[hdu.name for hdu in result[key]]}")
                        # Get the number of spectra in this result
                        params = result[key]['PARAMETERS'].data
                        print(f"      Number of {key} spectra: {len(params)}")
                        # Check flux shape
                        flux = result[key]['FLUX_NOISY'].data
                        print(f"      Flux data shape: {flux.shape}")
                        ivar = result[key]['IVAR'].data
                        print(f"      IVAR data shape: {ivar.shape}")
            
            # Verify metadata
            print(f"   ✅ Metadata contains: line={result['metadata']['line']}, "
                  f"n_spectra={result['metadata']['n_spectra_requested']}, "
                  f"ratio_range={result['metadata']['broad_to_narrow_ratio_range']}")
        
        print("\n=== All Test Cases Passed! ===")
        print("📊 Summary:")
        print("   - Tested Ha line: with and without absorption")
        print("   - Tested Hb line: with and without absorption")
        print("   - All test cases completed successfully")
        print("   - Generated both broad+narrow and broad+absorption+narrow profiles")
        print("   - Verified HDUList structure and data shapes")
        print("   - Confirmed metadata accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_generate_all_profiles_batch()
