"""
Run everything: Train model, save preprocessor, and optionally start API
"""

import os
import sys

def run_all():
    """Execute complete pipeline"""
    
    print("\n" + "=" * 80)
    print("TELSTRA NETWORK DISRUPTIONS - COMPLETE PIPELINE")
    print("=" * 80)
    print()
    
    # Step 1: Train Model
    print("[STEP 1/3] Training Model...")
    print("-" * 80)
    try:
        import main
        print("\n✓ Model training completed!")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        return
    
    print("\n" + "=" * 80)
    input("Press Enter to continue to Step 2...")
    
    # Step 2: Save Preprocessor
    print("\n[STEP 2/3] Saving Preprocessor...")
    print("-" * 80)
    try:
        import save_preprocessor
        save_preprocessor.save_preprocessor()
        print("\n✓ Preprocessor saved!")
    except Exception as e:
        print(f"\n✗ Error saving preprocessor: {e}")
        return
    
    print("\n" + "=" * 80)
    
    # Step 3: Ask about API
    print("\n[STEP 3/3] API Deployment")
    print("-" * 80)
    response = input("\nDo you want to start the API server? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nStarting API server...")
        try:
            import run_api
            run_api.main()
        except Exception as e:
            print(f"\n✗ Error starting API: {e}")
    else:
        print("\n✓ Pipeline completed!")
        print("\nTo start the API later, run:")
        print("  python run_api.py")
    
    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        run_all()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\nPress Enter to exit...")