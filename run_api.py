import os
import sys

def main():    
    print("=" * 80)
    print("TELSTRA NETWORK DISRUPTIONS API SERVER")
    print("=" * 80)
    print()
    
    if not os.path.exists('outputs/models/telstra_nn_model.keras'):
        print(" ERROR: Model not found!")
        print()
        print("Please train the model first:")
        print("  python main.py")
        print()
        input("Press Enter to exit...")
        return
    
    if not os.path.exists('outputs/models/preprocessor.pkl'):
        print(" ERROR: Preprocessor not found!")
        print()
        print("Please save the preprocessor first:")
        print("  python save_preprocessor.py")
        print()
        input("Press Enter to exit...")
        return
    
    print("✓ Model found")
    print("✓ Preprocessor found")
    print()
    print("=" * 80)
    print("Starting API Server")
    print("=" * 80)
    print()
    print("Server will start on: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    print()
    
    try:
        import uvicorn
        sys.path.insert(0, os.path.join(os.getcwd(), 'deployment'))
        uvicorn.run("deployment.api:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("✗ ERROR: uvicorn not installed!")
        print()
        print("Install deployment dependencies:")
        print("  pip install -r requirements_deploy.txt")
        print()
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()