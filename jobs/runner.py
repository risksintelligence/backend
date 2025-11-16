#!/usr/bin/env python3
import os
import sys
import time

def main():
    print("🚀 RRIO Worker Starting...")
    
    # Wait for environment to be ready
    for i in range(30):
        try:
            import pydantic
            break
        except ImportError:
            time.sleep(2)
    else:
        print("❌ Dependencies not ready")
        sys.exit(1)
    
    # Import and run worker
    sys.path.insert(0, '/opt/render/project/src/scripts')
    from start_worker import main as worker_main
    worker_main()

if __name__ == "__main__":
    main()