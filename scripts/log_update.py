#!/usr/bin/env python3
import sys

from app.services.transparency import record_update

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.log_update "description"')
        sys.exit(1)
    record_update(sys.argv[1])
