#!/usr/bin/env python3
import sys

from app.services.submissions import update_submission_status

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python scripts/update_submission_status.py <id> <status>')
        sys.exit(1)
    update_submission_status(sys.argv[1], sys.argv[2])
