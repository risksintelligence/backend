#!/usr/bin/env python3
"""
Cleanup script for ACLED error logs that flooded the L3 cache.
Removes duplicate error files and keeps only the most recent 100 for analysis.
"""

import os
import glob
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def cleanup_acled_errors():
    """Clean up the excessive ACLED error log files."""
    error_log_dir = "/Users/omoshola/Documents/riskx_obervatory 2/backend/l3_cache/error_logs"
    
    if not os.path.exists(error_log_dir):
        logger.info("Error log directory does not exist")
        return
    
    # Find all ACLED error files
    acled_files = glob.glob(os.path.join(error_log_dir, "error_acled_*.json"))
    total_files = len(acled_files)
    
    logger.info(f"Found {total_files} ACLED error files")
    
    if total_files <= 100:
        logger.info("File count is reasonable, no cleanup needed")
        return
    
    # Sort by modification time (newest first)
    acled_files.sort(key=os.path.getmtime, reverse=True)
    
    # Keep the newest 100 files, delete the rest
    files_to_keep = acled_files[:100]
    files_to_delete = acled_files[100:]
    
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except OSError as e:
            logger.error(f"Failed to delete {file_path}: {e}")
    
    logger.info(f"Cleanup complete: kept {len(files_to_keep)} files, deleted {deleted_count} files")
    
    # Calculate space saved (assuming ~23 bytes per file based on error.md pattern)
    space_saved_kb = deleted_count * 23 / 1024
    logger.info(f"Estimated space saved: {space_saved_kb:.1f} KB")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_acled_errors()