#!/usr/bin/env python3
"""
Log cleanup utilities for RiskX platform.
Manages log file rotation, compression, and removal of old logs.
"""

import os
import sys
import gzip
import shutil
import datetime
import glob
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class LogCleanup:
    """Log file cleanup and management for RiskX platform."""
    
    def __init__(self):
        self.log_base_dir = Path("logs")
        self.log_dirs = ["app", "etl", "ml", "api"]
        self.cleanup_stats = {
            "files_compressed": 0,
            "files_removed": 0,
            "space_saved": 0,
            "errors": []
        }
    
    def ensure_log_directories(self) -> None:
        """Ensure all log directories exist."""
        self.log_base_dir.mkdir(exist_ok=True)
        for log_dir in self.log_dirs:
            (self.log_base_dir / log_dir).mkdir(exist_ok=True)
    
    def get_log_files(self, pattern: str = "*.log") -> List[Path]:
        """Get all log files matching pattern."""
        log_files = []
        for log_dir in self.log_dirs:
            dir_path = self.log_base_dir / log_dir
            if dir_path.exists():
                log_files.extend(dir_path.glob(pattern))
        
        # Also check for logs in base directory
        if self.log_base_dir.exists():
            log_files.extend(self.log_base_dir.glob(pattern))
        
        return log_files
    
    def compress_old_logs(self, days_old: int = 7) -> None:
        """Compress log files older than specified days."""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
        
        log_files = self.get_log_files("*.log")
        
        for log_file in log_files:
            try:
                # Skip if file is already compressed or very recent
                if log_file.suffix == '.gz':
                    continue
                
                file_mtime = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    # Compress the file
                    compressed_path = Path(str(log_file) + '.gz')
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Get space saved
                    original_size = log_file.stat().st_size
                    compressed_size = compressed_path.stat().st_size
                    space_saved = original_size - compressed_size
                    
                    # Remove original file
                    log_file.unlink()
                    
                    self.cleanup_stats["files_compressed"] += 1
                    self.cleanup_stats["space_saved"] += space_saved
                    
                    logger.info(f"Compressed {log_file} -> {compressed_path} "
                               f"(saved {space_saved} bytes)")
                    
            except Exception as e:
                error_msg = f"Failed to compress {log_file}: {e}"
                logger.error(error_msg)
                self.cleanup_stats["errors"].append(error_msg)
    
    def remove_old_logs(self, days_old: int = 30) -> None:
        """Remove log files older than specified days."""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)
        
        # Get both regular and compressed log files
        log_patterns = ["*.log", "*.log.gz", "*.json"]
        
        for pattern in log_patterns:
            log_files = self.get_log_files(pattern)
            
            for log_file in log_files:
                try:
                    file_mtime = datetime.datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        file_size = log_file.stat().st_size
                        log_file.unlink()
                        
                        self.cleanup_stats["files_removed"] += 1
                        self.cleanup_stats["space_saved"] += file_size
                        
                        logger.info(f"Removed old log file: {log_file}")
                        
                except Exception as e:
                    error_msg = f"Failed to remove {log_file}: {e}"
                    logger.error(error_msg)
                    self.cleanup_stats["errors"].append(error_msg)
    
    def rotate_large_logs(self, max_size_mb: int = 100) -> None:
        """Rotate log files that exceed maximum size."""
        max_size_bytes = max_size_mb * 1024 * 1024
        
        log_files = self.get_log_files("*.log")
        
        for log_file in log_files:
            try:
                if log_file.stat().st_size > max_size_bytes:
                    # Create rotated filename with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    rotated_name = f"{log_file.stem}_{timestamp}.log"
                    rotated_path = log_file.parent / rotated_name
                    
                    # Move current log to rotated name
                    shutil.move(log_file, rotated_path)
                    
                    # Create new empty log file
                    log_file.touch()
                    
                    logger.info(f"Rotated large log file: {log_file} -> {rotated_path}")
                    
                    # Compress the rotated file immediately
                    compressed_path = Path(str(rotated_path) + '.gz')
                    with open(rotated_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    rotated_path.unlink()
                    self.cleanup_stats["files_compressed"] += 1
                    
            except Exception as e:
                error_msg = f"Failed to rotate {log_file}: {e}"
                logger.error(error_msg)
                self.cleanup_stats["errors"].append(error_msg)
    
    def clean_cache_logs(self) -> None:
        """Clean up cache-related log files."""
        try:
            cache_dir = Path("data/cache")
            if cache_dir.exists():
                # Remove temporary cache files
                temp_files = list(cache_dir.rglob("*.tmp"))
                temp_files.extend(cache_dir.rglob("*.temp"))
                temp_files.extend(cache_dir.rglob("*.lock"))
                
                for temp_file in temp_files:
                    try:
                        file_size = temp_file.stat().st_size
                        temp_file.unlink()
                        self.cleanup_stats["files_removed"] += 1
                        self.cleanup_stats["space_saved"] += file_size
                        logger.info(f"Removed cache temp file: {temp_file}")
                    except Exception as e:
                        error_msg = f"Failed to remove cache file {temp_file}: {e}"
                        logger.error(error_msg)
                        self.cleanup_stats["errors"].append(error_msg)
                        
        except Exception as e:
            error_msg = f"Cache cleanup failed: {e}"
            logger.error(error_msg)
            self.cleanup_stats["errors"].append(error_msg)
    
    def generate_cleanup_report(self) -> Dict[str, Any]:
        """Generate cleanup report with statistics."""
        report = {
            "cleanup_date": datetime.datetime.now().isoformat(),
            "files_compressed": self.cleanup_stats["files_compressed"],
            "files_removed": self.cleanup_stats["files_removed"],
            "space_saved_bytes": self.cleanup_stats["space_saved"],
            "space_saved_mb": round(self.cleanup_stats["space_saved"] / (1024 * 1024), 2),
            "errors": self.cleanup_stats["errors"],
            "log_directories": []
        }
        
        # Add directory statistics
        for log_dir in self.log_dirs:
            dir_path = self.log_base_dir / log_dir
            if dir_path.exists():
                log_files = list(dir_path.glob("*"))
                total_size = sum(f.stat().st_size for f in log_files if f.is_file())
                
                report["log_directories"].append({
                    "directory": str(dir_path),
                    "file_count": len(log_files),
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                })
        
        return report
    
    def full_cleanup(self, compress_days: int = 7, remove_days: int = 30, 
                    max_size_mb: int = 100) -> Dict[str, Any]:
        """Perform complete log cleanup operation."""
        logger.info("Starting full log cleanup")
        
        # Ensure directories exist
        self.ensure_log_directories()
        
        # Perform cleanup operations
        self.rotate_large_logs(max_size_mb)
        self.compress_old_logs(compress_days)
        self.remove_old_logs(remove_days)
        self.clean_cache_logs()
        
        # Generate report
        report = self.generate_cleanup_report()
        
        logger.info(f"Log cleanup completed: "
                   f"{report['files_compressed']} compressed, "
                   f"{report['files_removed']} removed, "
                   f"{report['space_saved_mb']} MB saved")
        
        return report


def main():
    """Main log cleanup script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RiskX Log Cleanup Utility")
    parser.add_argument("--compress-days", type=int, default=7,
                       help="Compress logs older than X days (default: 7)")
    parser.add_argument("--remove-days", type=int, default=30,
                       help="Remove logs older than X days (default: 30)")
    parser.add_argument("--max-size-mb", type=int, default=100,
                       help="Rotate logs larger than X MB (default: 100)")
    parser.add_argument("--report-file", type=str,
                       help="Save cleanup report to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    cleanup_manager = LogCleanup()
    
    # Perform cleanup
    report = cleanup_manager.full_cleanup(
        compress_days=args.compress_days,
        remove_days=args.remove_days,
        max_size_mb=args.max_size_mb
    )
    
    # Print summary
    print(f"Log cleanup completed:")
    print(f"  Files compressed: {report['files_compressed']}")
    print(f"  Files removed: {report['files_removed']}")
    print(f"  Space saved: {report['space_saved_mb']} MB")
    
    if report["errors"]:
        print(f"  Errors: {len(report['errors'])}")
        for error in report["errors"]:
            print(f"    - {error}")
    
    # Save report if requested
    if args.report_file:
        with open(args.report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.report_file}")


if __name__ == "__main__":
    main()