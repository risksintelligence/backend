#!/usr/bin/env python3
"""
Database backup utilities for RiskX platform.
Provides automated backup functionality for PostgreSQL and SQLite databases.
"""

import os
import sys
import subprocess
import datetime
import gzip
import shutil
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import get_settings
from src.core.database import get_database_url

logger = logging.getLogger(__name__)


class DatabaseBackup:
    """Database backup management for RiskX platform."""
    
    def __init__(self):
        self.settings = get_settings()
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_backup_filename(self, db_type: str) -> str:
        """Generate backup filename with timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"riskx_{db_type}_{timestamp}.sql"
    
    def backup_sqlite(self, db_path: str) -> Optional[str]:
        """Backup SQLite database."""
        try:
            backup_filename = self.create_backup_filename("sqlite")
            backup_path = self.backup_dir / backup_filename
            
            # Connect to database and create backup
            with sqlite3.connect(db_path) as conn:
                with open(backup_path, 'w') as f:
                    for line in conn.iterdump():
                        f.write('%s\n' % line)
            
            # Compress backup
            compressed_path = f"{backup_path}.gz"
            with open(backup_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed file
            os.remove(backup_path)
            
            logger.info(f"SQLite backup created: {compressed_path}")
            return compressed_path
            
        except Exception as e:
            logger.error(f"SQLite backup failed: {e}")
            return None
    
    def backup_postgresql(self, database_url: str) -> Optional[str]:
        """Backup PostgreSQL database using pg_dump."""
        try:
            backup_filename = self.create_backup_filename("postgres")
            backup_path = self.backup_dir / backup_filename
            
            # Extract connection details from URL
            # postgresql://user:password@host:port/database
            url_parts = database_url.replace("postgresql://", "").split("/")
            db_name = url_parts[-1]
            auth_host = url_parts[0].split("@")
            
            if len(auth_host) == 2:
                auth_parts = auth_host[0].split(":")
                host_port = auth_host[1].split(":")
                user = auth_parts[0]
                password = auth_parts[1] if len(auth_parts) > 1 else ""
                host = host_port[0]
                port = host_port[1] if len(host_port) > 1 else "5432"
            else:
                host = auth_host[0].split(":")[0]
                port = auth_host[0].split(":")[1] if ":" in auth_host[0] else "5432"
                user = os.getenv("PGUSER", "postgres")
                password = os.getenv("PGPASSWORD", "")
            
            # Set environment variables for pg_dump
            env = os.environ.copy()
            if password:
                env["PGPASSWORD"] = password
            
            # Run pg_dump
            cmd = [
                "pg_dump",
                "-h", host,
                "-p", port,
                "-U", user,
                "-d", db_name,
                "--no-password",
                "-f", str(backup_path)
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Compress backup
                compressed_path = f"{backup_path}.gz"
                with open(backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed file
                os.remove(backup_path)
                
                logger.info(f"PostgreSQL backup created: {compressed_path}")
                return compressed_path
            else:
                logger.error(f"pg_dump failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"PostgreSQL backup failed: {e}")
            return None
    
    def backup_cache_data(self) -> Optional[str]:
        """Backup cache data directory."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"riskx_cache_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_filename
            
            cache_dir = Path("data/cache")
            if not cache_dir.exists():
                logger.warning("Cache directory not found")
                return None
            
            # Create tar.gz archive of cache directory
            cmd = ["tar", "-czf", str(backup_path), "-C", "data", "cache"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Cache backup created: {backup_path}")
                return str(backup_path)
            else:
                logger.error(f"Cache backup failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Cache backup failed: {e}")
            return None
    
    def cleanup_old_backups(self, keep_days: int = 30) -> None:
        """Remove backup files older than specified days."""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
            
            for backup_file in self.backup_dir.glob("riskx_*"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    logger.info(f"Removed old backup: {backup_file}")
                    
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def create_full_backup(self) -> Dict[str, Any]:
        """Create complete backup of all data."""
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "backups": [],
            "errors": []
        }
        
        try:
            # Backup database
            database_url = get_database_url()
            
            if database_url.startswith("sqlite"):
                # Extract SQLite file path
                db_path = database_url.replace("sqlite:///", "")
                if os.path.exists(db_path):
                    backup_path = self.backup_sqlite(db_path)
                    if backup_path:
                        results["backups"].append({"type": "sqlite", "path": backup_path})
                    else:
                        results["errors"].append("SQLite backup failed")
                        
            elif database_url.startswith("postgresql"):
                backup_path = self.backup_postgresql(database_url)
                if backup_path:
                    results["backups"].append({"type": "postgresql", "path": backup_path})
                else:
                    results["errors"].append("PostgreSQL backup failed")
            
            # Backup cache data
            cache_backup = self.backup_cache_data()
            if cache_backup:
                results["backups"].append({"type": "cache", "path": cache_backup})
            else:
                results["errors"].append("Cache backup failed")
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            logger.info(f"Full backup completed. Created {len(results['backups'])} backups")
            
        except Exception as e:
            logger.error(f"Full backup failed: {e}")
            results["errors"].append(str(e))
        
        return results


def main():
    """Main backup script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RiskX Database Backup Utility")
    parser.add_argument("--type", choices=["full", "db", "cache"], default="full",
                       help="Type of backup to perform")
    parser.add_argument("--cleanup-days", type=int, default=30,
                       help="Days of backups to keep (default: 30)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    backup_manager = DatabaseBackup()
    
    if args.type == "full":
        results = backup_manager.create_full_backup()
        print(f"Backup completed: {len(results['backups'])} files created")
        if results["errors"]:
            print(f"Errors: {results['errors']}")
            sys.exit(1)
            
    elif args.type == "db":
        database_url = get_database_url()
        if database_url.startswith("sqlite"):
            db_path = database_url.replace("sqlite:///", "")
            result = backup_manager.backup_sqlite(db_path)
        else:
            result = backup_manager.backup_postgresql(database_url)
        
        if result:
            print(f"Database backup created: {result}")
        else:
            print("Database backup failed")
            sys.exit(1)
            
    elif args.type == "cache":
        result = backup_manager.backup_cache_data()
        if result:
            print(f"Cache backup created: {result}")
        else:
            print("Cache backup failed")
            sys.exit(1)


if __name__ == "__main__":
    main()