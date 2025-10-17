"""
File storage utilities for RiskX platform.
Provides local and cloud file storage with automatic fallback mechanisms.
"""

import os
import json
import csv
import pickle
import logging
from typing import Optional, Dict, Any, List, Union, BinaryIO, TextIO
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
import hashlib
import shutil
import tempfile

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logger = logging.getLogger('riskx.data.storage.files')
    logger.warning("boto3 not available, S3 storage disabled")

import pandas as pd

from ...core.exceptions import StorageError, ConfigurationError
from ...utils.helpers import generate_hash, safe_filename, get_file_extension
from ...utils.constants import StorageConfig

logger = logging.getLogger('riskx.data.storage.files')


@dataclass
class FileConfig:
    """File storage configuration settings."""
    local_storage_path: str = "/tmp/riskx_storage"
    s3_bucket_name: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    compression_enabled: bool = True
    encryption_enabled: bool = False
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: List[str] = None
    backup_enabled: bool = True
    versioning_enabled: bool = False
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.json', '.csv', '.parquet', '.pkl', '.txt', '.log']
    
    @classmethod
    def from_env(cls) -> 'FileConfig':
        """Create config from environment variables."""
        return cls(
            local_storage_path=os.getenv('FILE_STORAGE_PATH', '/tmp/riskx_storage'),
            s3_bucket_name=os.getenv('S3_BUCKET_NAME'),
            s3_region=os.getenv('S3_REGION', 'us-east-1'),
            s3_access_key=os.getenv('AWS_ACCESS_KEY_ID'),
            s3_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            s3_endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            compression_enabled=os.getenv('FILE_COMPRESSION', 'true').lower() == 'true',
            encryption_enabled=os.getenv('FILE_ENCRYPTION', 'false').lower() == 'true',
            max_file_size=int(os.getenv('MAX_FILE_SIZE', str(100 * 1024 * 1024))),
            backup_enabled=os.getenv('FILE_BACKUP', 'true').lower() == 'true',
            versioning_enabled=os.getenv('FILE_VERSIONING', 'false').lower() == 'true'
        )


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_file(self, file_path: str, data: Union[bytes, str], 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save file to storage."""
        pass
    
    @abstractmethod
    def load_file(self, file_path: str) -> Optional[bytes]:
        """Load file from storage."""
        pass
    
    @abstractmethod
    def delete_file(self, file_path: str) -> bool:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in storage."""
        pass
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix filter."""
        pass
    
    @abstractmethod
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata and information."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, config: FileConfig):
        self.config = config
        self.storage_path = Path(config.local_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._stats = {
            'files_saved': 0,
            'files_loaded': 0,
            'files_deleted': 0,
            'bytes_written': 0,
            'bytes_read': 0,
            'errors': 0
        }
    
    def save_file(self, file_path: str, data: Union[bytes, str], 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save file to local storage."""
        try:
            # Validate file path and extension
            if not self._validate_file_path(file_path):
                return False
            
            full_path = self.storage_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle backup if enabled
            if self.config.backup_enabled and full_path.exists():
                self._create_backup(full_path)
            
            # Write data
            if isinstance(data, str):
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(data)
                bytes_written = len(data.encode('utf-8'))
            else:
                with open(full_path, 'wb') as f:
                    f.write(data)
                bytes_written = len(data)
            
            # Save metadata if provided
            if metadata:
                self._save_metadata(full_path, metadata)
            
            self._stats['files_saved'] += 1
            self._stats['bytes_written'] += bytes_written
            
            logger.debug(f"Saved file: {file_path} ({bytes_written} bytes)")
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error saving file {file_path}: {e}")
            return False
    
    def load_file(self, file_path: str) -> Optional[bytes]:
        """Load file from local storage."""
        try:
            full_path = self.storage_path / file_path
            
            if not full_path.exists():
                return None
            
            with open(full_path, 'rb') as f:
                data = f.read()
            
            self._stats['files_loaded'] += 1
            self._stats['bytes_read'] += len(data)
            
            logger.debug(f"Loaded file: {file_path} ({len(data)} bytes)")
            return data
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error loading file {file_path}: {e}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from local storage."""
        try:
            full_path = self.storage_path / file_path
            
            if not full_path.exists():
                return False
            
            # Delete metadata file if exists
            metadata_path = self._get_metadata_path(full_path)
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Delete main file
            full_path.unlink()
            
            self._stats['files_deleted'] += 1
            logger.debug(f"Deleted file: {file_path}")
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in local storage."""
        full_path = self.storage_path / file_path
        return full_path.exists()
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix filter."""
        try:
            files = []
            search_path = self.storage_path / prefix if prefix else self.storage_path
            
            if search_path.is_file():
                return [str(search_path.relative_to(self.storage_path))]
            
            pattern = "**/*" if not prefix else f"{prefix}/**/*"
            for file_path in self.storage_path.glob(pattern):
                if file_path.is_file() and not file_path.name.endswith('.metadata'):
                    relative_path = str(file_path.relative_to(self.storage_path))
                    files.append(relative_path)
            
            return sorted(files)
        
        except Exception as e:
            logger.error(f"Error listing files with prefix {prefix}: {e}")
            return []
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata and information."""
        try:
            full_path = self.storage_path / file_path
            
            if not full_path.exists():
                return None
            
            stat = full_path.stat()
            info = {
                'path': file_path,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'extension': get_file_extension(file_path),
                'is_file': full_path.is_file()
            }
            
            # Load custom metadata if available
            metadata = self._load_metadata(full_path)
            if metadata:
                info['metadata'] = metadata
            
            return info
        
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get local storage statistics."""
        stats = self._stats.copy()
        
        try:
            total_files = len(self.list_files())
            total_size = sum(
                (self.storage_path / f).stat().st_size 
                for f in self.list_files()
                if (self.storage_path / f).exists()
            )
            
            stats['total_files'] = total_files
            stats['total_size'] = total_size
            stats['storage_path'] = str(self.storage_path)
        except Exception:
            stats['total_files'] = 0
            stats['total_size'] = 0
        
        return stats
    
    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path and extension."""
        if not file_path:
            return False
        
        # Check file size would be within limits
        extension = get_file_extension(file_path)
        if self.config.allowed_extensions and extension not in self.config.allowed_extensions:
            logger.warning(f"File extension {extension} not allowed")
            return False
        
        return True
    
    def _create_backup(self, file_path: Path):
        """Create backup of existing file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.parent / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {e}")
    
    def _save_metadata(self, file_path: Path, metadata: Dict[str, Any]):
        """Save metadata for file."""
        try:
            metadata_path = self._get_metadata_path(file_path)
            metadata['saved_at'] = datetime.utcnow().isoformat()
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, default=str, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata for {file_path}: {e}")
    
    def _load_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata for file."""
        try:
            metadata_path = self._get_metadata_path(file_path)
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {file_path}: {e}")
        return None
    
    def _get_metadata_path(self, file_path: Path) -> Path:
        """Get metadata file path."""
        return file_path.parent / f".{file_path.name}.metadata"


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend."""
    
    def __init__(self, config: FileConfig):
        self.config = config
        self._client = None
        self._is_connected = False
        self._stats = {
            'files_saved': 0,
            'files_loaded': 0,
            'files_deleted': 0,
            'bytes_written': 0,
            'bytes_read': 0,
            'errors': 0
        }
        
        if S3_AVAILABLE and config.s3_bucket_name:
            self._initialize_s3_client()
    
    def _initialize_s3_client(self):
        """Initialize S3 client."""
        try:
            session = boto3.Session(
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key,
                region_name=self.config.s3_region
            )
            
            self._client = session.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url
            )
            
            # Test connection
            self._client.head_bucket(Bucket=self.config.s3_bucket_name)
            self._is_connected = True
            logger.info("S3 storage backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize S3 storage: {e}")
            self._is_connected = False
    
    def save_file(self, file_path: str, data: Union[bytes, str], 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save file to S3."""
        if not self._is_connected:
            return False
        
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Prepare metadata
            s3_metadata = {
                'uploaded_at': datetime.utcnow().isoformat(),
                'file_size': str(len(data))
            }
            if metadata:
                s3_metadata.update({k: str(v) for k, v in metadata.items()})
            
            # Upload file
            self._client.put_object(
                Bucket=self.config.s3_bucket_name,
                Key=file_path,
                Body=data,
                Metadata=s3_metadata
            )
            
            self._stats['files_saved'] += 1
            self._stats['bytes_written'] += len(data)
            
            logger.debug(f"Saved file to S3: {file_path} ({len(data)} bytes)")
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error saving file to S3 {file_path}: {e}")
            return False
    
    def load_file(self, file_path: str) -> Optional[bytes]:
        """Load file from S3."""
        if not self._is_connected:
            return None
        
        try:
            response = self._client.get_object(
                Bucket=self.config.s3_bucket_name,
                Key=file_path
            )
            
            data = response['Body'].read()
            
            self._stats['files_loaded'] += 1
            self._stats['bytes_read'] += len(data)
            
            logger.debug(f"Loaded file from S3: {file_path} ({len(data)} bytes)")
            return data
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            self._stats['errors'] += 1
            logger.error(f"Error loading file from S3 {file_path}: {e}")
            return None
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error loading file from S3 {file_path}: {e}")
            return None
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from S3."""
        if not self._is_connected:
            return False
        
        try:
            self._client.delete_object(
                Bucket=self.config.s3_bucket_name,
                Key=file_path
            )
            
            self._stats['files_deleted'] += 1
            logger.debug(f"Deleted file from S3: {file_path}")
            return True
        
        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"Error deleting file from S3 {file_path}: {e}")
            return False
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in S3."""
        if not self._is_connected:
            return False
        
        try:
            self._client.head_object(
                Bucket=self.config.s3_bucket_name,
                Key=file_path
            )
            return True
        
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking file existence in S3 {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking file existence in S3 {file_path}: {e}")
            return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix filter."""
        if not self._is_connected:
            return []
        
        try:
            files = []
            paginator = self._client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.config.s3_bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files.append(obj['Key'])
            
            return sorted(files)
        
        except Exception as e:
            logger.error(f"Error listing files from S3 with prefix {prefix}: {e}")
            return []
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file metadata and information."""
        if not self._is_connected:
            return None
        
        try:
            response = self._client.head_object(
                Bucket=self.config.s3_bucket_name,
                Key=file_path
            )
            
            info = {
                'path': file_path,
                'size': response['ContentLength'],
                'etag': response['ETag'].strip('"'),
                'last_modified': response['LastModified'].isoformat(),
                'extension': get_file_extension(file_path)
            }
            
            # Add custom metadata
            if 'Metadata' in response:
                info['metadata'] = response['Metadata']
            
            return info
        
        except Exception as e:
            logger.error(f"Error getting file info from S3 {file_path}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get S3 storage statistics."""
        stats = self._stats.copy()
        stats['is_connected'] = self._is_connected
        stats['bucket_name'] = self.config.s3_bucket_name
        
        if self._is_connected:
            try:
                files = self.list_files()
                stats['total_files'] = len(files)
                
                # Calculate total size (this could be expensive for large buckets)
                total_size = 0
                for file_path in files[:100]:  # Limit to first 100 files for performance
                    info = self.get_file_info(file_path)
                    if info:
                        total_size += info['size']
                
                stats['sample_total_size'] = total_size
                stats['sample_file_count'] = min(len(files), 100)
            except Exception:
                stats['total_files'] = 0
                stats['sample_total_size'] = 0
        
        return stats


class FileManager:
    """Multi-backend file manager with automatic fallback."""
    
    def __init__(self, config: FileConfig):
        self.config = config
        self._backends = []
        self._primary_backend = None
        self._stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'fallback_used': 0
        }
        
        self._initialize_backends()
    
    def _initialize_backends(self):
        """Initialize file storage backends in priority order."""
        try:
            # Primary: S3 storage (if configured)
            if S3_AVAILABLE and self.config.s3_bucket_name:
                s3_backend = S3StorageBackend(self.config)
                if s3_backend._is_connected:
                    self._backends.append(s3_backend)
                    self._primary_backend = s3_backend
                    logger.info("S3 storage backend enabled as primary")
            
            # Fallback: Local storage
            local_backend = LocalStorageBackend(self.config)
            self._backends.append(local_backend)
            
            if not self._primary_backend:
                self._primary_backend = local_backend
                logger.info("Local storage backend enabled as primary")
                
        except Exception as e:
            logger.error(f"Error initializing file storage backends: {e}")
    
    def save_json(self, file_path: str, data: Dict[str, Any], 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save JSON data to file."""
        try:
            json_data = json.dumps(data, default=str, indent=2)
            return self.save_file(file_path, json_data, metadata)
        except Exception as e:
            logger.error(f"Error saving JSON file {file_path}: {e}")
            return False
    
    def load_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data from file."""
        try:
            data = self.load_file(file_path)
            if data:
                return json.loads(data.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def save_csv(self, file_path: str, df: pd.DataFrame, 
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save DataFrame as CSV file."""
        try:
            csv_data = df.to_csv(index=False)
            return self.save_file(file_path, csv_data, metadata)
        except Exception as e:
            logger.error(f"Error saving CSV file {file_path}: {e}")
            return False
    
    def load_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load CSV file as DataFrame."""
        try:
            data = self.load_file(file_path)
            if data:
                from io import StringIO
                return pd.read_csv(StringIO(data.decode('utf-8')))
            return None
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return None
    
    def save_parquet(self, file_path: str, df: pd.DataFrame, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save DataFrame as Parquet file."""
        try:
            import io
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            return self.save_file(file_path, buffer.getvalue(), metadata)
        except Exception as e:
            logger.error(f"Error saving Parquet file {file_path}: {e}")
            return False
    
    def load_parquet(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load Parquet file as DataFrame."""
        try:
            data = self.load_file(file_path)
            if data:
                import io
                return pd.read_parquet(io.BytesIO(data))
            return None
        except Exception as e:
            logger.error(f"Error loading Parquet file {file_path}: {e}")
            return None
    
    def save_file(self, file_path: str, data: Union[bytes, str], 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save file with fallback to available backends."""
        self._stats['total_operations'] += 1
        
        for i, backend in enumerate(self._backends):
            try:
                if backend.save_file(file_path, data, metadata):
                    self._stats['successful_operations'] += 1
                    if i > 0:
                        self._stats['fallback_used'] += 1
                    return True
            except Exception as e:
                logger.warning(f"Storage backend {type(backend).__name__} failed: {e}")
                continue
        
        logger.error(f"All storage backends failed for file: {file_path}")
        return False
    
    def load_file(self, file_path: str) -> Optional[bytes]:
        """Load file with fallback to available backends."""
        self._stats['total_operations'] += 1
        
        for i, backend in enumerate(self._backends):
            try:
                data = backend.load_file(file_path)
                if data is not None:
                    self._stats['successful_operations'] += 1
                    if i > 0:
                        self._stats['fallback_used'] += 1
                    return data
            except Exception as e:
                logger.warning(f"Storage backend {type(backend).__name__} failed: {e}")
                continue
        
        return None
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from all available backends."""
        success = False
        
        for backend in self._backends:
            try:
                if backend.delete_file(file_path):
                    success = True
            except Exception as e:
                logger.warning(f"Storage backend {type(backend).__name__} delete failed: {e}")
                continue
        
        return success
    
    def file_exists(self, file_path: str) -> bool:
        """Check if file exists in any backend."""
        for backend in self._backends:
            try:
                if backend.file_exists(file_path):
                    return True
            except Exception as e:
                logger.warning(f"Storage backend {type(backend).__name__} exists check failed: {e}")
                continue
        
        return False
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files from primary backend."""
        if self._primary_backend:
            try:
                return self._primary_backend.list_files(prefix)
            except Exception as e:
                logger.warning(f"Primary backend list failed: {e}")
        
        # Fallback to other backends
        for backend in self._backends[1:]:
            try:
                return backend.list_files(prefix)
            except Exception as e:
                logger.warning(f"Backend {type(backend).__name__} list failed: {e}")
                continue
        
        return []
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get file information from available backends."""
        for backend in self._backends:
            try:
                info = backend.get_file_info(file_path)
                if info:
                    return info
            except Exception as e:
                logger.warning(f"Storage backend {type(backend).__name__} info failed: {e}")
                continue
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive file storage statistics."""
        stats = self._stats.copy()
        stats['success_rate'] = (
            stats['successful_operations'] / max(stats['total_operations'], 1) * 100
        )
        stats['backends'] = {}
        
        for backend in self._backends:
            backend_name = type(backend).__name__
            try:
                stats['backends'][backend_name] = backend.get_stats()
            except Exception:
                stats['backends'][backend_name] = {'error': 'Failed to get stats'}
        
        return stats


# Global file manager instance
_file_manager = None

def create_file_manager(config: Optional[FileConfig] = None) -> FileManager:
    """Create and initialize file manager."""
    global _file_manager
    
    if config is None:
        config = FileConfig.from_env()
    
    _file_manager = FileManager(config)
    return _file_manager

def get_file_storage() -> Optional[FileManager]:
    """Get global file manager instance."""
    return _file_manager