"""
Custom JSON encoder for handling datetime, enum, and other non-serializable objects
"""

import json
from datetime import datetime, date, time
from enum import Enum
from decimal import Decimal
from dataclasses import asdict, is_dataclass
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles datetime objects, enums, and other common types
    """
    
    def default(self, obj: Any) -> Any:
        """Convert non-serializable objects to JSON-serializable format"""
        
        # Handle datetime objects
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
        
        # Handle Enum objects
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle Decimal objects
        if isinstance(obj, Decimal):
            return float(obj)
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Handle bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        
        # For any other non-serializable object, convert to string
        try:
            # Try the default encoder first
            return super().default(obj)
        except TypeError:
            # If that fails, convert to string representation
            logger.warning(f"Converting non-serializable object {type(obj)} to string: {obj}")
            return str(obj)

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string using custom encoder
    """
    kwargs.setdefault('cls', CustomJSONEncoder)
    kwargs.setdefault('ensure_ascii', False)
    kwargs.setdefault('separators', (',', ':'))  # Compact format
    return json.dumps(obj, **kwargs)

def safe_json_dump(obj: Any, fp, **kwargs) -> None:
    """
    Safely serialize object to JSON file using custom encoder
    """
    kwargs.setdefault('cls', CustomJSONEncoder)
    kwargs.setdefault('ensure_ascii', False)
    kwargs.setdefault('indent', 2)  # Pretty format for files
    return json.dump(obj, fp, **kwargs)

def safe_asdict(obj: Any) -> Dict[str, Any]:
    """
    Safely convert dataclass to dictionary, handling enums and other non-serializable types
    """
    if is_dataclass(obj):
        result = {}
        for field_name, field_value in asdict(obj).items():
            if isinstance(field_value, Enum):
                result[field_name] = field_value.value
            elif isinstance(field_value, (datetime, date, time)):
                result[field_name] = field_value.isoformat()
            elif isinstance(field_value, list):
                result[field_name] = [
                    item.value if isinstance(item, Enum) else 
                    item.isoformat() if isinstance(item, (datetime, date, time)) else 
                    item for item in field_value
                ]
            else:
                result[field_name] = field_value
        return result
    else:
        # If not a dataclass, use the custom JSON encoder approach
        return json.loads(safe_json_dumps(obj))