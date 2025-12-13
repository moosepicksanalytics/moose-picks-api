"""
JSON sanitization utilities to handle NaN and Inf values.
"""
import math
from typing import Any, Dict, List, Union

# Try to import numpy for numpy NaN detection
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def safe_float(value: Any) -> Union[float, None]:
    """
    Convert float to JSON-safe value (None if nan, inf, or None).
    Handles both Python float NaN/Inf and numpy NaN/Inf.
    
    Args:
        value: Value to convert
    
    Returns:
        Float value or None if invalid
    """
    if value is None:
        return None
    
    # Handle numpy NaN/Inf if numpy is available
    if NUMPY_AVAILABLE:
        try:
            # Check if it's a numpy scalar type
            if isinstance(value, np.generic):
                # For floating point types, check for NaN/Inf
                if isinstance(value, (np.floating, np.complexfloating)):
                    if np.isnan(value) or np.isinf(value):
                        return None
                # Convert numpy scalar to Python float
                return float(value)
            # Also check regular floats that might be numpy NaN/Inf values
            elif isinstance(value, float):
                # This will be handled by the math.isnan check below
                pass
        except (ValueError, TypeError, OverflowError):
            # If conversion fails, treat as invalid
            return None
    
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def sanitize_dict(d: Dict) -> Dict:
    """
    Recursively sanitize a dictionary, converting NaN/Inf to None.
    
    Args:
        d: Dictionary to sanitize
    
    Returns:
        Sanitized dictionary
    """
    if not isinstance(d, dict):
        return d
    
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = sanitize_dict(v)
        elif isinstance(v, list):
            result[k] = [sanitize_value(item) for item in v]
        else:
            result[k] = sanitize_value(v)
    return result


def sanitize_value(value: Any) -> Any:
    """
    Sanitize a single value for JSON serialization.
    
    Args:
        value: Value to sanitize
    
    Returns:
        Sanitized value
    """
    if value is None:
        return None
    
    # Handle numpy types first (before float check)
    if NUMPY_AVAILABLE:
        try:
            if isinstance(value, np.generic):
                # Check for NaN/Inf in numpy scalars
                if isinstance(value, (np.floating, np.complexfloating)):
                    if np.isnan(value) or np.isinf(value):
                        return None
                # Convert numpy scalar to Python native type
                return safe_float(float(value))
        except (ValueError, TypeError, OverflowError):
            # If conversion fails, treat as invalid
            return None
    
    if isinstance(value, (int, float)):
        return safe_float(value)
    
    if isinstance(value, dict):
        return sanitize_dict(value)
    
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    
    # For other types (str, bool, etc.), return as-is
    return value

