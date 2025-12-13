"""
JSON sanitization utilities to handle NaN and Inf values.
"""
import math
from typing import Any, Dict, List, Union


def safe_float(value: Any) -> Union[float, None]:
    """
    Convert float to JSON-safe value (None if nan, inf, or None).
    
    Args:
        value: Value to convert
    
    Returns:
        Float value or None if invalid
    """
    if value is None:
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
    
    if isinstance(value, (int, float)):
        return safe_float(value)
    
    if isinstance(value, dict):
        return sanitize_dict(value)
    
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    
    # For other types (str, bool, etc.), return as-is
    return value

