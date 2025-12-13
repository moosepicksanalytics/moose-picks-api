# JSON NaN Serialization Bug Fix

## Issue
API endpoints were returning 500 Internal Server Error:
```
ValueError: Out of range float values are not JSON compliant: nan
```

## Root Cause
Edge calculation functions could return NaN (Not a Number) or Inf (Infinity) values in dictionaries. Python's JSON encoder cannot serialize these values.

## Solution
Created comprehensive JSON sanitization utilities to handle NaN/Inf values:

1. **`app/utils/json_sanitize.py`** - New utility module
   - `safe_float()`: Converts NaN/Inf to None (handles both Python and numpy NaN)
   - `sanitize_dict()`: Recursively sanitizes dictionaries
   - `sanitize_value()`: Sanitizes individual values

2. **Updated `app/api_endpoints.py`**:
   - All edge calculation results sanitized before use
   - CSV fallback path also sanitizes data
   - Final sanitization pass on all picks before returning JSON

## Changes Made

### Files Created:
- `app/utils/json_sanitize.py` - JSON sanitization utilities

### Files Modified:
- `app/api_endpoints.py` - Added sanitization to prediction endpoints

## Testing

```powershell
# Test the fix
.\test_api_debug.ps1 -Sport NFL

# Or test manually
$baseUrl = "https://moose-picks-api-production.up.railway.app"
Invoke-RestMethod -Uri "$baseUrl/api/predictions/latest?sport=NFL&limit=5"
```

Expected: 200 OK response with valid JSON (no NaN/Inf values)

## Technical Details

The sanitization handles:
- Python `float('nan')` and `float('inf')`
- Numpy `np.nan` and `np.inf`
- Numpy scalar types (np.floating, np.generic)
- Pandas DataFrame NaN values (converted to numpy scalars in dict)

All NaN/Inf values are converted to `None`, which JSON serializes as `null`.

## Status
✅ **FIXED** - All NaN/Inf values are now sanitized before JSON serialization.

