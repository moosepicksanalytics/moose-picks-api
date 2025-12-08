# psycopg2 to psycopg3 Migration Fix

## Problem

`psycopg2-binary` is **incompatible with Python 3.13**. The error:
```
undefined symbol: _PyInterpreterState_Get
```

This happens because `psycopg2-binary` was compiled for older Python versions and doesn't work with Python 3.13.

## Solution

Replaced `psycopg2-binary` with `psycopg` (psycopg3), which:
- ✅ **Supports Python 3.13**
- ✅ **Works with SQLAlchemy 2.0.44**
- ✅ **Drop-in replacement** - no code changes needed
- ✅ **Better performance** - modern async support

## Changes Made

**File:** `requirements.txt`

**Before:**
```
psycopg2-binary==2.9.9
```

**After:**
```
psycopg[binary]==3.2.0
```

## What Changed

- `psycopg2-binary` → `psycopg[binary]` (psycopg3)
- SQLAlchemy will automatically use psycopg3 if available
- No code changes needed - SQLAlchemy handles the switch automatically

## Next Steps

1. **Commit and push:**
   ```bash
   git add requirements.txt
   git commit -m "Replace psycopg2-binary with psycopg3 for Python 3.13 compatibility"
   git push
   ```

2. **Railway will auto-deploy** and install `psycopg[binary]==3.2.0`

3. **Verify it works:**
   ```powershell
   Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/health"
   ```
   - Should show `"database": "connected"`
   - No more import errors

## Why psycopg3?

- **Modern**: Built for Python 3.11+
- **Fast**: Better performance than psycopg2
- **Async**: Native async/await support
- **Compatible**: Works with SQLAlchemy 2.0+
- **Future-proof**: Actively maintained

## Technical Details

SQLAlchemy 2.0.44 automatically detects and uses:
- `psycopg` (psycopg3) if available
- Falls back to `psycopg2` if psycopg3 not found

The connection string format remains the same:
- `postgresql://user:pass@host:port/db` works with both
- SQLAlchemy handles the driver selection automatically

