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

**File:** `requirements.txt`
- `psycopg2-binary` → `psycopg[binary]` (psycopg3)

**File:** `app/database.py`
- Added `get_database_url()` function to convert `postgresql://` → `postgresql+psycopg://`
- This ensures SQLAlchemy uses psycopg3 instead of defaulting to psycopg2
- Works seamlessly with Railway's `postgresql://` connection strings

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

**Important:** SQLAlchemy 2.0.44 **defaults to `psycopg2`** for `postgresql://` URLs.

To use `psycopg3`, you must:
1. Install `psycopg[binary]` (done in requirements.txt)
2. Use `postgresql+psycopg://` in the connection string (done in database.py)

**Why the conversion is needed:**
- Railway provides: `postgresql://user:pass@host:port/db`
- SQLAlchemy sees `postgresql://` → tries to use `psycopg2` (not installed)
- Our code converts to: `postgresql+psycopg://user:pass@host:port/db`
- SQLAlchemy sees `postgresql+psycopg://` → uses `psycopg3` (installed) ✅

**Connection string formats:**
- `postgresql://` → Uses psycopg2 (default, not installed)
- `postgresql+psycopg://` → Uses psycopg3 (explicit, installed) ✅
- `sqlite://` → Uses sqlite3 (unchanged)

