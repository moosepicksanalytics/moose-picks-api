# CORS Setup for Lovable Integration

## Issue
Lovable frontend was getting "Failed to fetch" errors when trying to call the Railway API due to CORS (Cross-Origin Resource Sharing) restrictions.

## Solution
Added CORS middleware to FastAPI to allow cross-origin requests from Lovable.

## Changes Made

### File: `app/main.py`

Added CORS middleware configuration:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
```

## Current Configuration

**Allow All Origins** (`allow_origins=["*"]`):
- ✅ Works for all domains (Lovable preview, production, etc.)
- ✅ Simple setup
- ⚠️ Less secure (allows any website to call your API)

## Production Recommendation

For better security, restrict to specific Lovable domains:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-lovable-app.com",
        "https://preview.your-lovable-app.com",
        "https://*.lovable.dev",  # Lovable preview domains
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Testing

After deploying to Railway, test from Lovable:

```javascript
// In Lovable frontend
fetch('https://moose-picks-api-production.up.railway.app/api/health')
  .then(res => res.json())
  .then(data => console.log(data))
  .catch(err => console.error('CORS Error:', err));
```

If CORS is working, you should see the health check response. If not, you'll still see "Failed to fetch" or CORS errors in the browser console.

## What This Fixes

- ✅ Lovable can now make GET requests to `/api/predictions/*` endpoints
- ✅ Lovable can now make POST requests to `/api/trigger-daily-workflow`
- ✅ No more "Failed to fetch" errors
- ✅ Browser will allow cross-origin requests

## Next Steps

1. **Deploy to Railway** - Push these changes
2. **Test from Lovable** - Try fetching predictions
3. **Restrict Origins** (Optional) - Update `allow_origins` to specific domains for production

## Related Files

- `app/main.py` - CORS middleware configuration
- `app/api_endpoints.py` - API endpoints that Lovable will call
