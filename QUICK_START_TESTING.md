# Quick Start: Testing Railway API

## 1. Activate Virtual Environment

```powershell
# Navigate to project directory
cd C:\Users\mhess\Coding\moose-picks-api

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Verify activation (should show (venv) in prompt)
python --version
```

## 2. Set Environment Variables (Local Testing)

```powershell
# For local testing (development mode - no auth required)
$env:ALLOWED_ORIGINS = "*"
# Leave API_KEYS empty for development

# Or for production-like testing
$env:ALLOWED_ORIGINS = "https://yourdomain.com,http://localhost:3000"
$env:API_KEYS = "test-key-123,test-key-456"
```

## 3. Test API Endpoints

### Health Check (No Auth Required)

```powershell
$baseUrl = "https://moose-picks-api-production.up.railway.app"

# Simple health check
Invoke-RestMethod -Uri "$baseUrl/health"

# Expected response:
# {
#   "status": "ok",
#   "version": "0.1"
# }
```

### Get Predictions (No Auth Required)

```powershell
# Get latest NFL predictions
Invoke-RestMethod -Uri "$baseUrl/api/predictions/latest?sport=NFL&limit=5"

# Get predictions for next 3 days
Invoke-RestMethod -Uri "$baseUrl/api/predictions/next-days?sport=NHL&days=3"

# Get predictions for date range
Invoke-RestMethod -Uri "$baseUrl/api/predictions/date-range?sport=NFL&start_date=2024-12-15&end_date=2024-12-21"
```

### Protected Endpoints (Require API Key)

```powershell
# Set your API key (get from Railway environment variables)
$apiKey = "your-api-key-here"

$headers = @{
    "X-API-Key" = $apiKey
    "Content-Type" = "application/json"
}

# Trigger daily workflow (no training/prediction, just test)
Invoke-RestMethod -Uri "$baseUrl/api/trigger-daily-workflow?train=false&predict=false&sports=NFL" -Method POST -Headers $headers

# Expected response:
# {
#   "status": "started",
#   "message": "Daily workflow started in background",
#   "sports": ["NFL"],
#   "train": false,
#   "predict": false
# }
```

### Test Authentication Failures

```powershell
# Test without API key (should fail)
try {
    Invoke-RestMethod -Uri "$baseUrl/api/trigger-daily-workflow" -Method POST
} catch {
    Write-Host "Expected error: $($_.Exception.Message)" -ForegroundColor Yellow
    # Expected: 401 Unauthorized
}

# Test with invalid API key (should fail)
$badHeaders = @{
    "X-API-Key" = "invalid-key"
}
try {
    Invoke-RestMethod -Uri "$baseUrl/api/trigger-daily-workflow" -Method POST -Headers $badHeaders
} catch {
    Write-Host "Expected error: $($_.Exception.Message)" -ForegroundColor Yellow
    # Expected: 403 Forbidden
}
```

## 4. Run Comprehensive Test Suite

```powershell
# Run the automated test script
.\test_api_powershell.ps1 -ApiKey "your-api-key" -BaseUrl $baseUrl

# Without API key (development mode)
.\test_api_powershell.ps1 -BaseUrl $baseUrl
```

## 5. Test Rate Limiting

```powershell
# Make rapid requests to test rate limiting
$apiKey = "your-api-key"
$headers = @{"X-API-Key" = $apiKey}

Write-Host "Making 65 rapid requests (limit is 60/min)..." -ForegroundColor Cyan

$rateLimited = 0
$successful = 0

1..65 | ForEach-Object {
    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/health" -Headers $headers
        $successful++
        Write-Host "Request $_ : ✓" -ForegroundColor Green -NoNewline
    } catch {
        if ($_.Exception.Response.StatusCode -eq 429) {
            $rateLimited++
            Write-Host "Request $_ : ✗ Rate Limited (429)" -ForegroundColor Yellow -NoNewline
        } else {
            Write-Host "Request $_ : ✗ Error" -ForegroundColor Red -NoNewline
        }
    }
    Write-Host ""
    Start-Sleep -Milliseconds 50  # Small delay between requests
}

Write-Host ""
Write-Host "Results:" -ForegroundColor Cyan
Write-Host "  Successful: $successful" -ForegroundColor Green
Write-Host "  Rate Limited: $rateLimited" -ForegroundColor Yellow
```

## 6. Verify Environment Variables in Railway

```powershell
# Check Railway CLI (if installed)
railway variables

# Or check Railway dashboard:
# 1. Go to your Railway project
# 2. Click on your service
# 3. Go to "Variables" tab
# 4. Verify these are set:
#    - ALLOWED_ORIGINS
#    - API_KEYS
#    - ODDS_API_KEY
#    - DATABASE_URL (auto-set)
```

## 7. Test CORS (Browser Console)

Open browser console on your frontend and test:

```javascript
// Test CORS from allowed origin
fetch('https://moose-picks-api-production.up.railway.app/health')
  .then(r => r.json())
  .then(data => console.log('Success:', data))
  .catch(err => console.error('CORS Error:', err));

// Should work if your domain is in ALLOWED_ORIGINS
// Should fail with CORS error if not
```

## 8. Generate API Key (If Needed)

```powershell
# Generate a random API key
function Generate-ApiKey {
    $length = 32
    $chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    $key = -join ((1..$length) | ForEach-Object { $chars[(Get-Random -Maximum $chars.Length)] })
    return $key
}

$newKey = Generate-ApiKey
Write-Host "Generated API Key: $newKey" -ForegroundColor Green
Write-Host ""
Write-Host "Add this to Railway environment variable API_KEYS:" -ForegroundColor Yellow
Write-Host "API_KEYS=$newKey" -ForegroundColor White
```

## Common Issues

### "401 Unauthorized"
- **Cause:** API key missing or invalid
- **Fix:** Add `X-API-Key` header with valid key from Railway `API_KEYS`

### "403 Forbidden"
- **Cause:** Invalid API key
- **Fix:** Check API key matches one in Railway `API_KEYS` (comma-separated)

### "429 Too Many Requests"
- **Cause:** Rate limit exceeded (60 req/min default)
- **Fix:** Reduce request frequency or increase `RATE_LIMIT_PER_MINUTE`

### "CORS Error" (in browser)
- **Cause:** Frontend domain not in `ALLOWED_ORIGINS`
- **Fix:** Add your domain to Railway `ALLOWED_ORIGINS` (comma-separated)

### Authentication Not Working
- **Cause:** `API_KEYS` environment variable not set (development mode)
- **Fix:** Set `API_KEYS` in Railway for production, or test without auth in development

---

**Need Help?** Check `DEPLOYMENT_GUIDE.md` for detailed deployment instructions.

