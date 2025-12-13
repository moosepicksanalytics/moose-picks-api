# Moose Picks API - Deployment Guide

## Quick Start: Railway Deployment

### 1. Environment Variables (Railway Dashboard)

Set these in Railway project settings:

#### Required Variables:

```bash
# Security Configuration
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://lovable.app
API_KEYS=your-secret-api-key-1,your-secret-api-key-2

# Data Sources
ODDS_API_KEY=your-odds-api-key

# Database (auto-set by Railway)
DATABASE_URL=postgresql://... (automatically provided)
```

#### Optional Configuration:

```bash
# Rate Limiting (defaults shown)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60

# Model Directory (defaults shown)
MODEL_DIR=models/
```

### 2. Generate API Keys

Generate secure API keys:

```powershell
# PowerShell: Generate random API key
$apiKey = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
Write-Host "API Key: $apiKey"
```

Or use online generator: https://randomkeygen.com/

### 3. Test API Endpoints

Use the provided PowerShell test script:

```powershell
# Test health endpoint (no auth required)
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/health"

# Test with authentication
$headers = @{
    "X-API-Key" = "your-api-key-here"
    "Content-Type" = "application/json"
}
Invoke-RestMethod -Uri "https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NFL" -Headers $headers

# Run full test suite
.\test_api_powershell.ps1 -ApiKey "your-api-key" -BaseUrl "https://moose-picks-api-production.up.railway.app"
```

### 4. Frontend Integration

#### Making Authenticated Requests

```javascript
// JavaScript/TypeScript example
const response = await fetch('https://moose-picks-api-production.up.railway.app/api/predictions/latest?sport=NFL', {
  headers: {
    'X-API-Key': 'your-api-key-here',
    'Content-Type': 'application/json'
  }
});

const data = await response.json();
```

#### CORS Configuration

Make sure your frontend domain is in `ALLOWED_ORIGINS`:
- Development: `ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173`
- Production: `ALLOWED_ORIGINS=https://yourdomain.com`

### 5. Protected Endpoints

These endpoints require API key authentication:

- `POST /api/trigger-daily-workflow` - Run daily automation
- `POST /api/backfill` - Backfill historical data
- `POST /api/backfill-odds` - Backfill odds data
- `POST /api/train` - Train models
- `POST /api/migrate-ou-columns` - Database migration
- `POST /api/backfill-ou-data` - Backfill O/U data

#### Public Endpoints (No Auth Required):

- `GET /health` - Health check
- `GET /api/predictions/latest` - Get latest predictions
- `GET /api/predictions/next-days` - Get predictions for next N days
- `GET /api/predictions/date-range` - Get predictions for date range
- `GET /api/predictions/week` - Get predictions for current week
- `GET /api/validate-ou-coverage` - Validate O/U data coverage

### 6. Rate Limiting

- Default: 60 requests per minute per IP address
- Configure via `RATE_LIMIT_PER_MINUTE` environment variable
- Rate limit headers in response:
  - `X-RateLimit-Limit`: Maximum requests per window
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset time (Unix timestamp)

### 7. Error Handling

#### Authentication Errors

```json
// 401 Unauthorized (missing API key)
{
  "detail": "API key required. Provide X-API-Key header."
}

// 403 Forbidden (invalid API key)
{
  "detail": "Invalid API key"
}
```

#### Rate Limit Errors

```json
// 429 Too Many Requests
{
  "detail": "Rate limit exceeded. Maximum 60 requests per minute."
}
```

### 8. Monitoring

Check Railway logs for:
- API authentication failures
- Rate limit violations
- Database connection issues
- External API (Odds API) quota usage

### 9. Security Best Practices

1. **Never commit API keys to Git**
   - Use Railway environment variables only
   - Rotate keys periodically

2. **Restrict CORS Origins**
   - Never use `ALLOWED_ORIGINS=*` in production
   - Whitelist specific domains only

3. **Use Strong API Keys**
   - Minimum 32 characters
   - Mix of letters, numbers, symbols
   - Different keys for different environments

4. **Monitor Rate Limits**
   - Watch for unusual traffic patterns
   - Adjust `RATE_LIMIT_PER_MINUTE` as needed

### 10. Troubleshooting

#### "401 Unauthorized" Errors
- Check `X-API-Key` header is included
- Verify API key is in `API_KEYS` environment variable
- Check for typos in API key

#### "429 Too Many Requests" Errors
- Reduce request frequency
- Increase `RATE_LIMIT_PER_MINUTE` if needed
- Check for duplicate requests in frontend code

#### CORS Errors
- Verify frontend domain is in `ALLOWED_ORIGINS`
- Check for trailing slashes in URLs
- Ensure `Content-Type` header matches request body

#### Database Connection Errors
- Verify `DATABASE_URL` is set correctly
- Check Railway database is not paused
- Review connection pool settings

---

**Last Updated:** December 2024  
**Version:** 1.0.0

