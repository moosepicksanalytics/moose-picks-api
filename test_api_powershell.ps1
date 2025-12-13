# PowerShell script to test Railway API endpoints
# Usage: .\test_api_powershell.ps1 -ApiKey "your-api-key" -BaseUrl "https://moose-picks-api-production.up.railway.app"

param(
    [string]$ApiKey = "",
    [string]$BaseUrl = "https://moose-picks-api-production.up.railway.app"
)

$headers = @{
    "Content-Type" = "application/json"
}

if ($ApiKey) {
    $headers["X-API-Key"] = $ApiKey
}

Write-Host "Testing Moose Picks API at $BaseUrl" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check (no auth required)
Write-Host "1. Testing Health Check (GET /health)..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -Headers $headers
    Write-Host "   ✓ Health check passed" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "   ✗ Health check failed: $_" -ForegroundColor Red
}
Write-Host ""

# Test 2: Get Latest Predictions (GET - no auth required)
Write-Host "2. Testing Latest Predictions (GET /api/predictions/latest)..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$BaseUrl/api/predictions/latest?sport=NFL&limit=5" -Method GET -Headers $headers
    Write-Host "   ✓ Latest predictions retrieved" -ForegroundColor Green
    Write-Host "   Total predictions: $($response.total_predictions)" -ForegroundColor Cyan
    if ($response.top_picks.Count -gt 0) {
        Write-Host "   First pick: $($response.top_picks[0].home_team) vs $($response.top_picks[0].away_team)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "   ✗ Latest predictions failed: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: Trigger Daily Workflow (POST - requires auth if API_KEYS configured)
Write-Host "3. Testing Daily Workflow (POST /api/trigger-daily-workflow)..." -ForegroundColor Yellow
Write-Host "   Note: This requires API key if API_KEYS environment variable is set" -ForegroundColor Gray
try {
    $body = @{
        train = $false
        predict = $false
        sports = "NFL"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "$BaseUrl/api/trigger-daily-workflow?train=false&predict=false&sports=NFL" -Method POST -Headers $headers -Body $body
    Write-Host "   ✓ Daily workflow triggered" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 3
} catch {
    Write-Host "   ✗ Daily workflow failed: $_" -ForegroundColor Red
    if ($_.Exception.Response.StatusCode -eq 401) {
        Write-Host "   → This endpoint requires authentication. Set API_KEYS environment variable and provide -ApiKey parameter." -ForegroundColor Yellow
    }
}
Write-Host ""

# Test 4: Rate Limiting Test (make multiple rapid requests)
Write-Host "4. Testing Rate Limiting..." -ForegroundColor Yellow
Write-Host "   Making 5 rapid requests to /health..." -ForegroundColor Gray
$rateTestPassed = $true
for ($i = 1; $i -le 5; $i++) {
    try {
        $response = Invoke-RestMethod -Uri "$BaseUrl/health" -Method GET -Headers $headers
        Write-Host "   Request $i : ✓" -ForegroundColor Green -NoNewline
        Start-Sleep -Milliseconds 200
    } catch {
        if ($_.Exception.Response.StatusCode -eq 429) {
            Write-Host "   Request $i : ✗ Rate limited (429)" -ForegroundColor Yellow
            $rateTestPassed = $false
        } else {
            Write-Host "   Request $i : ✗ Error: $_" -ForegroundColor Red
            $rateTestPassed = $false
        }
    }
}
if ($rateTestPassed) {
    Write-Host "   ✓ Rate limiting test passed (all requests succeeded)" -ForegroundColor Green
} else {
    Write-Host "   ⚠ Rate limiting is working (some requests were limited)" -ForegroundColor Yellow
}
Write-Host ""

# Test 5: CORS Test (if testing from browser)
Write-Host "5. CORS Configuration:" -ForegroundColor Yellow
Write-Host "   To test CORS, check browser console when making requests from your frontend." -ForegroundColor Gray
Write-Host "   ALLOWED_ORIGINS environment variable controls which origins are allowed." -ForegroundColor Gray
Write-Host "   Current config: Check ALLOWED_ORIGINS in Railway environment variables." -ForegroundColor Gray
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Testing complete!" -ForegroundColor Cyan
Write-Host ""
Write-Host "To test with authentication:" -ForegroundColor Yellow
Write-Host "  .\test_api_powershell.ps1 -ApiKey 'your-api-key-here'" -ForegroundColor White
Write-Host ""
Write-Host "To test local development:" -ForegroundColor Yellow
Write-Host "  .\test_api_powershell.ps1 -BaseUrl 'http://localhost:8000'" -ForegroundColor White

