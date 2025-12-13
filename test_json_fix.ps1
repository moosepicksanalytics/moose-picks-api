# Test script to verify JSON NaN fix
# This tests the predictions endpoint to ensure no NaN values in response

$baseUrl = "https://moose-picks-api-production.up.railway.app"

Write-Host "Testing JSON NaN Fix" -ForegroundColor Cyan
Write-Host "====================" -ForegroundColor Cyan
Write-Host ""

# Test predictions endpoint
Write-Host "1. Testing /api/predictions/latest (should not have NaN values)..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/predictions/latest?sport=NFL&limit=5" -Method GET
    
    # Check for NaN values in response (convert to JSON string and check)
    $jsonString = $response | ConvertTo-Json -Depth 10
    if ($jsonString -match "NaN" -or $jsonString -match "Infinity" -or $jsonString -match "-Infinity") {
        Write-Host "   ✗ ERROR: Found NaN or Infinity values in response!" -ForegroundColor Red
        Write-Host "   JSON contains invalid float values" -ForegroundColor Red
    } else {
        Write-Host "   ✓ No NaN/Infinity values found in response" -ForegroundColor Green
        Write-Host "   Response structure looks good" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "   Response summary:" -ForegroundColor Cyan
    Write-Host "   - Sport: $($response.sport)" -ForegroundColor Gray
    Write-Host "   - Source: $($response.source)" -ForegroundColor Gray
    Write-Host "   - Total predictions: $($response.total_predictions)" -ForegroundColor Gray
    Write-Host "   - Top picks returned: $($response.top_picks.Count)" -ForegroundColor Gray
    
    if ($response.top_picks.Count -gt 0) {
        Write-Host ""
        Write-Host "   Sample pick:" -ForegroundColor Cyan
        $sample = $response.top_picks[0]
        Write-Host "   - Game: $($sample.away_team) @ $($sample.home_team)" -ForegroundColor Gray
        Write-Host "   - Market: $($sample.market)" -ForegroundColor Gray
        Write-Host "   - Edge: $($sample.edge)" -ForegroundColor Gray
    }
} catch {
    Write-Host "   ✗ Error calling endpoint: $_" -ForegroundColor Red
    if ($_.Exception.Response.StatusCode -eq 500) {
        Write-Host "   → 500 error suggests JSON serialization issue still exists" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "2. Testing /api/predictions/next-days..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/api/predictions/next-days?sport=NHL&days=3" -Method GET
    
    $jsonString = $response | ConvertTo-Json -Depth 10
    if ($jsonString -match "NaN" -or $jsonString -match "Infinity") {
        Write-Host "   ✗ ERROR: Found NaN or Infinity values!" -ForegroundColor Red
    } else {
        Write-Host "   ✓ No NaN/Infinity values found" -ForegroundColor Green
    }
} catch {
    Write-Host "   ✗ Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "Test complete!" -ForegroundColor Cyan

