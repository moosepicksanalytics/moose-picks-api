# Debug script to get detailed error information from API
param(
    [string]$BaseUrl = "https://moose-picks-api-production.up.railway.app",
    [string]$Sport = "NFL"
)

Write-Host "API Debug Test" -ForegroundColor Cyan
Write-Host "==============" -ForegroundColor Cyan
Write-Host ""

# Test with detailed error handling
try {
    Write-Host "Testing: GET $BaseUrl/api/predictions/latest?sport=$Sport&limit=5" -ForegroundColor Yellow
    
    $response = Invoke-RestMethod -Uri "$BaseUrl/api/predictions/latest?sport=$Sport&limit=5" -Method GET -ErrorAction Stop
    
    Write-Host "✓ Success!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Cyan
    $response | ConvertTo-Json -Depth 10 | Write-Host
    
} catch {
    Write-Host "✗ Error occurred" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error Details:" -ForegroundColor Yellow
    Write-Host "  Message: $($_.Exception.Message)" -ForegroundColor White
    Write-Host "  Status: $($_.Exception.Response.StatusCode.value__)" -ForegroundColor White
    
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host ""
        Write-Host "Response Body:" -ForegroundColor Yellow
        Write-Host $responseBody -ForegroundColor White
        
        try {
            $errorJson = $responseBody | ConvertFrom-Json
            Write-Host ""
            Write-Host "Parsed Error:" -ForegroundColor Yellow
            $errorJson | ConvertTo-Json -Depth 5 | Write-Host
        } catch {
            Write-Host "  (Could not parse error as JSON)" -ForegroundColor Gray
        }
    }
    
    Write-Host ""
    Write-Host "Stack Trace:" -ForegroundColor Yellow
    Write-Host $_.Exception.StackTrace -ForegroundColor Gray
}

Write-Host ""
Write-Host "Test complete!" -ForegroundColor Cyan

