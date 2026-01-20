# Run Streamlit App Script
Write-Host "🛡️ Starting Hate Speech Detection App..." -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is installed
try {
    $streamlitVersion = streamlit --version 2>&1
    Write-Host "✓ Streamlit found: $streamlitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Streamlit not found. Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host ""
Write-Host "Starting application..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the Streamlit app
streamlit run app.py
