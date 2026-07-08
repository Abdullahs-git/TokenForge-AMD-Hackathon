# cleanup.ps1
# Delete the input/ folder contents
if (Test-Path "input") {
    Remove-Item -Path "input\*" -Recurse -Force -ErrorAction SilentlyContinue
}

# Delete the output/ folder contents
if (Test-Path "output") {
    Remove-Item -Path "output\*" -Recurse -Force -ErrorAction SilentlyContinue
}

# Delete __pycache__ folders
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Delete the .env file
if (Test-Path ".env") {
    Remove-Item -Path ".env" -Force -ErrorAction SilentlyContinue
}

# Delete setup_test.py
if (Test-Path "setup_test.py") {
    Remove-Item -Path "setup_test.py" -Force -ErrorAction SilentlyContinue
}

# Print completion message
Write-Output "Cleanup complete! Repository is ready for final push."
