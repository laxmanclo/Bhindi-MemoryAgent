# PowerShell Script to Clean MemoryOS Data
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "MemoryOS Data Cleanup Utility" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

# Define the data path
$dataPath = "memoryos_data"
$chatHistoryFile = Join-Path $dataPath "chat_histories.json"

# Create a clean chat history structure
$cleanChatHistories = @{
    "chats" = @{}
    "current_chat_id" = $null
} | ConvertTo-Json -Depth 4

# Check if data directory exists
if (Test-Path $dataPath) {
    Write-Host "Data directory found at: $dataPath" -ForegroundColor Green
    
    # Check if chat history file exists
    if (Test-Path $chatHistoryFile) {
        Write-Host "Backing up existing chat history to ${chatHistoryFile}.bak" -ForegroundColor Yellow
        Copy-Item -Path $chatHistoryFile -Destination "${chatHistoryFile}.bak" -Force
        
        # Write clean chat history structure
        Write-Host "Clearing chat history..." -ForegroundColor Yellow
        Set-Content -Path $chatHistoryFile -Value $cleanChatHistories -Force
        
        Write-Host "Chat history cleared successfully!" -ForegroundColor Green
    } else {
        Write-Host "Chat history file not found. Creating new empty file." -ForegroundColor Yellow
        # Create data directory if it doesn't exist
        if (-not (Test-Path $dataPath)) {
            New-Item -ItemType Directory -Path $dataPath -Force | Out-Null
        }
        
        # Write clean chat history structure
        Set-Content -Path $chatHistoryFile -Value $cleanChatHistories -Force
        Write-Host "Empty chat history created successfully!" -ForegroundColor Green
    }
} else {
    Write-Host "Data directory not found at: $dataPath" -ForegroundColor Red
    Write-Host "Creating data directory and empty chat history..." -ForegroundColor Yellow
    
    # Create data directory
    New-Item -ItemType Directory -Path $dataPath -Force | Out-Null
    
    # Write clean chat history structure
    Set-Content -Path $chatHistoryFile -Value $cleanChatHistories -Force
    Write-Host "Empty chat history created successfully!" -ForegroundColor Green
}

Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "Cleanup Complete!" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "To apply changes, restart the MemoryOS services." -ForegroundColor Yellow
