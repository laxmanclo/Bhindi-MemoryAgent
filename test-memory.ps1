# PowerShell Script to Test MemoryOS Memory Integration
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "MemoryOS Memory Integration Test" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan

$baseUrl = "http://localhost:3000/tools"

# Step 1: Create a new chat
Write-Host "`nStep 1: Creating a new chat..." -ForegroundColor Yellow
$createChatBody = '{"chatName": "Integration Test Chat"}'
try {
    $createChatResponse = Invoke-RestMethod -Uri "$baseUrl/createNewChat" -Method Post -ContentType "application/json" -Body $createChatBody
    Write-Host "Chat created successfully!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Green
    $createChatResponse | ConvertTo-Json -Depth 10
}
catch {
    Write-Host "Failed to create chat: $_" -ForegroundColor Red
    exit
}

# Step 2: Add a memory
Write-Host "`nStep 2: Adding a memory..." -ForegroundColor Yellow
$addMemoryBody = '{"userInput": "My name is Integration Tester", "agentResponse": "Hello Integration Tester, nice to meet you!"}'
try {
    $addMemoryResponse = Invoke-RestMethod -Uri "$baseUrl/addMemory" -Method Post -ContentType "application/json" -Body $addMemoryBody
    Write-Host "Memory added successfully!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Green
    $addMemoryResponse | ConvertTo-Json -Depth 10
}
catch {
    Write-Host "Failed to add memory: $_" -ForegroundColor Red
    exit
}

# Step 3: Query the memory about name
Write-Host "`nStep 3: Querying memory about name..." -ForegroundColor Yellow
$queryMemoryBody = '{"query": "What is my name?"}'
try {
    $queryMemoryResponse = Invoke-RestMethod -Uri "$baseUrl/queryMemory" -Method Post -ContentType "application/json" -Body $queryMemoryBody
    Write-Host "Memory queried successfully!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Green
    $queryMemoryResponse | ConvertTo-Json -Depth 10
}
catch {
    Write-Host "Failed to query memory: $_" -ForegroundColor Red
    exit
}

# Summary
Write-Host "`n=================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "Integration test completed!" -ForegroundColor Green
