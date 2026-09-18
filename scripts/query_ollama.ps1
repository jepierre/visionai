param(
    [string]$ImagePath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'images\dog_running_in_park.jpg'),
    [string]$Model = 'gemma4:e4b',
    [string]$Prompt = 'Describe this image in detail. Mention the setting, the main subject, its action, and notable visual details.',
    [string]$OllamaUrl = 'http://localhost:11434'
)

$ErrorActionPreference = 'Stop'

try {
    $resolvedImagePath = (Resolve-Path -LiteralPath $ImagePath -ErrorAction Stop).Path
    $bytes = [System.IO.File]::ReadAllBytes($resolvedImagePath)
    $base64 = [Convert]::ToBase64String($bytes)
    $body = @{
        model = $Model
        prompt = $Prompt
        images = @($base64)
        stream = $false
    } | ConvertTo-Json -Depth 5

    $response = Invoke-RestMethod `
        -Uri "$OllamaUrl/api/generate" `
        -Method Post `
        -ContentType 'application/json' `
        -Body $body

    if ($null -eq $response.response) {
        throw "MODEL_ERROR: Ollama returned no response."
    }

    Write-Output $response.response
}
catch {
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $detail = $reader.ReadToEnd()
        $reader.Dispose()
        Write-Error ("HTTP_ERROR: {0} {1}`n{2}" -f $_.Exception.Response.StatusCode.value__, $_.Exception.Response.StatusDescription, $detail)
    }
    else {
        Write-Error $_.Exception.Message
    }
    exit 1
}