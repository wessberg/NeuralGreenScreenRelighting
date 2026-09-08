# Reproduce this example only. Other footage requires the per-shot adaptation described in README.
param([switch]$SkipKey, [switch]$SkipScene)
$ErrorActionPreference = 'Stop'
$vpRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $vpRoot
$vpPython = Join-Path $vpRoot 'dependencies/CorridorKey/.venv/Scripts/python.exe'
$vpFfmpeg = Get-ChildItem -LiteralPath (Join-Path $vpRoot 'dependencies/ffmpeg') -Recurse -Filter ffmpeg.exe | Select-Object -First 1 -ExpandProperty FullName
$vpBlender = Get-ChildItem -Path 'C:/Program Files/Blender Foundation/*/blender.exe' | Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
if (!(Test-Path -LiteralPath $vpPython)) { throw 'Run the dependency setup recorded in README first.' }
function Invoke-Stage([string]$vpScript, [string[]]$vpArguments = @()) {
    & $vpPython $vpScript @vpArguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $vpScript" }
}
if (!(Test-Path -LiteralPath 'source/frames/000299.png')) {
    New-Item -ItemType Directory -Path source/frames -Force | Out-Null
    & $vpFfmpeg -hide_banner -y -i source/plate.mp4 -frames:v 300 -pix_fmt rgb48be -start_number 0 source/frames/%06d.png
    if ($LASTEXITCODE -ne 0) { throw 'Source extraction failed' }
}
if (!$SkipKey) { Invoke-Stage 'scripts/key_plate.py' }
Invoke-Stage 'scripts/video_depth.py'
$vpMcpProcess = $null
try {
    if (!$SkipScene) {
        # Isolated background add-on instance, avoids taking over an interactive Blender session.
        $env:VP_MCP_PORT = '9886'
        $vpMcpProcess = Start-Process -FilePath $vpBlender -ArgumentList @('-b','--online-mode','--command','blender_mcp','--host','127.0.0.1','--port','9886') -WorkingDirectory $vpRoot -WindowStyle Hidden -RedirectStandardOutput 'logs/reproduce_mcp_server.log' -RedirectStandardError 'logs/reproduce_mcp_server_errors.log' -PassThru
        $vpReady = $false
        for ($vpAttempt = 0; $vpAttempt -lt 20; $vpAttempt++) {
            Start-Sleep -Milliseconds 500
            $vpClient = [System.Net.Sockets.TcpClient]::new()
            try { $vpClient.Connect('127.0.0.1',9886); $vpReady = $true } catch {} finally { $vpClient.Dispose() }
            if ($vpReady) { break }
        }
        if (!$vpReady) { throw 'Blender MCP add-on did not start; see logs/reproduce_mcp_server_errors.log.' }
        Invoke-Stage 'scripts/blender_mcp_client.py' @('--script','scripts/build_scene.py')
    }
    Invoke-Stage 'scripts/relight_controls.py'
    Invoke-Stage 'scripts/test_hdr_encoding.py'
    Invoke-Stage 'scripts/composite.py' @('--final')
    Invoke-Stage 'scripts/hdr_mux.py'
    Invoke-Stage 'scripts/validate_final.py'
} finally {
    if ($null -ne $vpMcpProcess -and !$vpMcpProcess.HasExited) { Stop-Process -Id $vpMcpProcess.Id }
    Remove-Item Env:VP_MCP_PORT -ErrorAction SilentlyContinue
}
