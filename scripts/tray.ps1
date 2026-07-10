param(
  [int]$Port = 3000,
  [switch]$NoHttps
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Protocol = if ($NoHttps) { "http" } else { "https" }
$DashboardUrl = "${Protocol}://localhost:${Port}/dashboard"
$ChatUrl = "${Protocol}://localhost:${Port}/"
$ServerProcess = $null

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function Start-LocalServer {
  if ($script:ServerProcess -and -not $script:ServerProcess.HasExited) {
    return
  }

  $node = (Get-Command node -ErrorAction Stop).Source
  $envVars = @{
    PORT = [string]$Port
    HTTPS = if ($NoHttps) { "0" } else { "1" }
  }

  foreach ($key in $envVars.Keys) {
    [Environment]::SetEnvironmentVariable($key, $envVars[$key], "Process")
  }

  $script:ServerProcess = Start-Process `
    -WindowStyle Hidden `
    -FilePath $node `
    -ArgumentList "server.js" `
    -WorkingDirectory $Root `
    -PassThru
}

function Stop-LocalServer {
  if ($script:ServerProcess -and -not $script:ServerProcess.HasExited) {
    Stop-Process -Id $script:ServerProcess.Id -ErrorAction SilentlyContinue
  }
}

function Open-Url([string]$Url) {
  Start-Process $Url
}

function Restart-LocalServer {
  Stop-LocalServer
  Start-Sleep -Milliseconds 500
  Start-LocalServer
}

Start-LocalServer
Start-Sleep -Milliseconds 900
Open-Url $DashboardUrl

$notifyIcon = New-Object System.Windows.Forms.NotifyIcon
$notifyIcon.Text = "Local LLM Serve"
$notifyIcon.Icon = [System.Drawing.SystemIcons]::Application
$notifyIcon.Visible = $true

$menu = New-Object System.Windows.Forms.ContextMenuStrip

$openDashboardItem = $menu.Items.Add("Open Dashboard")
$openDashboardItem.add_Click({ Open-Url $DashboardUrl })

$openChatItem = $menu.Items.Add("Open Chat")
$openChatItem.add_Click({ Open-Url $ChatUrl })

$restartItem = $menu.Items.Add("Restart Server")
$restartItem.add_Click({ Restart-LocalServer })

$stopItem = $menu.Items.Add("Stop Server")
$stopItem.add_Click({ Stop-LocalServer })

$menu.Items.Add("-") | Out-Null

$exitItem = $menu.Items.Add("Exit")
$exitItem.add_Click({
  Stop-LocalServer
  $notifyIcon.Visible = $false
  $notifyIcon.Dispose()
  [System.Windows.Forms.Application]::Exit()
})

$notifyIcon.ContextMenuStrip = $menu
$notifyIcon.add_DoubleClick({ Open-Url $DashboardUrl })
$notifyIcon.ShowBalloonTip(2500, "Local LLM Serve", "Server is running. Double-click for the dashboard.", [System.Windows.Forms.ToolTipIcon]::Info)

[System.Windows.Forms.Application]::Run()
