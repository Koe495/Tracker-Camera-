param([string]$Sdk = ${env:KINECTSDK10_DIR})
$ErrorActionPreference = 'Stop'
if (!$Sdk) { $Sdk = Join-Path $env:ProgramFiles 'Microsoft SDKs/Kinect/v1.8' }
$dll = Join-Path $Sdk 'Assemblies/Microsoft.Kinect.dll'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
if (!(Test-Path -LiteralPath $dll)) { throw "Kinect SDK v1.8 assembly missing: $dll" }
if (!(Test-Path -LiteralPath $compiler)) { throw '.NET Framework x64 compiler missing.' }
$exe = Join-Path $PSScriptRoot 'KinectBridge.exe'
& $compiler /nologo /optimize+ /unsafe /platform:x64 "/reference:$dll" "/out:$exe" (Join-Path $PSScriptRoot 'KinectBridge.cs')
if ($LASTEXITCODE) { throw 'Kinect bridge compilation failed.' }
Copy-Item -LiteralPath $dll -Destination $PSScriptRoot -Force
