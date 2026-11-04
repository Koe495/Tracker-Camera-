$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$scratch = Join-Path $root '.tmp'
$models = Join-Path $root 'models'
New-Item -ItemType Directory -Force -Path $scratch, $models | Out-Null
$weights = Join-Path $models 'tennis-ball.onnx'
$sha256 = '0020f0356923712947d223fcbbe053748ee862f5642c963041a6dc67ca8f910d'
if (!(Test-Path -LiteralPath $weights)) {
    $partial = Join-Path $models 'tennis-ball.onnx.part'
    $url = 'https://huggingface.co/RJTPP/tennis-ball-detection/resolve/83ec86c/best.onnx'
    & curl.exe --fail --silent --show-error --location --retry 1 --max-time 120 $url --output $partial
    if ($LASTEXITCODE) { throw 'Model download failed.' }
    if ((Get-FileHash -LiteralPath $partial).Hash -ne $sha256) { throw 'Model SHA-256 mismatch.' }
    Move-Item -LiteralPath $partial -Destination $weights
} elseif ((Get-FileHash -LiteralPath $weights).Hash -ne $sha256) {
    throw 'Existing default model does not match the pinned SHA-256.'
}
$env:TEMP = $scratch
$env:TMP = $scratch
python -m venv (Join-Path $root '.venv')
if ($LASTEXITCODE) { throw 'Could not create the Python environment.' }
& (Join-Path $root '.venv/Scripts/python.exe') -m pip install --disable-pip-version-check --no-cache-dir -r (Join-Path $root 'requirements.txt')
if ($LASTEXITCODE) { throw 'Dependency installation failed.' }
& (Join-Path $root 'kinect/build.ps1')
