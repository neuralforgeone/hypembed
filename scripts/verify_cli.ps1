param(
    [int]$RunId = 1
)

$cmd = Join-Path $PSScriptRoot "verify_cli.cmd"
& $cmd $RunId
exit $LASTEXITCODE