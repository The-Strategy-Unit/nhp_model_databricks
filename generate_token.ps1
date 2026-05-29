$ErrorActionPreference = "Stop"

$EnvFile = ".env"

if (-not (Test-Path $EnvFile)) {
    throw ".env file not found at path: $EnvFile"
}

Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*#' -or [string]::IsNullOrWhiteSpace($_)) {
        return
    }

    $name, $value = $_ -split '=', 2

    $name  = $name.Trim()
    $value = $value.Trim()

    Set-Item -Path "Env:$name" -Value $value
}

# --- Read required variables ---
$TableName          = $env:TABLE_NAME
$SecretScope        = $env:DATABRICKS_SECRET_SCOPE
$StorageAccountName = $env:AZURE_STORAGE_ACCOUNT_NAME

foreach ($var in @(
    "TABLE_NAME",
    "DATABRICKS_SECRET_SCOPE",
    "AZURE_STORAGE_ACCOUNT_NAME"
)) {
    if (-not $var) {
        throw "Missing required environment variable: $var"
    }
}

$ExpiryDate = (Get-Date).ToUniversalTime().AddDays(1).ToString("yyyy-MM-ddTHH:mmZ")

Write-Host "Generating Table SAS token (expires $ExpiryDate)..."

$TableToken = az storage table generate-sas `
    --account-name $StorageAccountName `
    --name $TableName `
    --permissions aur `
    --expiry $ExpiryDate `
    --output tsv

if (-not $TableToken) {
    throw "Failed to generate Table SAS token"
}

Write-Host "Table SAS token generated successfully."

Write-Host "Uploading Table SAS token to Databricks Secrets..."

$TableToken | databricks secrets put-secret `
    $SecretScope `
    table_sas

Write-Host "✅ Table SAS stored in Databricks secret scope '$SecretScope' as table_sas."

$StorageAccountName | databricks secrets put-secret `
    $SecretScope `
    account_name

$TableName | databricks secrets put-secret `
    $SecretScope `
    table_name


Write-Host "✅ account_name and table_name stored in Databricks secret scope '$SecretScope'"