<#
.SYNOPSIS
    Imports teb_vae_model, with its full git history, into another git repo.

.DESCRIPTION
    Merges the rewritten teb_vae_model history into the current branch of the
    target repo. Every imported file lands under the subfolder "teb_vae_model".
    The script stops before it changes anything if a check fails, and it never
    pushes.

    The source is a GitHub URL (default) or a local .bundle file.

    Steps:
      1. Check that git, the target repo and the source are valid.
      2. Check that the target repo is clean and has no "teb_vae_model" folder.
      3. Fetch the source branch into a temporary local branch.
      4. Merge that branch with --allow-unrelated-histories.
      5. Check the result and delete the temporary branch.

    To run it with no command line arguments, edit the values in the
    "EDIT THESE VALUES" block below and run the file.

.PARAMETER TargetRepo
    Root folder of the repo that receives the import.

.PARAMETER Source
    GitHub URL of the repo that holds the rewritten history, or the path of a
    .bundle file.

.PARAMETER SourceBranch
    Branch in the source that holds the rewritten history. Use "main" for a bundle.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\import_teb_vae_model.ps1 -TargetRepo "D:\work\my_repo"
#>
param(
    # ===================== EDIT THESE VALUES =====================
    [string]$TargetRepo   = "C:\path\to\target_repo",
    [string]$Source       = "https://github.com/Mahdi-Si/VAE-TEB.git",
    [string]$SourceBranch = "teb-vae-model-import"
    # =============================================================
)

$ErrorActionPreference = "Stop"

$Subfolder    = "teb_vae_model"
$TempBranch   = "teb-import"
$MergeMessage = "Import teb_vae_model with full history"

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "STOPPED: $Message" -ForegroundColor Red
    Write-Host "Nothing was merged." -ForegroundColor Red
    exit 1
}

function Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# Runs git inside the target repo and returns its stdout. Throws on a non-zero exit code.
function Invoke-Git {
    $out = & git.exe -C $TargetRepo @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') failed with exit code $LASTEXITCODE" }
    return $out
}

# Runs git inside the target repo and returns only $true or $false. Never throws.
function Test-Git {
    $ErrorActionPreference = "Continue"
    & git.exe -C $TargetRepo @args *> $null
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------- 1. Basic checks
Step "Checking git, the target repo and the source"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is not on PATH. Install Git for Windows and open a new PowerShell window."
}
if (-not (Test-Path -LiteralPath $TargetRepo -PathType Container)) {
    Fail "The target repo folder does not exist: $TargetRepo"
}
$TargetRepo = (Resolve-Path -LiteralPath $TargetRepo).Path
if (-not (Test-Git rev-parse --is-inside-work-tree)) {
    Fail "This folder is not a git repo: $TargetRepo"
}
$topLevel = (Invoke-Git rev-parse --show-toplevel) -replace "/", "\"
if ($topLevel.TrimEnd("\") -ne $TargetRepo.TrimEnd("\")) {
    Fail "TargetRepo must be the repo root. Use this folder instead: $topLevel"
}
if ($Source -match '^(https?|ssh|git)://|^git@') {
    if (-not (Test-Git ls-remote --exit-code --heads $Source $SourceBranch)) {
        Fail "Cannot read branch '$SourceBranch' from $Source. Check the internet connection and the branch name."
    }
} else {
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        Fail "The bundle file does not exist: $Source"
    }
    $Source = (Resolve-Path -LiteralPath $Source).Path
    if (-not (Test-Git bundle verify $Source)) {
        Fail "The bundle is damaged or incomplete. Copy it again from the source machine."
    }
}
Write-Host "Target repo: $TargetRepo"
Write-Host "Source:      $Source ($SourceBranch)"

# ---------------------------------------------------------------- 2. Safety checks
Step "Checking that the target repo is ready"

$branch = (& git.exe -C $TargetRepo symbolic-ref --quiet --short HEAD)
if ($LASTEXITCODE -ne 0) {
    Fail "HEAD is detached. Check out the branch that must receive the import, for example: git checkout main"
}
$gitDir = (Invoke-Git rev-parse --absolute-git-dir)
foreach ($marker in "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply") {
    if (Test-Path -LiteralPath (Join-Path $gitDir $marker)) {
        Fail "A merge, rebase, cherry-pick or revert is in progress. Finish or abort it first."
    }
}
$dirty = Invoke-Git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Host ($dirty -join "`n")
    Fail "The target repo has uncommitted changes (listed above). Commit or stash them first."
}
if (Test-Path -LiteralPath (Join-Path $TargetRepo $Subfolder)) {
    Fail "A '$Subfolder' folder already exists in the target repo. Rename or remove it first."
}
if (Test-Git rev-parse --verify --quiet "refs/heads/$TempBranch") {
    Fail "A branch named '$TempBranch' already exists. Delete it with: git branch -D $TempBranch"
}
$hasCommits = Test-Git rev-parse --verify --quiet HEAD
$headBefore = $null
if ($hasCommits) { $headBefore = (Invoke-Git rev-parse HEAD) }
Write-Host "Current branch: $branch"

# ---------------------------------------------------------------- 3. Fetch
Step "Fetching '$SourceBranch' into the temporary branch '$TempBranch'"
Invoke-Git fetch --no-tags $Source "$($SourceBranch):$TempBranch" | Out-Null
$importTip   = (Invoke-Git rev-parse $TempBranch)
$importCount = [int](Invoke-Git rev-list --count $TempBranch)
Write-Host "Fetched $importCount commits."

$outside = Invoke-Git ls-tree -r --name-only $TempBranch | Where-Object { -not $_.StartsWith("$Subfolder/") }
if ($outside) {
    Invoke-Git branch -D $TempBranch | Out-Null
    Fail "The source has files outside '$Subfolder/'. It is not the rewritten history."
}

# ---------------------------------------------------------------- 4. Merge
Step "Merging into '$branch'"
& git.exe -C $TargetRepo merge --allow-unrelated-histories --no-edit --no-stat -m $MergeMessage $TempBranch
if ($LASTEXITCODE -ne 0) {
    if (Test-Path -LiteralPath (Join-Path $gitDir "MERGE_HEAD")) {
        & git.exe -C $TargetRepo merge --abort
    }
    [void](Test-Git branch -D $TempBranch)
    Fail "The merge failed. The script aborted it, so the repo is as it was before."
}

# ---------------------------------------------------------------- 5. Verify and clean up
Step "Checking the result"

if (-not (Test-Git merge-base --is-ancestor $importTip HEAD)) {
    Fail "The imported history is not part of '$branch'. Check the repo by hand."
}
# Equal tree hashes mean that every file under the subfolder is byte-identical.
$treeImported = (Invoke-Git rev-parse "$($importTip):$Subfolder")
$treeHead     = (Invoke-Git rev-parse "HEAD:$Subfolder")
if ($treeImported -ne $treeHead) {
    Fail "The '$Subfolder' folder in HEAD differs from the source. Check the repo by hand."
}
Write-Host "All $importCount imported commits are in '$branch'."
Write-Host "All imported files are present and identical under '$Subfolder\'."

Invoke-Git branch -d $TempBranch | Out-Null
Write-Host "Deleted the temporary branch '$TempBranch'."

Write-Host ""
Write-Host "DONE. Latest commits:" -ForegroundColor Green
Invoke-Git log --oneline -5 | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Nothing was pushed."
if ($headBefore) {
    Write-Host "To undo before you push, run: git -C `"$TargetRepo`" reset --hard $headBefore"
}
