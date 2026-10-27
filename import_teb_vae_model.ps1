<#
.SYNOPSIS
    Imports teb_vae_model, with its full git history, into a folder of another git repo.

.DESCRIPTION
    Merges the rewritten teb_vae_model history into the current branch of the
    repo that contains TargetFolder. Every imported file lands in TargetFolder.
    The script stops before it changes anything if a check fails, and it never
    pushes.

    The rewritten history already stores every file under one fixed path in the
    repo (Subfolder). TargetFolder must be that same folder.

    The source is a GitHub URL (default) or a local .bundle file.

    Steps:
      1. Check that git, the target folder and the source are valid.
      2. Check that the repo is clean and that the target folder has no tracked files.
      3. Fetch the source branch into a temporary local branch.
      4. Merge that branch with --allow-unrelated-histories.
      5. Check the result and delete the temporary branch.

    To run it with no command line arguments, edit the values in the
    "EDIT THESE VALUES" block below and run the file.

.PARAMETER TargetFolder
    Folder inside the target repo that receives the files. It must exist.

.PARAMETER Source
    GitHub URL of the repo that holds the rewritten history, or the path of a
    .bundle file.

.PARAMETER SourceBranch
    Branch in the source that holds the rewritten history. Use "main" for a bundle.

.PARAMETER Subfolder
    Path of TargetFolder relative to the repo root, with forward slashes. It must
    match the path that the rewritten history uses.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\import_teb_vae_model.ps1 -TargetFolder "Y:\em_analysis\python\TrainWithDataset\EarlyMaestra\early_maestra\vae\vae-teb-model"
#>
param(
    # ===================== EDIT THESE VALUES =====================
    [Alias("TargetRepo")]
    [string]$TargetFolder = "Y:\em_analysis\python\TrainWithDataset\EarlyMaestra\early_maestra\vae\vae-teb-model",
    [string]$Source       = "https://github.com/Mahdi-Si/VAE-TEB.git",
    [string]$SourceBranch = "teb-vae-model-import",
    [string]$Subfolder    = "python/TrainWithDataset/EarlyMaestra/early_maestra/vae/vae-teb-model"
    # =============================================================
)

$ErrorActionPreference = "Stop"

$TempBranch   = "teb-import"
$MergeMessage = "Import teb_vae_model with full history"
$Subfolder    = $Subfolder.Trim("/")

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

# Runs git inside $Repo and returns its stdout. Throws on a non-zero exit code.
function Invoke-Git {
    $out = & git.exe -C $Repo @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args -join ' ') failed with exit code $LASTEXITCODE" }
    return $out
}

# Runs git inside $Repo and returns only $true or $false. Never throws.
function Test-Git {
    $ErrorActionPreference = "Continue"
    & git.exe -C $Repo @args *> $null
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------- 1. Basic checks
Step "Checking git, the target folder and the source"

if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
    Fail "git is not on PATH. Install Git for Windows and open a new PowerShell window."
}
if (-not (Test-Path -LiteralPath $TargetFolder -PathType Container)) {
    Fail "The target folder does not exist. Create it first: $TargetFolder"
}
$TargetFolder = (Resolve-Path -LiteralPath $TargetFolder).Path

# Every git command below runs from the repo root, which git reports itself.
$Repo = $TargetFolder
if (-not (Test-Git rev-parse --is-inside-work-tree)) {
    Fail "This folder is not inside a git repo: $TargetFolder"
}
$prefix = ((Invoke-Git rev-parse --show-prefix) | Out-String).Trim().TrimEnd("/")
$Repo   = ((Invoke-Git rev-parse --show-toplevel) | Out-String).Trim() -replace "/", "\"
if ($prefix -ine $Subfolder) {
    Fail ("The target folder is '$prefix' inside the repo, but the imported files use '$Subfolder'. " +
          "Choose the folder that matches '$Subfolder', or ask for the history to be rewritten.")
}
# Use the path that git reports, not a mapped drive letter, so that all paths below compare equal.
$TargetFolder = Join-Path $Repo ($Subfolder -replace "/", "\")

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
Write-Host "Repo root:     $Repo"
Write-Host "Target folder: $Subfolder"
Write-Host "Source:        $Source ($SourceBranch)"

# ---------------------------------------------------------------- 2. Safety checks
Step "Checking that the repo is ready"

$branch = (& git.exe -C $Repo symbolic-ref --quiet --short HEAD)
if ($LASTEXITCODE -ne 0) {
    Fail "HEAD is detached. Check out the branch that must receive the import, for example: git checkout main"
}
$gitDir = ((Invoke-Git rev-parse --absolute-git-dir) | Out-String).Trim()
foreach ($marker in "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply") {
    if (Test-Path -LiteralPath (Join-Path $gitDir $marker)) {
        Fail "A merge, rebase, cherry-pick or revert is in progress. Finish or abort it first."
    }
}
$dirty = Invoke-Git status --porcelain --untracked-files=no
if ($dirty) {
    Write-Host ($dirty -join "`n")
    Fail "The repo has uncommitted changes (listed above). Commit or stash them first."
}
$tracked = Invoke-Git ls-files -- $Subfolder
if ($tracked) {
    Fail "Git already tracks files in '$Subfolder'. Move or remove them first."
}
if (Test-Git rev-parse --verify --quiet "refs/heads/$TempBranch") {
    Fail "A branch named '$TempBranch' already exists. Delete it with: git branch -D $TempBranch"
}

$hasCommits = Test-Git rev-parse --verify --quiet HEAD
$headBefore = $null
if ($hasCommits) {
    $headBefore = ((Invoke-Git rev-parse HEAD) | Out-String).Trim()

    # Windows ignores letter case in paths, but git does not. If git already tracks a
    # parent folder with other letter case, the import would create a second folder.
    $trackedDirs = Invoke-Git ls-tree -r -d --name-only HEAD
    $parts = $Subfolder.Split("/")
    for ($i = 1; $i -le $parts.Length; $i++) {
        $ancestor = ($parts[0..($i - 1)] -join "/")
        $sameName = @($trackedDirs | Where-Object { $_ -ieq $ancestor })
        if ($sameName.Count -gt 0 -and -not ($sameName -ccontains $ancestor)) {
            Fail ("Git tracks the folder '$($sameName[0])', but the import uses '$ancestor'. " +
                  "The letter case differs, so the history must be rewritten with the correct case.")
        }
    }
}

$untracked = @(Get-ChildItem -LiteralPath $TargetFolder -Force -Recurse -File -ErrorAction SilentlyContinue)
if ($untracked.Count -gt 0) {
    Write-Host "Note: the target folder has $($untracked.Count) untracked files. The merge stops if one has the same name as an imported file." -ForegroundColor Yellow
}
Write-Host "Current branch: $branch"

# ---------------------------------------------------------------- 3. Fetch
Step "Fetching '$SourceBranch' into the temporary branch '$TempBranch'"
Invoke-Git fetch --no-tags $Source "$($SourceBranch):$TempBranch" | Out-Null
$importTip   = ((Invoke-Git rev-parse $TempBranch) | Out-String).Trim()
$importCount = [int]((Invoke-Git rev-list --count $TempBranch) | Out-String).Trim()
Write-Host "Fetched $importCount commits."

$importFiles = @(Invoke-Git ls-tree -r --name-only $TempBranch)
$outside = $importFiles | Where-Object { -not $_.StartsWith("$Subfolder/") }
if ($outside) {
    Invoke-Git branch -D $TempBranch | Out-Null
    Fail "The source has files outside '$Subfolder/'. It was rewritten for a different folder."
}

# Windows limits a full path to 259 characters unless git uses core.longpaths.
$longPaths = ((& git.exe -C $Repo config --type=bool core.longpaths) | Out-String).Trim()
if ($longPaths -ne "true") {
    $tooLong = @($importFiles | Where-Object { ($Repo.Length + 1 + $_.Length) -gt 259 })
    if ($tooLong.Count -gt 0) {
        Invoke-Git branch -D $TempBranch | Out-Null
        Write-Host "Longest path: $Repo\$($tooLong | Sort-Object Length | Select-Object -Last 1)"
        Fail ("$($tooLong.Count) imported files would have a path longer than 259 characters. " +
              "Move the repo to a shorter folder, or run: git -C `"$Repo`" config core.longpaths true")
    }
}

# ---------------------------------------------------------------- 4. Merge
Step "Merging into '$branch'"
& git.exe -C $Repo merge --allow-unrelated-histories --no-edit --no-stat -m $MergeMessage $TempBranch
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "The merge failed. Restoring the repo..." -ForegroundColor Yellow
    if (Test-Path -LiteralPath (Join-Path $gitDir "MERGE_HEAD")) {
        & git.exe -C $Repo merge --abort
    } elseif ($hasCommits) {
        # The repo was clean before the merge, so this only undoes what the merge did.
        & git.exe -C $Repo reset --hard --quiet $headBefore
    }
    # A merge that stops during checkout can leave imported files on disk. Remove only
    # those, and keep every file that was in the target folder before the merge.
    $before = @{}
    foreach ($f in $untracked) { $before[$f.FullName.ToLowerInvariant()] = $true }
    $dirs = @{}
    foreach ($rel in $importFiles) {
        $full = Join-Path $Repo ($rel -replace "/", "\")
        if ((Test-Path -LiteralPath $full -PathType Leaf) -and -not $before.ContainsKey($full.ToLowerInvariant())) {
            Remove-Item -LiteralPath $full -Force
        }
        $d = Split-Path $full -Parent
        while ($d.Length -gt $TargetFolder.Length) { $dirs[$d] = $true; $d = Split-Path $d -Parent }
    }
    $dirs.Keys | Sort-Object Length -Descending | ForEach-Object {
        if ((Test-Path -LiteralPath $_) -and -not (Get-ChildItem -LiteralPath $_ -Force)) {
            Remove-Item -LiteralPath $_ -Force
        }
    }
    [void](Test-Git branch -D $TempBranch)
    $left = Invoke-Git status --porcelain --untracked-files=no
    if ($left) {
        Fail "The merge failed (see the git message above), and the repo still has changes. Check it by hand."
    }
    Fail "The merge failed (see the git message above). The script restored the repo to its earlier state."
}

# ---------------------------------------------------------------- 5. Verify and clean up
Step "Checking the result"

if (-not (Test-Git merge-base --is-ancestor $importTip HEAD)) {
    Fail "The imported history is not part of '$branch'. Check the repo by hand."
}
# Equal tree hashes mean that every file under the subfolder is byte-identical.
$treeImported = ((Invoke-Git rev-parse "$($importTip):$Subfolder") | Out-String).Trim()
$treeHead     = ((Invoke-Git rev-parse "HEAD:$Subfolder") | Out-String).Trim()
if ($treeImported -ne $treeHead) {
    Fail "The folder '$Subfolder' in HEAD differs from the source. Check the repo by hand."
}
Write-Host "All $importCount imported commits are in '$branch'."
Write-Host "All imported files are present and identical in '$Subfolder'."

Invoke-Git branch -d $TempBranch | Out-Null
Write-Host "Deleted the temporary branch '$TempBranch'."

Write-Host ""
Write-Host "DONE. Latest commits:" -ForegroundColor Green
Invoke-Git log --oneline -5 | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Nothing was pushed."
if ($headBefore) {
    Write-Host "To undo before you push, run: git -C `"$Repo`" reset --hard $headBefore"
}
