# Aggregate per-sequence mot_eval CSV (+ optional timing CSV) of one result
# directory into a single summary row, and print markdown rows for the docs.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools\mot_eval\aggregate.ps1 `
#     -Dir C:\work\home\mtracker\results\bench2_visdrone_yolov8l `
#     -Label "YOLOv8l-visdrone, TensorRT FP16" `
#     -SummaryCsv C:\work\home\mtracker\results\bench2_summary.csv
#
# The metrics CSV is the per-sequence CSV written by mot_eval (e.g. visdrone.csv,
# mot20.csv). The timing CSV (<dataset>_timing.csv) is optional and written only
# by VisDrone runs.
#
# Aggregation follows the existing summary convention (results/trt1_visdrone_summary.csv):
#   MOTA = 1 - (FN + FP + IDSW) / num_objects   (summed)
#   IDF1/IDP/IDR from summed IDTP/IDFP/IDFN
#   Recall = matches / GT, Precision = matches / predictions
#   MOTP (IoU) weighted by matches
#
# Numbers are written in the invariant culture (dot decimal) so the summary CSV
# stays machine-readable regardless of the host locale.

param(
    [Parameter(Mandatory = $true)][string]$Dir,
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$SummaryCsv = ""
)

$ErrorActionPreference = "Stop"
$Inv = [Globalization.CultureInfo]::InvariantCulture

function ToDouble {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return 0.0 }
    $s = $Value.Trim().Replace(',', '.')
    return [double]::Parse($s, $Inv)
}

function ToLong {
    param([string]$Value)
    return [long][math]::Round((ToDouble $Value))
}

function Get-MetricsCsv {
    param([string]$Directory)
    Get-ChildItem $Directory -Filter *.csv |
        Where-Object { $_.Name -notlike "*_timing.csv" } |
        Select-Object -First 1
}

# Fixed-decimal, invariant-culture string for a float; "" for $null.
function FNum {
    param($Value, [int]$Decimals)
    if ($null -eq $Value) { return "" }
    return ([double]$Value).ToString("F" + $Decimals, $Inv)
}

$metricsCsv = Get-MetricsCsv $Dir
if (-not $metricsCsv) { throw "no metrics csv in $Dir" }

$rows = @(Import-Csv $metricsCsv.FullName)
if ($rows.Count -eq 0) { throw "empty metrics csv: $($metricsCsv.FullName)" }

$fp = 0L; $fn = 0L; $idsw = 0L; $idtp = 0L; $idfp = 0L; $idfn = 0L
$obj = 0L; $pred = 0L; $matches = 0L; $frames = 0L
$motpWeightedNum = 0.0

foreach ($r in $rows) {
    $m = ToLong $r.matches
    $fp      += (ToLong $r.fp)
    $fn      += (ToLong $r.fn)
    $idsw    += (ToLong $r.idsw)
    $idtp    += (ToLong $r.idtp)
    $idfp    += (ToLong $r.idfp)
    $idfn    += (ToLong $r.idfn)
    $obj     += (ToLong $r.num_objects)
    $pred    += (ToLong $r.num_predictions)
    $matches += $m
    $frames  += (ToLong $r.frames)
    $motpWeightedNum += (ToDouble $r.motp_overlap) * $m
}

$idp = if (($idtp + $idfp) -gt 0) { $idtp / ($idtp + $idfp) } else { 0.0 }
$idr = if (($idtp + $idfn) -gt 0) { $idtp / ($idtp + $idfn) } else { 0.0 }
$idf1 = if ((2 * $idtp + $idfp + $idfn) -gt 0) { (2 * $idtp) / (2 * $idtp + $idfp + $idfn) } else { 0.0 }
$mota = if ($obj -gt 0) { 1 - ($fn + $fp + $idsw) / $obj } else { 0.0 }
$recall = if ($obj -gt 0) { $matches / $obj } else { 0.0 }
$precision = if ($pred -gt 0) { $matches / $pred } else { 0.0 }
$motpIou = if ($matches -gt 0) { $motpWeightedNum / $matches } else { 0.0 }

# Optional timing (VisDrone runs)
$seconds = $null; $msPerFrame = $null; $fps = $null; $rawDetections = $null
$timingCsv = Get-ChildItem $Dir -Filter "*_timing.csv" | Select-Object -First 1
if ($timingCsv) {
    $trows = @(Import-Csv $timingCsv.FullName)
    $sec = 0.0; $dets = 0L; $tframes = 0L
    foreach ($t in $trows) {
        $sec += (ToDouble $t.seconds)
        $dets += (ToLong $t.detections)
        $tframes += (ToLong $t.frames)
    }
    $seconds = $sec
    $fps = if ($sec -gt 0) { $tframes / $sec } else { 0.0 }
    $msPerFrame = if ($tframes -gt 0) { 1000.0 * $sec / $tframes } else { 0.0 }
    $rawDetections = $dets
}

# Markdown quality row (match the docs/mot_bench.md column order)
$qual = "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} |" -f `
    $Label, (FNum $mota 4), (FNum $idf1 4), (FNum $idp 4), (FNum $idr 4), (FNum $recall 4), (FNum $precision 4), (FNum $motpIou 4), $idsw, $fp, $fn, $pred
Write-Output "QUALITY: $qual"

if ($seconds -ne $null) {
    $speed = "| {0} | {1} | {2} | {3} |" -f $Label, (FNum $seconds 2), (FNum $msPerFrame 2), (FNum $fps 2)
    Write-Output "SPEED: $speed"
}

if ($SummaryCsv) {
    $header = "label,mota,idf1,idp,idr,recall,precision,motp_iou,idsw,fp,fn,num_objects,num_predictions,matches,frames,seconds,ms_per_frame,fps,raw_detections"
    $writeHeader = (-not (Test-Path $SummaryCsv)) -or ((Get-Item $SummaryCsv).Length -eq 0)
    $fields = @(
        ('"' + $Label + '"'),
        (FNum $mota 4), (FNum $idf1 4), (FNum $idp 4), (FNum $idr 4),
        (FNum $recall 4), (FNum $precision 4), (FNum $motpIou 4),
        $idsw, $fp, $fn, $obj, $pred, $matches, $frames,
        (FNum $seconds 2), (FNum $msPerFrame 2), (FNum $fps 2), $rawDetections
    )
    $line = $fields -join ","
    if ($writeHeader) { Add-Content -Path $SummaryCsv -Value $header -Encoding UTF8 }
    Add-Content -Path $SummaryCsv -Value $line -Encoding UTF8
    Write-Output "appended to $SummaryCsv"
}
