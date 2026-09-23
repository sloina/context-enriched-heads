# Hey Snips: reproduces Table 1 of the paper (and the anchor row).
#   stage 2  train heads anchor/V1/V2/V2.5/V3 x {hinge, logistic},
#            seeds 1-4, 300 epochs, last checkpoint (no model selection)
#   stage 3  ONE backbone pass over the test set: label-blind gated
#            candidates (local maxima of the baseline posterior >= 0.1,
#            boundary frames admitted one-sidedly, no cap, no fallback),
#            every head scored on the same candidates; the dense baseline
#            posterior is written from the same pass
#   eval     exact event-based FRR at FA = 1.0/h (frr_at_fa_exact.py)
# Completed steps are skipped, so an interrupted run can be re-run.
#
# Edit the path block, then run from the wekws checkout root (the Snips
# config resolves its global_cmvn relative to it):
#   powershell -ExecutionPolicy Bypass -File <repo>\run_snips_heads.ps1

# --- paths: edit to your setup --------------------------------------------
$CODE   = $PSScriptRoot                    # this repository
$WEKWS  = "C:\path\to\wekws"               # patched wekws checkout (PATCH_wekws_feature_extraction.md)
$DATA   = "C:\path\to\snips_workdir"       # feats\{train,dev}, dev_data.list, test_data.list
$CONFIG = "$WEKWS\examples\hey_snips\s0\exp\ds_tcn\config.yaml"
$DICT   = "$WEKWS\examples\hey_snips\s0\dict"
$CKPT   = "$CODE\checkpoints\snips_dstcn_avg30.pt"
$SEEDS  = @(1, 2, 3, 4)
$N_TEST = 23072                            # utterances in test_data.list (pass sanity check)
# ---------------------------------------------------------------------------

$RUNS    = "$DATA\runs"
$SCORES  = "$RUNS\scores"
$TRAIN   = "$CODE\train_margin_head.py"
$PROTO   = "$CODE\score_heads_protocols.py"
$EXACT   = "$CODE\frr_at_fa_exact.py"
$SUMMARY = "$RUNS\summary_snips_heads.txt"
$KEYWORD = "<HEY_SNIPS>"
$FA      = 1.0

cd $WEKWS
$env:PYTHONPATH = $WEKWS
New-Item -ItemType Directory -Force -Path $RUNS, $SCORES | Out-Null
foreach ($p in @($CONFIG, "$DICT\dict.txt", $CKPT, "$DATA\feats\train", "$DATA\feats\dev",
                 "$DATA\dev_data.list", "$DATA\test_data.list")) {
  if (-not (Test-Path -LiteralPath $p)) { throw "preflight: missing $p" }
}

$configs = @(
  @{ name = "anchor"; variant = "anchor" },
  @{ name = "v1";     variant = "v1"     },
  @{ name = "v2";     variant = "v2"     },
  @{ name = "v25";    variant = "v2.5"   },
  @{ name = "v3";     variant = "v3"     }
)

# --- stage 2: train ---------------------------------------------------------
foreach ($seed in $SEEDS) {
  foreach ($cfg in $configs) {
    foreach ($loss in @("hinge", "ce")) {
      $tag = "snips_$($cfg.name)_$($loss)_full_seed$seed"
      $done = (Test-Path "$RUNS\$tag.log") -and
              (Select-String -Path "$RUNS\$tag.log" -Pattern "epoch 299" -Quiet)
      if ($done) { Write-Host "== $tag trained, skip =="; continue }
      Write-Host "== TRAIN $tag =="
      python -u $TRAIN `
        --train_feats "$DATA\feats\train" --dev_feats "$DATA\feats\dev" `
        --dev_list "$DATA\dev_data.list" `
        --variant $cfg.variant --context 15 --stored_k 15 --seed $seed `
        --keyword_txt $KEYWORD --target_fa $FA `
        --lr 0.001 --weight_decay 0.0001 `
        --loss $loss --select last --epochs 300 `
        --out_model "$RUNS\$tag.pt" 2>&1 | Tee-Object "$RUNS\$tag.log"
      if ($LASTEXITCODE -ne 0) { throw "training failed: $tag" }
    }
  }
}

# --- stage 3: one backbone pass over the test set ----------------------------
$PASS_DONE = "$SCORES\pass.complete"
if (-not (Test-Path $PASS_DONE)) {
  Remove-Item "$SCORES\*.score" -ErrorAction SilentlyContinue
  $jobs = @(@{
    name = "snips"; keyword = $KEYWORD; keyword_index = 0
    heads = @("$RUNS\snips_*_full_seed*.pt")
    gated_dir = $SCORES
    baseline_out = "$SCORES\baseline.score"
  })
  ConvertTo-Json -Depth 5 @($jobs) | Set-Content -Encoding UTF8 "$SCORES\jobs.json"
  Write-Host "== BACKBONE PASS (gated, one-sided) =="
  python -u $PROTO `
    --config $CONFIG --checkpoint $CKPT --dict $DICT `
    --test_data "$DATA\test_data.list" `
    --pre_thresh 0.1 --num_workers 2 --gpu -1 `
    --expected_utts $N_TEST --jobs "$SCORES\jobs.json"
  if ($LASTEXITCODE -ne 0) { throw "backbone pass failed (score files incomplete)" }
  New-Item -ItemType File -Force -Path $PASS_DONE | Out-Null
} else {
  Write-Host "== backbone pass already complete, skip =="
}

# --- eval: exact FRR @ FA=1.0/h --------------------------------------------
foreach ($sf in @(Get-ChildItem "$SCORES\*.score" | Sort-Object Name)) {
  $tag = $sf.BaseName
  if ((Test-Path $SUMMARY) -and
      (Select-String -Path $SUMMARY -Pattern ("^" + [regex]::Escape($tag) + "  ") -Quiet)) {
    Write-Host "== $tag measured, skip =="; continue
  }
  $outp = @(python $EXACT --keyword $KEYWORD --test_data "$DATA\test_data.list" `
            --fa_target $FA --score_file $sf.FullName 2>&1)
  if ($LASTEXITCODE -ne 0 -or $outp[-1] -notmatch '^threshold ') {
    throw "evaluation failed for $tag`:`n$($outp -join "`n")"
  }
  "$tag  $($outp[-1])" | Tee-Object -Append $SUMMARY
}

Write-Host "`n===== SUMMARY (FRR @ $FA FA/h; baseline = dense posterior) ====="
Get-Content $SUMMARY
