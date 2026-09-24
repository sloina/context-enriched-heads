# Mobvoi (Hi Xiaowen / Nihao Wenwen): reproduces Table 2 of the paper.
#   stage 2  train heads V1/V2/V2.5/V3 x {hinge, logistic} per keyword,
#            seeds 1-3, 300 epochs, last checkpoint, l2 = 1e-3
#   stage 3  ONE backbone pass over the test set serving BOTH keywords
#            (one MDTC model, two classifier rows): label-blind gated
#            candidates (local maxima of the baseline posterior >= 0.1,
#            boundary frames admitted one-sidedly, no cap, no fallback);
#            the dense baseline posterior of each keyword is written from
#            the same pass
#   eval     exact event-based FRR at FA = 0.5/h on the full test set and
#            on the three cumulative cleaning levels of the label-noise
#            audit (-bad, -ultra, -vh), built by make_clean_lists.py from
#            the removed-key snapshots in audit_lists/
# Completed steps are skipped, so an interrupted run can be re-run.
#
# Edit the path block, then run:
#   powershell -ExecutionPolicy Bypass -File <repo>\run_mobvoi_heads.ps1

# --- paths: edit to your setup --------------------------------------------
$CODE   = $PSScriptRoot                      # this repository
$WEKWS  = "C:\path\to\wekws"                 # patched wekws checkout (PATCH_wekws_feature_extraction.md)
$DATA   = "C:\path\to\mobvoi_workdir"        # margin_features_K31_kw{0,1}\{train,dev}, mobvoi_{dev,test}_data.list
$CONFIG = "$WEKWS\examples\hi_xiaowen\s0\exp\mdtc\config.yaml"
$DICT   = "$WEKWS\examples\hi_xiaowen\s0\dict"
$CKPT   = "$CODE\checkpoints\mobvoi_mdtc_avg30.pt"
$SEEDS  = @(1, 2, 3)
$N_TEST = 73459                              # utterances in mobvoi_test_data.list (pass sanity check)
$KEYS   = "$CODE\audit_lists"   # removed-key snapshots (see README there)
# ---------------------------------------------------------------------------

$RUNS    = "$DATA\runs"
$SCORES  = "$RUNS\scores"
$TRAIN   = "$CODE\train_margin_head.py"
$PROTO   = "$CODE\score_heads_protocols.py"
$EXACT   = "$CODE\frr_at_fa_exact.py"
$SUMMARY = "$RUNS\summary_mobvoi_heads.txt"
$FA      = 0.5
$LISTS   = "$RUNS\lists"

# --- cleaning levels: built from the removed-key snapshots (cumulative) -------
if (-not (Test-Path "$LISTS\mobvoi_test_nobad_noultra_novh.list")) {
  python "$CODE\make_clean_lists.py" --test_list "$DATA\mobvoi_test_data.list" `
    --keys_dir $KEYS --out_dir $LISTS
  if ($LASTEXITCODE -ne 0) { throw "building the cleaned test lists failed" }
}
$LEVELS = @(
  @{ name = "full";   file = "$DATA\mobvoi_test_data.list" },
  @{ name = "-bad";   file = "$LISTS\mobvoi_test_nobad.list" },
  @{ name = "-ultra"; file = "$LISTS\mobvoi_test_nobad_noultra.list" },
  @{ name = "-vh";    file = "$LISTS\mobvoi_test_nobad_noultra_novh.list" }
)

$kws = @(
  @{ kw = "kw0"; keyword = "<HI_XIAOWEN>";   idx = 0 },
  @{ kw = "kw1"; keyword = "<NIHAO_WENWEN>"; idx = 1 }
)
$configs = @(
  @{ name = "v1";  variant = "v1"   },
  @{ name = "v2";  variant = "v2"   },
  @{ name = "v25"; variant = "v2.5" },
  @{ name = "v3";  variant = "v3"   }
)

cd $WEKWS
$env:PYTHONPATH = $WEKWS
New-Item -ItemType Directory -Force -Path $RUNS, "$SCORES\kw0", "$SCORES\kw1" | Out-Null
$need = @($CONFIG, "$DICT\dict.txt", $CKPT, "$DATA\mobvoi_dev_data.list") + ($LEVELS | ForEach-Object { $_.file })
foreach ($k in $kws) { $need += "$DATA\margin_features_K31_$($k.kw)\train", "$DATA\margin_features_K31_$($k.kw)\dev" }
foreach ($p in $need) { if (-not (Test-Path -LiteralPath $p)) { throw "preflight: missing $p" } }

# --- stage 2: train ---------------------------------------------------------
foreach ($seed in $SEEDS) {
  foreach ($k in $kws) {
    foreach ($cfg in $configs) {
      foreach ($loss in @("hinge", "ce")) {
        $tag = "mobvoi_$($k.kw)_$($cfg.name)_$($loss)_full_seed$seed"
        $done = (Test-Path "$RUNS\$tag.log") -and
                (Select-String -Path "$RUNS\$tag.log" -Pattern "epoch 299" -Quiet)
        if ($done) { Write-Host "== $tag trained, skip =="; continue }
        Write-Host "== TRAIN $tag =="
        python -u $TRAIN `
          --train_feats "$DATA\margin_features_K31_$($k.kw)\train" `
          --dev_feats "$DATA\margin_features_K31_$($k.kw)\dev" `
          --dev_list "$DATA\mobvoi_dev_data.list" `
          --variant $cfg.variant --context 15 --stored_k 31 --seed $seed `
          --pos_label $($k.idx) --keyword_txt "$($k.keyword)" `
          --lr 0.001 --weight_decay 0.001 --target_fa $FA `
          --loss $loss --select last --epochs 300 `
          --out_model "$RUNS\$tag.pt" 2>&1 | Tee-Object "$RUNS\$tag.log"
        if ($LASTEXITCODE -ne 0) { throw "training failed: $tag" }
      }
    }
  }
}

# --- stage 3: one backbone pass, both keywords -------------------------------
$PASS_DONE = "$SCORES\pass.complete"
if (-not (Test-Path $PASS_DONE)) {
  Remove-Item "$SCORES\kw*\*.score" -ErrorAction SilentlyContinue
  $jobs = @()
  foreach ($k in $kws) {
    $jobs += @{
      name = $k.kw; keyword = $k.keyword; keyword_index = $k.idx
      heads = @("$RUNS\mobvoi_$($k.kw)_*_full_seed*.pt")
      gated_dir = "$SCORES\$($k.kw)"
      baseline_out = "$SCORES\$($k.kw)\baseline.score"
    }
  }
  ConvertTo-Json -Depth 5 @($jobs) | Set-Content -Encoding UTF8 "$SCORES\jobs.json"
  Write-Host "== BACKBONE PASS (gated, one-sided; kw0 + kw1) =="
  python -u $PROTO `
    --config $CONFIG --checkpoint $CKPT --dict $DICT `
    --test_data "$DATA\mobvoi_test_data.list" `
    --pre_thresh 0.1 --num_workers 2 --gpu -1 `
    --expected_utts $N_TEST --jobs "$SCORES\jobs.json"
  if ($LASTEXITCODE -ne 0) { throw "backbone pass failed (score files incomplete)" }
  New-Item -ItemType File -Force -Path $PASS_DONE | Out-Null
} else {
  Write-Host "== backbone pass already complete, skip =="
}

# --- eval: exact FRR @ FA=0.5/h, per cleaning level --------------------------
foreach ($k in $kws) {
  foreach ($sf in @(Get-ChildItem "$SCORES\$($k.kw)\*.score" | Sort-Object Name)) {
    $tag = $sf.BaseName
    foreach ($l in $LEVELS) {
      $key = "$tag  [$($k.kw):$($l.name)]"
      if ((Test-Path $SUMMARY) -and
          (Select-String -Path $SUMMARY -Pattern ("^" + [regex]::Escape($key) + "  ") -Quiet)) {
        Write-Host "== $key measured, skip =="; continue
      }
      $outp = @(python $EXACT --keyword "$($k.keyword)" --test_data $l.file `
                --fa_target $FA --score_file $sf.FullName 2>&1)
      if ($LASTEXITCODE -ne 0 -or $outp[-1] -notmatch '^threshold ') {
        throw "evaluation failed for $key`:`n$($outp -join "`n")"
      }
      "$key  $($outp[-1])" | Tee-Object -Append $SUMMARY
    }
  }
}

Write-Host "`n===== SUMMARY (FRR @ $FA FA/h; baseline = dense posterior) ====="
Get-Content $SUMMARY
