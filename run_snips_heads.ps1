# Hey Snips: reproduces Table 1 of the paper.
# Heads V1/V2/V2.5/V3 x {hinge, logistic}, seeds 1-4,
# 300 epochs, last checkpoint (no model selection).
# FRR is evaluated by exact event counting at FA = 1.0/h.
#
# Edit the paths below, then run:  powershell -File run_snips_heads.ps1

# --- paths: edit to your setup --------------------------------------------
$CODE  = $PSScriptRoot                     # this repository
$DATA  = "C:\path\to\snips_workdir"        # feats\{train,dev,test}, dev_data.list, test_data.list
$RUNS  = "$DATA\runs"
$SEEDS = @(1, 2, 3, 4)
# ---------------------------------------------------------------------------

cd $DATA
New-Item -ItemType Directory -Force -Path $RUNS | Out-Null
$TRAIN = "$CODE\train_margin_head.py"
$EXACT = "$CODE\frr_at_fa_exact.py"
$SUMMARY = "$RUNS\summary_snips_heads.txt"

$configs = @(
  @{ name = "v1";  variant = "v1"   },
  @{ name = "v2";  variant = "v2"   },
  @{ name = "v25"; variant = "v2.5" },
  @{ name = "v3";  variant = "v3"   }
)

foreach ($seed in $SEEDS) {
  foreach ($cfg in $configs) {
    foreach ($loss in @("hinge", "ce")) {
      $tag = "snips_$($cfg.name)_$($loss)_full_seed$seed"
      $done = (Test-Path "$RUNS\$tag.log") -and
              (Select-String -Path "$RUNS\$tag.log" -Pattern "epoch 299" -Quiet)
      if ($done) {
        Write-Host "== $tag training complete, skipping train =="
      } else {
        Write-Host "== TRAIN $tag =="
        python -u $TRAIN `
          --train_feats feats\train --dev_feats feats\dev `
          --dev_list dev_data.list `
          --variant $cfg.variant --context 15 --stored_k 15 --seed $seed `
          --keyword_txt '<HEY_SNIPS>' --target_fa 1.0 `
          --lr 0.001 --weight_decay 0.0001 `
          --loss $loss --select last --epochs 300 `
          --out_model $RUNS\$tag.pt 2>&1 | Tee-Object $RUNS\$tag.log
      }
      if (-not (Test-Path "$RUNS\$tag.score")) {
        Write-Host "== SCORE $tag =="
        python -u $CODE\score_margin.py `
          --feats feats\test --model $RUNS\$tag.pt `
          --stored_k 15 --keyword '<HEY_SNIPS>' `
          --score_file $RUNS\$tag.score
      }
      Write-Host "== EXACT FA=1.0 $tag =="
      $line = python $EXACT --keyword '<HEY_SNIPS>' `
        --test_data test_data.list --fa_target 1.0 `
        --score_file $RUNS\$tag.score | Select-Object -Last 1
      "$tag  $line" | Tee-Object -Append $SUMMARY
    }
  }
}
Write-Host "`n===== SUMMARY ====="
Get-Content $SUMMARY
