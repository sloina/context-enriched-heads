# Mobvoi (Hi Xiaowen / Nihao Wenwen): reproduces the head rows of Table 2.
# Heads V1/V2/V2.5/V3 x {hinge, logistic} per keyword, seeds 1-4,
# 300 epochs, last checkpoint (no model selection), l2 = 1e-3.
# FRR is evaluated by exact event counting at FA = 0.5/h.
#
# Edit the paths below, then run:  powershell -File run_mobvoi_heads.ps1

# --- paths: edit to your setup --------------------------------------------
$CODE  = $PSScriptRoot                       # this repository
$DATA  = "C:\path\to\mobvoi_workdir"         # margin_features_K31_kw{0,1}\, mobvoi_{dev,test}_data.list
$RUNS  = "$DATA\runs"
$SEEDS = @(1, 2, 3, 4)
# ---------------------------------------------------------------------------

cd $DATA
New-Item -ItemType Directory -Force -Path $RUNS | Out-Null
$TRAIN = "$CODE\train_margin_head.py"
$EXACT = "$CODE\frr_at_fa_exact.py"
$SUMMARY = "$RUNS\summary_mobvoi_heads.txt"

$kws = @(
  @{ kw = "kw0"; keyword = "<HI_XIAOWEN>";   pos = 0 },
  @{ kw = "kw1"; keyword = "<NIHAO_WENWEN>"; pos = 1 }
)
$configs = @(
  @{ name = "v1";  variant = "v1"   },
  @{ name = "v2";  variant = "v2"   },
  @{ name = "v25"; variant = "v2.5" },
  @{ name = "v3";  variant = "v3"   }
)

foreach ($seed in $SEEDS) {
  foreach ($k in $kws) {
    foreach ($cfg in $configs) {
      foreach ($loss in @("hinge", "ce")) {
        $tag = "mobvoi_$($k.kw)_$($cfg.name)_$($loss)_full_seed$seed"
        $done = (Test-Path "$RUNS\$tag.log") -and
                (Select-String -Path "$RUNS\$tag.log" -Pattern "epoch 299" -Quiet)
        if ($done) {
          Write-Host "== $tag training complete, skipping train =="
        } else {
          Write-Host "== TRAIN $tag =="
          python -u $TRAIN `
            --train_feats "margin_features_K31_$($k.kw)\train" `
            --dev_feats "margin_features_K31_$($k.kw)\dev" `
            --dev_list mobvoi_dev_data.list `
            --variant $cfg.variant --context 15 --stored_k 31 --seed $seed `
            --pos_label $($k.pos) --keyword_txt "$($k.keyword)" `
            --lr 0.001 --weight_decay 0.001 --target_fa 0.5 `
            --loss $loss --select last --epochs 300 `
            --out_model $RUNS\$tag.pt 2>&1 | Tee-Object $RUNS\$tag.log
        }
        if (-not (Test-Path "$RUNS\$tag.score")) {
          Write-Host "== SCORE $tag =="
          python -u $CODE\score_margin.py `
            --feats "margin_features_K31_$($k.kw)\test" --model $RUNS\$tag.pt `
            --stored_k 31 --keyword "$($k.keyword)" `
            --score_file $RUNS\$tag.score
        }
        Write-Host "== EXACT FA=0.5 $tag =="
        $line = python $EXACT --keyword "$($k.keyword)" `
          --test_data mobvoi_test_data.list --fa_target 0.5 `
          --score_file $RUNS\$tag.score | Select-Object -Last 1
        "$tag  $line" | Tee-Object -Append $SUMMARY
      }
    }
  }
}
Write-Host "`n===== SUMMARY ====="
Get-Content $SUMMARY
