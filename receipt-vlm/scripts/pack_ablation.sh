#!/usr/bin/env bash
# Pack the ablation results into ONE archive and print the numbers to the terminal, so they
# survive even if the download fails. Download the archive before deleting the GPU server.
#   bash scripts/pack_ablation.sh /workspace/ablation_results.tgz
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-/workspace/ablation_results.tgz}"

paths=()
for p in outputs/preds outputs/metrics outputs/sft_control outputs/grpo_base \
         reports/RESULTS.md reports/figures; do
  if [[ -e $p ]]; then paths+=("$p"); fi
done
tar czhf "$out" --exclude='*/checkpoints' --exclude='*/trainer' "${paths[@]}"

echo
echo "=== Archive: $out ($(du -h "$out" | cut -f1), $(tar tzf "$out" | wc -l) files)"
for d in outputs/sft_control outputs/grpo_base; do
  if [[ -f $d/train_info.json ]]; then echo "$d train: $(tr -d '\n' < "$d/train_info.json")"; fi
  if [[ -f $d/best/selection.json ]]; then
    echo "$d best: $(grep -o '"tag": "[^"]*"' "$d/best/selection.json")"
  fi
  if [[ -f $d/reward_hacking_warning.json ]]; then
    echo "$d reward-hacking flag: $(tr -d '\n' < "$d/reward_hacking_warning.json")"
  fi
done
echo
echo "=== Val results (copy this text if the download fails)"
awk '/^### Headline metrics \(val/{on=1} on&&/^!\[/{exit} on' reports/RESULTS.md
grep -E '^\| (M3|C1|M2) vs ' reports/RESULTS.md || true
echo
echo "Next: download $out (JupyterLab: right-click -> Download), check that its size on your"
echo "computer matches the size above, and only then terminate the pod."
