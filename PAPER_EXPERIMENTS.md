# PIPER 论文实验运行手册

论文主设置固定为：

```text
presence_mode = soft
delta_presence = 0.0
delta_payload = 2.0
presence_test = exact_binomial
counting_mode = unique_context
primary_decoding_policy = tie_zero
detection_interface = blind_text
```

`delta_presence > 0`、`hard`、`all_tokens`、Z-score gate、strict/hard-fill/error-erasure 均属于消融，不得混入 PIPER 主结果。

## 0. 公共环境

```bash
cd /data/yanlu/BREW/PIPER_paper_aligned
conda activate dual-watermark-blackwell
export PYTHONNOUSERSITE=1

MODEL=/data/yanlu/BREW/models/facebook/opt-1.3b
C4=/data/datasets/c4_realnewslike.jsonl
KEY=dual-layer-key-2026
```

工程或数据位于其他目录时，只修改 `cd`、`MODEL` 和 `C4`。

## 1. 代码—论文一致性测试

目的：验证 child-only bias、unique-context、精确二项检验、实际 partition 概率、gate-before-decode 和 tie-to-zero。

```bash
pytest -q
LOCAL_OPT_MODEL="$MODEL" pytest -q tests/test_integration_local_model.py -s
```

这一步不产生论文结果；只有全部通过后再跑大实验。

## 2. 端到端 smoke test

目的：用少量样本验证生成、盲检、条件解码和输出字段。

```bash
python run_generation.py \
  --model-path "$MODEL" --secret-key "$KEY" \
  --dataset c4 --dataset-path "$C4" \
  --max-samples 8 --max-new-tokens 64 \
  --presence-mode soft --delta-presence 0 --delta-payload 2 \
  --presence-test exact_binomial --counting-mode unique_context \
  --primary-decoding-policy tie_zero \
  --run-id paper_smoke --overwrite

python run_detection.py \
  --input-type samples --run-dir outputs/paper_smoke \
  --presence-test exact_binomial --target-fpr 0.01 \
  --counting-mode unique_context --primary-decoding-policy tie_zero \
  --overwrite

python evaluate_results.py \
  --run-dir outputs/paper_smoke --input-mode blind_text
```

Smoke test 只用于工程验收，不放入论文表格。

## 3. 主无攻击实验

目的：在相同 prompt、seed、负样本和质量评估器下报告 FPR、TPR、CAR、CDA 与文本质量。

```bash
python -m experiments.run_pareto_sweep \
  --model-path "$MODEL" --secret-key "$KEY" \
  --experiment-id piper_main_t200_b8 --preset paper \
  --calibration-samples 1000 --test-samples 1000 \
  --exact-tokens 200 --presence-test exact_binomial \
  --counting-mode unique_context --dataset-path "$C4" \
  --stage all --overwrite
```

主结果读取 `runs/soft_p0_m2/metrics.json`。headline 使用 `blind_text`；`known_boundary` 只作为附录上界。

需要报告：

- `model_fpr`、`natural_fpr`、presence TPR；
- `correct_attribution_rate`（CAR）；
- `conditional_decoding_accuracy`（CDA）；
- tie-zero exact message recovery、message/bit accuracy；
- decoder abstention 与 wrong-message rate；
- continuation-only PPL/NLL、相对 PPL 增幅、生成时间。

## 4. Random-key 理论 null 验证

目的：检验随机密钥条件下 exact-binomial p-value 是否校准。输入必须是已经保存 token IDs 的自然文本或无水印模型输出。

下面示例假设 OPT-1.3B 的 `vocab_size=50272`，eligible vocabulary 排除 IDs `0 1`、保留 EOS；正式运行前必须以本地模型配置和 tokenizer 为准。

```bash
python -m experiments.run_paper_null \
  --input-file outputs/experiments/piper_main_t200_b8/shared/baseline.jsonl \
  --text-class natural --keys-file configs/paper_keys_20.txt \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --alphas 0.01 0.001 0.0001 \
  --output-dir outputs/paper_null/random_key_natural --overwrite

python -m experiments.run_paper_null \
  --input-file outputs/experiments/piper_main_t200_b8/shared/baseline.jsonl \
  --text-class unwatermarked --keys-file configs/paper_keys_20.txt \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --alphas 0.01 0.001 0.0001 \
  --output-dir outputs/paper_null/random_key_model --overwrite
```

`null_summary.json` 包含 combined 和 per-key FPR、false-positive 数及 Clopper–Pearson 95% 区间。

## 5. Fixed-key 部署 FPR

目的：检查固定部署密钥在自然文本、模型输出和域迁移数据上的真实 FPR。

```bash
python -m experiments.run_paper_null \
  --input-file /data/piper_null/c4_100k_token_ids.jsonl \
  --text-class natural --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --alphas 0.01 0.001 \
  --output-dir outputs/paper_null/fixed_key_c4_100k --overwrite

python -m experiments.run_paper_null \
  --input-file /data/piper_null/opengen_100k_token_ids.jsonl \
  --text-class natural --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --alphas 0.01 0.001 \
  --output-dir outputs/paper_null/fixed_key_opengen_100k --overwrite
```

验证 `alpha=1e-4` 时，固定密钥至少使用 1,000,000 条独立负样本：

```bash
python -m experiments.run_paper_null \
  --input-file /data/piper_null/c4_1m_token_ids.jsonl \
  --text-class natural --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --alphas 0.0001 \
  --output-dir outputs/paper_null/fixed_key_c4_1m --overwrite
```

不要用 1,000 个负样本声称验证 `1e-4` FPR。

## 6. Payload capacity 实验

目的：验证 presence FPR 不随 payload bits `b` 扩张，同时观察 payload recovery 随容量变化。

```bash
python -m experiments.run_capacity_fpr \
  --model-path "$MODEL" --secret-key "$KEY" \
  --dataset-path "$C4" --test-samples 1000 \
  --target-fpr 0.01 --output-dir outputs/paper_capacity_t500 \
  --overwrite
```

该脚本固定：`T=500`、`b in {8,16,32}`、`delta_presence=0`、`delta_payload=2`、blind-text、unique-context、exact-binomial、tie-zero。

## 7. 文本长度实验

目的：研究有效样本量、presence power 和 payload recovery 随 `T` 的变化。

```bash
python -m experiments.run_length_sweep \
  --model-path "$MODEL" --secret-key "$KEY" \
  --dataset-path "$C4" --test-samples 1000 \
  --t-values 100 200 300 500 1000 --b 8 \
  --delta-presence 0 --delta-payload 2 --target-fpr 0.01 \
  --output-dir outputs/paper_length_b8 --overwrite
```

## 8. 生成偏置与质量 Pareto 消融

目的：区分主点 `soft_p0_m2` 与额外 parent bias/hard mask。

```bash
python -m experiments.run_pareto_sweep \
  --model-path "$MODEL" --secret-key "$KEY" \
  --experiment-id piper_delta_ablation --preset full \
  --calibration-samples 1000 --test-samples 1000 \
  --exact-tokens 200 --presence-test exact_binomial \
  --counting-mode unique_context --dataset-path "$C4" \
  --stage all --overwrite
```

主文只突出 `soft_p0_m2`；`delta_presence>0` 和 `hard_*` 标为 ablation。

## 9. Counting、presence test 与 decoder 消融

目的：证明重复 context、Z 近似和更强 decoder 会改变经验结果，但不会改变主 PIPER 的定义。

```bash
# 主 PIPER
python run_detection.py --input-type samples --run-dir outputs/paper_smoke \
  --presence-test exact_binomial --counting-mode unique_context \
  --primary-decoding-policy tie_zero \
  --output-file detections_paper.jsonl --overwrite

# 重复 context 消融
python run_detection.py --input-type samples --run-dir outputs/paper_smoke \
  --presence-test exact_binomial --counting-mode all_tokens \
  --primary-decoding-policy tie_zero \
  --output-file detections_all_tokens.jsonl --overwrite

# Z-score 消融
python run_detection.py --input-type samples --run-dir outputs/paper_smoke \
  --presence-test z_score --threshold-mode theoretical \
  --counting-mode unique_context --primary-decoding-policy tie_zero \
  --output-file detections_z_score.jsonl --overwrite

# Decoder 消融；presence gate 保持不变
for POLICY in strict hard_fill error_erasure; do
  python run_detection.py --input-type samples --run-dir outputs/paper_smoke \
    --presence-test exact_binomial --counting-mode unique_context \
    --primary-decoding-policy "$POLICY" \
    --output-file "detections_${POLICY}.jsonl" --overwrite
done
```

## 10. 编辑与改写攻击

目的：报告攻击后的 presence TPR、CAR、CDA 和文本长度变化。主 headline 必须使用 blind text。

```bash
python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/piper_main_t200_b8 \
  --point-id soft_p0_m2 --attack-rate 0.10 \
  --attacks replacement deletion insertion \
  --headline-input-mode blind_text --input-modes blind_text \
  --overwrite

python -m experiments.attack.run_paraphrase_attacks \
  --experiment-dir outputs/experiments/piper_main_t200_b8 \
  --point-id soft_p0_m2 \
  --headline-input-mode blind_text --input-modes blind_text \
  --paraphraser-model /data/models/tuner007/pegasus_paraphrase \
  --overwrite
```

## 11. 对比方法

最低限度应复现与 FPR 论点最接近的三类：Fu–Russell Dual Watermark、Every Bit/MPAC、BREW。工程已有 MPAC 和 Segment/RS-BH 适配器：

```bash
python -m experiments.mpac_comparison \
  --experiment-dir outputs/experiments/piper_main_t200_b8 \
  --mb-repo /data/repos/mb-lm-watermarking \
  --output-dir outputs/baselines/mpac --model-path "$MODEL" \
  --brew-point-id soft_p0_m2 --exact-tokens 200 \
  --message-length 8 --target-fpr 0.01 --overwrite

python -m experiments.segment_rsbh_comparison \
  --experiment-dir outputs/experiments/piper_main_t200_b8 \
  --segment-repo /data/repos/segment-watermark \
  --output-dir outputs/baselines/segment_rsbh --model-path "$MODEL" \
  --brew-point-id soft_p0_m2 --exact-tokens 200 \
  --message-length 8 --target-fpr 0.01 --overwrite
```

Fu–Russell 和论文版 BREW 的 blind protocol 需要其对应公开实现。不要把当前 BREW 仓库中依赖 generation-time `codeword_queue` 的检测结果当作论文所述 key-only blind verification。

## 12. 论文主表建议

```text
FPR_pres | TPR_pres | CAR | CDA | tie-zero EMR | bit accuracy
model/natural spurious-attribution rate | wrong-message rate | abstention rate
N_eff | PPL / relative PPL increase
```

所有 FPR 必须同时给出负样本数和 95% 置信区间；多密钥实验同时报告 per-key 分布，不能只报 key-averaged FPR。
