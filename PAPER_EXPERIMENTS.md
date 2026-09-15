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
seeding_scheme = selfhash
partition_engine = v2
candidate_top_k = 50
sampling_top_k = 50
sampling_top_p = 0.95
```

`delta_presence > 0`、`hard`、`all_tokens`、Z-score gate、strict/hard-fill/error-erasure 均属于消融，不得混入 PIPER 主结果。

## 0. 公共环境

```bash
cd /data/yanlu/BREW/PIPER
conda activate dual-watermark-blackwell
export PYTHONNOUSERSITE=1

MODEL=/data/yanlu/BREW/models/facebook/opt-1.3b
C4=/data/yanlu/BREW/dataset/c4/processed_c4.json
KEY=dual-layer-key-2026
BATCH_SIZE=${BATCH_SIZE:-16}
```

工程或数据位于其他目录时，只修改 `cd`、`MODEL` 和 `C4`。

PIPER 生成默认按批使用 GPU。`BATCH_SIZE=16` 是 OPT-1.3B 的稳妥起点；显存充足可在命令前设置 `BATCH_SIZE=32` 或更高。若发生 CUDA OOM，程序会自动减半批量并重试，直至批量为 1；样本级随机种子不随批量或断点位置变化。

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
  --generation-batch-size "$BATCH_SIZE" \
  --context-width 4 --candidate-top-k 50 \
  --seeding-scheme selfhash --partition-engine v2 \
  --top-k 50 --top-p 0.95 --temperature 1.0 \
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
  --experiment-id piper_main_t300_b8_selfhash_top50 --preset paper \
  --calibration-samples 1000 --test-samples 1000 \
  --exact-tokens 300 --context-width 4 \
  --ecc-n 23 --ecc-k 8 --ecc-t 3 \
  --candidate-top-k 50 --seeding-scheme selfhash --partition-engine v2 \
  --temperature 1.0 --top-k 50 --top-p 0.95 \
  --presence-test exact_binomial \
  --generation-batch-size "$BATCH_SIZE" \
  --counting-mode unique_context --dataset-path "$C4" \
  --only-point soft_p0_m2 --stage all --overwrite
```

主结果读取 `runs/soft_p0_m2/metrics.json`。headline 使用 `blind_text`；`known_boundary` 只作为附录上界。

该 exact-binomial 主路径不会计算、写入或读取 Z-score 校准阈值。`--calibration-samples` 保留独立样本，供 Z-score 消融和需要经验阈值的对比方法使用；它们不参与 PIPER 主判定。

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
  --input-file outputs/experiments/piper_main_t300_b8_selfhash_top50/shared/baseline.jsonl \
  --text-class natural --keys-file configs/paper_keys_20.txt \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --context-width 4 --seeding-scheme selfhash --partition-engine v2 \
  --alphas 0.01 0.001 0.0001 \
  --output-dir outputs/paper_null/random_key_natural --overwrite

python -m experiments.run_paper_null \
  --input-file outputs/experiments/piper_main_t300_b8_selfhash_top50/shared/baseline.jsonl \
  --text-class unwatermarked --keys-file configs/paper_keys_20.txt \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --context-width 4 --seeding-scheme selfhash --partition-engine v2 \
  --alphas 0.01 0.001 0.0001 \
  --output-dir outputs/paper_null/random_key_model --overwrite
```

`null_summary.json` 包含 combined 和 per-key FPR、false-positive 数及 Clopper–Pearson 95% 区间。

## 5. Large-scale empirical FPR calibration（核心必做实验 A）

目的：验证闭式 exact-binomial presence test 在真实文本、固定部署密钥和域迁移条件下是否满足

```text
Pr_0(I = 1) <= alpha
```

该实验是 PIPER 核心 FPR claim 的正文证据，不得用第 4 节的 random-key 重采样结果替代。至少覆盖四类相互独立的负样本：human C4、OpenGen human-like、OPT 无水印输出和 Llama 无水印输出。所有文本必须预先保存原始 token IDs，并使用与正式检测完全相同的 tokenizer、特殊 token 排除规则、blind-text 接口和 unique-context 计数。

100k/来源用于稳定评估 `alpha in {1e-2, 1e-3}`：

```bash
python -m experiments.run_paper_null \
  --input-file /data/piper_null/c4_100k_token_ids.jsonl \
  --text-class natural --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --context-width 4 --seeding-scheme selfhash --partition-engine v2 \
  --alphas 0.01 0.001 \
  --output-dir outputs/paper_null/fixed_key_c4_100k --overwrite

python -m experiments.run_paper_null \
  --input-file /data/piper_null/opengen_100k_token_ids.jsonl \
  --text-class natural --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --context-width 4 --seeding-scheme selfhash --partition-engine v2 \
  --alphas 0.01 0.001 \
  --output-dir outputs/paper_null/fixed_key_opengen_100k --overwrite

python -m experiments.run_paper_null \
  --input-file /data/piper_null/opt_unwatermarked_100k_token_ids.jsonl \
  --text-class unwatermarked --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --context-width 4 --seeding-scheme selfhash --partition-engine v2 \
  --alphas 0.01 0.001 \
  --output-dir outputs/paper_null/fixed_key_opt_100k --overwrite

python -m experiments.run_paper_null \
  --input-file /data/piper_null/llama_unwatermarked_100k_token_ids.jsonl \
  --text-class unwatermarked --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --context-width 4 --seeding-scheme selfhash --partition-engine v2 \
  --alphas 0.01 0.001 \
  --output-dir outputs/paper_null/fixed_key_llama_100k --overwrite
```

验证 `alpha=1e-4` 时，每个声称通过验证的数据域至少使用 1,000,000 条独立负样本。下面只运行 C4，因此只能声称 C4 上的 `1e-4` 结果；若要做跨域 `1e-4` 结论，必须把 OpenGen、OPT 和 Llama 也分别扩展到 1M：

```bash
python -m experiments.run_paper_null \
  --input-file /data/piper_null/c4_1m_token_ids.jsonl \
  --text-class natural --secret-key "$KEY" \
  --vocab-size 50272 --excluded-token-ids 0 1 \
  --context-width 4 --seeding-scheme selfhash --partition-engine v2 \
  --alphas 0.0001 \
  --output-dir outputs/paper_null/fixed_key_c4_1m --overwrite
```

统计与报告要求：

- PIPER 使用理论 exact-binomial 阈值，不得在这些测试负样本上重新选择阈值；
- 需要经验校准的 baseline 必须使用独立 calibration split，再在上述共同 test split 上评估；
- 每个方法、来源和 `alpha` 报告负样本数、false-positive 数、经验 FPR 和 Clopper--Pearson 95% 区间；
- 观察到 0 个 false positive 也不能写成 `FPR=0`，必须报告置信区间上界；
- 不要用 1,000 个负样本声称验证 `1e-4` FPR。

正文 Figure：`Nominal FPR vs Empirical FPR`，横纵轴均采用 log scale，包含理想对角线、PIPER 及能够在同一协议下校准的 MPAC/EveryBit/BREW 结果，并绘制 95% CI。

正文 Table：

```text
Method | Null source | N | Nominal alpha | False positives | Empirical FPR | 95% CI
```

## 6. Shared evidence vs disjoint evidence（核心必做实验 B）

目的：证明 PIPER 的优势来自同一 token 上的 nested/shared evidence，而不是单纯增加 watermark strength。核心比较为：

```text
PIPER shared:
100% token positions -> parent presence evidence
                      -> conditional child payload evidence

Disjoint 50/50:
50% token positions -> presence-only indicator
50% token positions -> payload-only evidence
```

公平性要求：

- 使用完全相同的 prompt、payload、seed、负样本、continuation 长度和 tokenizer；
- 主比较采用 matched-quality operating points，在相同 continuation-only PPL/relative PPL increase 下比较；
- 补充 matched-bias 比较，两种方法的有效 bias 均以 `delta=2.0` 为基准；
- 固定 payload bits、ECC、目标 FPR、blind-text、unique-context 和 exact-binomial；
- disjoint token routing 必须在观察生成 token 之前由密钥确定，不得按结果选择位置；
- 主要使用 50/50 split，可在附录补充 25/75 与 75/25 split；
- presence 与 payload 的 `N_eff` 必须分别报告，不能只报告总 token 数。

当前仓库尚未实现 disjoint-indicator 生成器和统一比较入口。必须先实现 `experiments.run_shared_disjoint_comparison`，再运行下面的目标命令；在实现完成前不得把该命令标记为可复现实验：

```bash
python -m experiments.run_shared_disjoint_comparison \
  --model-path "$MODEL" --secret-key "$KEY" \
  --dataset-path "$C4" --test-samples 1000 \
  --methods piper_shared disjoint_50_50 \
  --t-values 100 200 300 500 --payload-bits 8 \
  --delta 2.0 --target-fpr 0.001 \
  --counting-mode unique_context --input-mode blind_text \
  --quality-match relative_ppl \
  --output-dir outputs/shared_vs_disjoint --overwrite
```

主要指标：

- `TPR_pres@FPR=1e-3`；
- CAR、CDA 和 exact message recovery；
- presence/payload `N_eff` 与达到固定 TPR/CAR 所需的生成 token 数；
- continuation-only PPL、relative PPL increase 和生成时间。

正文 Figure：`Detection power vs generated-token budget`，横轴为总生成 token 数，纵轴为 `TPR_pres@1e-3`；同图或第二子图报告 `CAR@1e-3`。正文 Table 报告 matched-quality operating point。split-ratio sweep 放附录。

## 7. Decoder independence（核心必做实验 C）

目的：直接验证 presence gate 与 payload decoder 解耦。必须固定同一批已生成文本和完全相同的 presence 配置，只改变下游 decoder；不得为不同 decoder 重新生成文本或重新选择 presence 阈值。

```bash
PIPER_MAIN_RUN=outputs/experiments/piper_main_t300_b8_selfhash_top50/runs/soft_p0_m2

for POLICY in tie_zero strict hard_fill error_erasure; do
  python run_detection.py --input-type samples --run-dir "$PIPER_MAIN_RUN" \
    --presence-test exact_binomial --target-fpr 0.001 \
    --counting-mode unique_context --primary-decoding-policy "$POLICY" \
    --output-file "detections_decoder_${POLICY}.jsonl" --overwrite

  python evaluate_results.py --run-dir "$PIPER_MAIN_RUN" \
    --detections-file "detections_decoder_${POLICY}.jsonl" \
    --input-mode blind_text

  cp "$PIPER_MAIN_RUN/metrics.json" \
    "$PIPER_MAIN_RUN/metrics_decoder_${POLICY}.json"
done
```

逐样本一致性验收：四个 detection 文件中的下列 presence 字段必须逐项完全一致：

```text
sample_id | text_class | input_mode | num_scored_tokens | upper_count
binomial_p_value | alpha | threshold | detected | decoder_invoked
```

`decoder_invoked` 只由 presence gate 是否接受决定，也必须一致。任何 presence p-value、`detected` 或 FPR 差异都表示实现错误，而不是实验现象。允许随 decoder 改变的只有 payload recovery、CDA、wrong-message rate 和 abstention/decode-failure rate。

正文或主文紧邻理论保证的 Figure 使用两个子图：

- (a) `FPR_pres` 和 `TPR_pres` vs decoder：应完全重合；
- (b) CDA、wrong-message rate 和 abstention rate vs decoder：可以不同。

表格同时报告四种 decoder 的 `FPR_pres`、`TPR_pres`、CAR、CDA、wrong-message rate 和 abstention rate。该实验直接支撑 `decoder-agnostic false-positive cap`，不能只用 smoke test 结果。

## 8. Payload capacity 实验

目的：验证 presence FPR 不随 payload bits `b` 扩张，同时观察 payload recovery 随容量变化。

```bash
python -m experiments.run_capacity_fpr \
  --model-path "$MODEL" --secret-key "$KEY" \
  --dataset-path "$C4" --test-samples 1000 \
  --generation-batch-size "$BATCH_SIZE" \
  --context-width 4 --candidate-top-k 50 \
  --seeding-scheme selfhash --partition-engine v2 \
  --temperature 1.0 --top-k 50 --top-p 0.95 \
  --target-fpr 0.01 --output-dir outputs/paper_capacity_t500 \
  --overwrite
```

该脚本固定：`T=500`、`b in {8,16,32}`、`delta_presence=0`、`delta_payload=2`、blind-text、unique-context、exact-binomial、tie-zero。

## 9. 文本长度实验

目的：研究有效样本量、presence power 和 payload recovery 随 `T` 的变化。

```bash
python -m experiments.run_length_sweep \
  --model-path "$MODEL" --secret-key "$KEY" \
  --dataset-path "$C4" --test-samples 1000 \
  --t-values 100 200 300 500 1000 --b 8 \
  --generation-batch-size "$BATCH_SIZE" \
  --context-width 4 --candidate-top-k 50 \
  --seeding-scheme selfhash --partition-engine v2 \
  --temperature 1.0 --top-k 50 --top-p 0.95 \
  --delta-presence 0 --delta-payload 2 --target-fpr 0.01 \
  --output-dir outputs/paper_length_b8 --overwrite
```

## 10. 生成偏置与质量 Pareto 消融

目的：区分主点 `soft_p0_m2` 与额外 parent bias/hard mask。

```bash
python -m experiments.run_pareto_sweep \
  --model-path "$MODEL" --secret-key "$KEY" \
  --experiment-id piper_delta_ablation --preset full \
  --calibration-samples 1000 --test-samples 1000 \
  --exact-tokens 300 --context-width 4 \
  --candidate-top-k 50 --seeding-scheme selfhash --partition-engine v2 \
  --temperature 1.0 --top-k 50 --top-p 0.95 \
  --presence-test exact_binomial \
  --generation-batch-size "$BATCH_SIZE" \
  --counting-mode unique_context --dataset-path "$C4" \
  --stage all --overwrite
```

主文只突出 `soft_p0_m2`；`delta_presence>0` 和 `hard_*` 标为 ablation。

## 11. Counting 与 presence test 消融

目的：证明重复 context 和 Z 近似会改变经验结果，但不会改变主 PIPER 的定义。decoder independence 已提升为第 7 节核心实验，不在这里重复。

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

# Z-score 理论阈值消融
python run_detection.py --input-type samples --run-dir outputs/paper_smoke \
  --presence-test z_score --threshold-mode theoretical \
  --counting-mode unique_context --primary-decoding-policy tie_zero \
  --output-file detections_z_score.jsonl --overwrite

# Z-score 经验阈值消融：先在独立负样本检测文件上校准
python calibrate_z_threshold.py --run-dir outputs/paper_smoke \
  --detections-file detections.jsonl --negative-source combined \
  --input-mode blind_text --target-fpr 0.01

python run_detection.py --input-type samples --run-dir outputs/paper_smoke \
  --presence-test z_score --threshold-mode calibrated \
  --calibration-file outputs/paper_smoke/z_calibration.json \
  --counting-mode unique_context --primary-decoding-policy tie_zero \
  --output-file detections_z_calibrated.jsonl --overwrite
```

## 12. 编辑与改写攻击

目的：报告攻击后的 presence TPR、CAR、CDA 和文本长度变化。主 headline 必须使用 blind text。

```bash
python -m experiments.attack.run_synonym_attacks \
  --experiment-dir outputs/experiments/piper_main_t300_b8_selfhash_top50 \
  --point-id soft_p0_m2 --attack-rate 0.10 \
  --attacks replacement deletion insertion \
  --headline-input-mode blind_text --input-modes blind_text \
  --overwrite

python -m experiments.attack.run_paraphrase_attacks \
  --experiment-dir outputs/experiments/piper_main_t300_b8_selfhash_top50 \
  --point-id soft_p0_m2 \
  --headline-input-mode blind_text --input-modes blind_text \
  --paraphraser-model /data/models/tuner007/pegasus_paraphrase \
  --overwrite
```

## 13. 对比方法

最低限度应复现与 FPR 论点最接近的三类：Fu–Russell Dual Watermark、Every Bit/MPAC、BREW。工程已有 MPAC 和 Segment/RS-BH 适配器：

```bash
python -m experiments.mpac_comparison \
  --experiment-dir outputs/experiments/piper_main_t300_b8_selfhash_top50 \
  --mb-repo /data/repos/mb-lm-watermarking \
  --output-dir outputs/baselines/mpac --model-path "$MODEL" \
  --brew-point-id soft_p0_m2 --exact-tokens 300 \
  --message-length 8 --target-fpr 0.01 --overwrite

python -m experiments.segment_rsbh_comparison \
  --experiment-dir outputs/experiments/piper_main_t300_b8_selfhash_top50 \
  --segment-repo /data/repos/segment-watermark \
  --output-dir outputs/baselines/segment_rsbh --model-path "$MODEL" \
  --brew-point-id soft_p0_m2 --exact-tokens 300 \
  --message-length 8 --target-fpr 0.01 --overwrite
```

Fu–Russell 和论文版 BREW 的 blind protocol 需要其对应公开实现。不要把当前 BREW 仓库中依赖 generation-time `codeword_queue` 的检测结果当作论文所述 key-only blind verification。

## 14. 论文主表建议

```text
FPR_pres | TPR_pres | CAR | CDA | tie-zero EMR | bit accuracy
model/natural spurious-attribution rate | wrong-message rate | abstention rate
N_eff | PPL / relative PPL increase
```

所有 FPR 必须同时给出负样本数和 95% 置信区间；多密钥实验同时报告 per-key 分布，不能只报 key-averaged FPR。
