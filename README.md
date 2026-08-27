# Lab 1 v2 — FashionMNIST Optimizer Benchmark: Best weight theo val F1

**Khác biệt so với [v1](https://github.com/thanh1912-ut/lab1-nhom8)** — giả thuyết: chọn best weight theo **val loss** không tối ưu vì loss nhạy với over-confident predictions; v2 chọn theo **val F1 macro**.

| | v1 (arm A) | **v2 (arm B)** |
|---|---|---|
| Best weight | theo val **loss** nhỏ nhất | theo **val F1 macro** cao nhất |
| Early stop | val loss, patience 30 | **val F1**, patience **10** |
| Validate | mỗi 5 epoch | **mỗi epoch** |
| Max epochs | 150 | 150 |

Còn lại giữ nguyên: 8 optimizers × 3 LRs × 3 seeds × MLP/CNN = 144 runs, batch 64, Train/Val/Test 54k/6k/10k, log YOLO-style, metrics acc/pre/rec/F1 + confusion matrix, TensorBoard, checkpoint resume mỗi 200 batch, auto-push GitHub.

## Selection config
```yaml
selection:
  best_weight_by: f1     # loss | acc | f1
  early_stop_by: f1      # loss | acc | f1
val_every: 1
early_stop_patience: 10
```
Các metric val (loss, acc, pre, rec, f1) đều được ghi đầy đủ vào history/TensorBoard ở **mọi epoch** — nên sau này có thể post-analyze theo tiêu chí khác mà không cần train lại.

## Cấu trúc
```
├── configs/{mlp,cnn}.yaml    # selection, val_every=1, patience=10
├── src/                      # data, models, metrics, logger, train, benchmark, gitbackup
├── train_mlp.py / train_cnn.py
├── kaggle_run.ipynb          # dual T4 song song (MLP→GPU0, CNN→GPU1)
└── outputs/                  # runs/, tensorboard/, results_{mlp,cnn}.json
```

## Chạy trên Kaggle (dual T4)
1. PAT (scope `repo`) → Kaggle Secrets `GITHUB_TOKEN` (push cần token kể cả repo public)
2. Settings: **GPU T4 x2** + Internet
3. Notebook clone repo v2, pre-download data, launch song song 2 process (log `mlp_train.out`/`cnn_train.out`), cells xem log live / TensorBoard / đợi hoàn tất / push manual
4. Session chết → chạy lại notebook: tự skip run xong, tự resume run dở từ checkpoint (mỗi 200 batch)

## Chạy local
```bash
pip install -r requirements.txt
python train_mlp.py            # hoặc --only adam
python train_cnn.py
tensorboard --logdir outputs/tensorboard
```

## Kết quả & so sánh v1 vs v2

*(cập nhật sau khi train xong — bảng so sánh test acc/F1 của best-loss-weight (v1) vs best-F1-weight (v2) cùng (optimizer, lr, seed))*

## Mapping yêu cầu lab
| Yêu cầu | Đáp ứng |
|---|---|
| Load FashionMNIST + transforms | `src/data.py` |
| Build neural network | `src/models.py` (MLP + CNN) |
| Training loop | `src/train.py` |
| Evaluate accuracy | `src/metrics.py` (acc/pre/rec/F1, CM) |
| Save/load model | best.pt/last.pt + verify khi test |
| Experiment hyperparams | 8 opt × 3 lr × 3 seed × 2 model |
| Visualize loss | TensorBoard + train.log |
| Predicted vs actual | `cm_test.png` + cm_best.png |
