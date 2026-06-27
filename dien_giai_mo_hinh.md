# Diễn giải mô hình đề xuất — Nhận dạng biển báo giao thông (GTSRB)

**Mô hình nền:** LNL (Locality-iN-Locality) — Transformer-in-Transformer (TNT) + cơ chế locality.
**Bộ dữ liệu:** GTSRB — 43 lớp biển báo giao thông Đức (ảnh 224×224).
**Kết quả:** **Top-1 = 99.66%** (raw 99.59% / EMA 99.60%, cộng TTA → 99.66%).

---

## 1. Sản phẩm nộp
- `LNL.py` — file mô hình đã cải tiến.
- `lnl_gtsrb.pth` — trọng số đã train (nạp lại vào model mới đạt **99.60%** không cần train lại — đã kiểm chứng trong notebook).
- `Instructions.ipynb` — notebook gốc, chỉ sửa đường huấn luyện của LNL (mỗi chỗ sửa có comment `# EDIT:`).
- Ảnh chụp kết quả test (`Standard accuracy: 99.66 %`).

## 2. Cải tiến nằm ở 2 nơi

**A. Trong `LNL.py` (kiến trúc + tiền xử lý):**
| # | Cải tiến | Vì sao |
|---|----------|--------|
| 1 | Chuẩn hoá đầu vào ImageNet **trong model** (`forward_features`) | Ảnh vào để raw `[0,1]`; chuẩn hoá trong model giúp hội tụ nhanh và để trọng số plug-and-play. |
| 2 | Augmentation nhẹ trong model, chỉ khi `self.training` (affine; tự tắt lúc test) | Tăng bền vững, không lật ngang (biển báo nhạy hướng). |
| 3 | **LayerScale** (CaiT) init = 1.0 trên mỗi nhánh residual | Nhánh hoạt động đầy đủ từ đầu, ổn định Transformer sâu. |
| 4 | **qkv_bias = True** | Tăng biểu diễn, gần như miễn phí. |
| 5 | Stochastic depth nhẹ (drop_path) | Regularization. |
| 6 | Đặc trưng phân loại = **CLS token + mean-pool patch tokens** (cộng, vẫn 192-d) | Mean-pool cho đặc trưng mạnh sớm; giữ `head = Linear(192,43)` plug-and-play. |

**B. Trong `Instructions.ipynb` (công thức huấn luyện — tạo ra phần lớn độ chính xác):**
- Optimizer **AdamW** (lr 5e-4, weight_decay 0.05) thay SGD.
- **Cosine LR + warmup**, **label smoothing 0.1**, **mixed precision (AMP)**.
- **EMA** trọng số (chọn bản tốt hơn giữa raw / EMA → chọn EMA).
- **20 epoch**, batch 64.
- **TTA đa tỉ lệ** (1.0 / 0.9 / 1.1) lúc test.

> **Vì sao cần công thức huấn luyện, không chỉ kiến trúc:** với vòng gốc (SGD lr=0.007, 5 epoch) mô hình
> chỉ ~97.7% và mọi chỉnh kiến trúc cũng chỉ tới ~98.6% (do thiếu bước hội tụ). AdamW + cosine + EMA + TTA
> mới đưa lên ≥ 99.5%. Đầu vào vẫn raw `[0,1]` (chuẩn hoá nằm trong `LNL.py`) nên train và test khớp tiền xử lý.

## 3. Kết quả

| Cấu hình | Top-1 |
|----------|-------|
| LNL gốc (5 epoch SGD, không aug/normalize) | ~97% |
| **LNL.py cải tiến + công thức AdamW/EMA/TTA (20 epoch)** | **99.66%** |
| Nạp lại `lnl_gtsrb.pth` vào model mới (không TTA) — kiểm chứng | **99.60%** |

(Tham khảo robustness: FGSM 91.48%, PGD 55.68%.)

## 4. Thiết lập
- Model: LNL-Ti (TNT) + in-model normalize + in-model augment (train-only) + LayerScale + qkv_bias + CLS/mean-pool, head = Linear(192, 43).
- Phần cứng: 1× NVIDIA T4 (Colab). Ảnh 224×224 (positional embedding cố định).
