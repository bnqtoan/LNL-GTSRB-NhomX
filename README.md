# LNL-GTSRB — Improved (Deep Learning giữa kỳ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bnqtoan/LNL-GTSRB/blob/main/Instructions.ipynb)

Cải tiến mô hình nhận dạng biển báo giao thông **GTSRB** dựa trên
**LNL (Locality-iN-Locality)** — Transformer-in-Transformer + cơ chế locality.
Mục tiêu chấm điểm: **Top-1 ≥ 99.5%**.

> **Academic fork.** Mã gốc thuộc về **Omid Nejati Manzari** (xem Citation cuối trang).
> Repo này chỉ dùng cho mục đích học tập.

## Cách chạy

Bấm **Open In Colab** ở trên (mở `Instructions.ipynb`) → Runtime → Change runtime type → **GPU (T4)**
→ **Run all** (~1.5–2h). Số cần báo cáo là **`Standard accuracy`** ở cell Test.

`Instructions.ipynb` chính là notebook gốc của bài báo, chỉ sửa **đường huấn luyện của LNL** (mỗi chỗ sửa
đều có comment `# EDIT:` giải thích lý do); các phần khác (tải dữ liệu, dựng model, MoEx, FLOPs) giữ nguyên.

## Cải tiến nằm ở 2 nơi

**1. Trong `LNL.py` (kiến trúc + tiền xử lý):**
- Chuẩn hoá đầu vào ngay trong model (ImageNet mean/std) — ảnh vào để raw `[0,1]`, model tự normalize.
- Augmentation nhẹ trong model, chỉ bật khi `self.training` (tự tắt lúc test); không lật ngang (biển báo nhạy hướng).
- LayerScale (CaiT) trên mỗi nhánh residual; `qkv_bias=True`; stochastic depth nhẹ.
- Đặc trưng phân loại = CLS token + mean-pool các patch token (giữ `head = Linear(192, 43)`).

**2. Trong `Instructions.ipynb` (công thức huấn luyện — phần tạo ra phần lớn độ chính xác):**
- Optimizer **AdamW** (lr 5e-4, weight_decay 0.05) thay cho SGD.
- **Cosine LR + warmup**, **label smoothing 0.1**, **mixed precision (AMP)**.
- **EMA** trọng số (chọn bản tốt hơn giữa raw / EMA).
- **20 epoch**, batch 64.
- **TTA đa tỉ lệ** (1.0 / 0.9 / 1.1) lúc test.

> Vì sao: SGD lr=0.007, 5 epoch từ đầu chỉ đạt ~97–98% dù chỉnh kiến trúc thế nào; AdamW + EMA + TTA mới
> đưa được lên ≥ 99.5%. Do đó cải tiến phải gồm cả công thức huấn luyện, không chỉ riêng `LNL.py`.

## Files

| File | Mô tả |
|------|-------|
| `LNL.py` | Model đã cải tiến (in-model normalize + augment, LayerScale, qkv_bias, CLS+mean-pool) |
| `Instructions.ipynb` | Notebook gốc + công thức AdamW/EMA/TTA (mỗi sửa đổi có comment `# EDIT:`) |
| `LNL_train_colab.ipynb` | (Tuỳ chọn) notebook train độc lập có checkpoint-resume trên Google Drive |
| `dien_giai_mo_hinh.md` | Diễn giải ngắn gọn mô hình & công thức |

---

## Original work / Citation

This repository is an academic fork of **"Locality iN Locality"** by Omid Nejati Manzari et al.
Original repo: https://github.com/Omid-Nejati/Locality-iN-Locality · Paper: https://arxiv.org/abs/2301.11553

```
@article{manzari2023robust,
  title={Robust transformer with locality inductive bias and feature normalization},
  author={Manzari, Omid Nejati and Kashiani, Hossein and Dehkordi, Hojat Asgarian and Shokouhi, Shahriar B},
  journal={Engineering Science and Technology, an International Journal},
  volume={38}, pages={101320}, year={2023}, publisher={Elsevier}
}
```
