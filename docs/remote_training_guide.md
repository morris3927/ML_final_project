# 🚀 遠端訓練完整流程指南

從 Git clone 到開始訓練的完整步驟。

---

## 📋 前置需求

- ✅ 遠端主機有 GPU（推薦）或 CPU
- ✅ Python 3.8+
- ✅ Git 已安裝
- ✅ 有足夠的儲存空間（建議 10GB+）

---

## 🔄 完整流程（6 步驟）

### 步驟 1️⃣：Clone 專案

```bash
# SSH 到遠端主機
ssh user@remote-server

# Clone 專案
cd ~/projects  # 或你想要的目錄
git clone <your-repo-url>
cd 期末專案

# 確認檔案結構
ls -la
```

---

### 步驟 2️⃣：設置 Python 環境

```bash
# 創建虛擬環境
python3 -m venv venv

# 啟動環境
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 驗證安裝
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### 步驟 3️⃣：下載 THETIS 資料集

**選項 A：直接在遠端主機下載（推薦）**

```bash
# 執行下載腳本
./scripts/download_thetis.sh

# 或手動下載
mkdir -p ~/Downloads/thetis_rgb
cd ~/Downloads/thetis_rgb
git init
git config core.sparseCheckout true
echo "VIDEO_RGB" >> .git/info/sparse-checkout
git remote add origin https://github.com/THETIS-dataset/dataset.git
git pull origin main

# 回到專案目錄
cd ~/projects/期末專案
```

**選項 B：從本地上傳（如果遠端網路慢）**

```bash
# 在你的 Mac 上
scp -r ~/Downloads/thetis_rgb/VIDEO_RGB \
    user@remote-server:~/Downloads/thetis_rgb/
```

---

### 步驟 4️⃣：整理資料到專案

```bash
# 檢查 THETIS 下載的影片
ls ~/Downloads/thetis_rgb/VIDEO_RGB/

# 運行整理工具（查看指引）
python3 scripts/organize_thetis.py \
    --thetis_dir ~/Downloads/thetis_rgb

# 根據 THETIS 標註，手動整理影片到對應類別
# 範例：
cp ~/Downloads/thetis_rgb/VIDEO_RGB/serve_*.mp4 data/raw/tennis/flat_service/
cp ~/Downloads/thetis_rgb/VIDEO_RGB/smash_*.mp4 data/raw/tennis/smash/
# ... 其他類別

# 或如果你已經在本地整理好
# 從本地上傳整理好的資料
scp -r data/raw/tennis user@remote-server:~/projects/期末專案/data/raw/
```

**驗證資料**：
```bash
# 檢查每個類別的影片數量
for dir in data/raw/tennis/*/; do
    echo "$(basename "$dir"): $(ls -1 "$dir" 2>/dev/null | wc -l) videos"
done
```

預期輸出：
```
flat_service: 20 videos
slice_service: 15 videos
smash: 25 videos
... 等
```

---

### 步驟 5️⃣：預處理資料

```bash
# 啟動虛擬環境（如果還沒啟動）
source venv/bin/activate

# 運行預處理（這會花一些時間）
python3 src/data/preprocess_videos.py \
    --raw_dir data/raw/tennis \
    --output_dir data/processed/tennis \
    --split_ratio 0.7 0.15 0.15

# 預期輸出：
# Processing category: flat_service
#   Found 20 videos
#   Split: Train=14, Val=3, Test=3
#   train: 100%|████████| 14/14
# ...
```

**驗證預處理結果**：
```bash
# 檢查處理後的資料
ls -lh data/processed/tennis/train/
ls -lh data/processed/tennis/train/flat_service/
```

---

### 步驟 6️⃣：開始訓練 🎯

#### 測試訓練（小規模驗證）

```bash
# 先用測試配置確認流程正常（使用 GPU 1）
CUDA_VISIBLE_DEVICES=1 python3 src/train.py --config configs/experiments/test_small.yaml
```

#### 正式訓練（使用 GPU 1）

```bash
# 網球 7 類訓練
CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name "tennis_7class_baseline"

# 使用 nohup 背景執行（推薦）⭐
CUDA_VISIBLE_DEVICES=1 nohup python3 src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name "tennis_7class_baseline" \
    > training.log 2>&1 &

# 查看訓練進度
tail -f training.log

# 或用 screen/tmux
screen -S training
CUDA_VISIBLE_DEVICES=1 python3 src/train.py --config configs/experiments/tennis_baseline.yaml
# Ctrl+A, D 離開 screen
# screen -r training  # 重新連接
```

---

## 📊 訓練進度監控

### 查看即時輸出

```bash
# 如果用 nohup
tail -f training.log

# 如果用 screen
screen -r training
```

### 查看訓練記錄

```bash
# 查看 CSV 記錄
cat results/training_history.csv

# 查看最新實驗
ls -lt weights/experiments/ | head -5
```

---

## 💾 訓練完成後

### 查看結果

```bash
# 查看最新實驗結果
ls -lh weights/experiments/tennis_7class_*/

# 查看訓練記錄
tail -1 results/training_history.csv
```

### 下載模型到本地（可選）

```bash
# 在你的 Mac 上
scp -r user@remote-server:~/projects/期末專案/weights/experiments/tennis_7class_20251125_* \
    ./weights/experiments/

# 下載訓練記錄
scp user@remote-server:~/projects/期末專案/results/training_history.csv \
    ./results/
```

### 提交結果

```bash
# 在遠端主機上
git add results/training_history.csv
git commit -m "Training: tennis 7class baseline, F1=0.XX"
git push
```

---

## ⚡ 常見加速技巧

### 1. 使用 GPU 1（已預設）

所有訓練命令已包含 `CUDA_VISIBLE_DEVICES=1`，會自動使用 GPU 1。

```bash
# 確認 GPU 1 可用
nvidia-smi

# 查看即時 GPU 使用狀況
watch -n 1 nvidia-smi

# 如果想改用 GPU 0
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --config ...

# 如果想同時使用兩張 GPU
CUDA_VISIBLE_DEVICES=0,1 python3 src/train.py --config ...
```

### 2. 調整 workers

```yaml
# configs/experiments/tennis_baseline.yaml
training:
  num_workers: 8  # 根據 CPU 核心數調整
```

### 3. 增加 batch size（如果記憶體夠）

```yaml
training:
  batch_size:16  # GPU 記憶體夠的話
```

---

## 🐛 疑難排解

### Q: SSL 證書錯誤（下載預訓練權重時）

```yaml
# 方案 1：關閉預訓練
model:
  use_pretrained: false

# 方案 2：手動下載權重（參考前面說明）
```

### Q: 記憶體不足 (OOM)

```yaml
# 減少 batch_size 和 seq_length
training:
  batch_size: 4  # 降低
model:
  seq_length: 8   # 降低
```

### Q: 訓練很慢

```bash
# 確認是否使用 GPU
python3 -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"

# 減少 seq_length
# 增加 num_workers
```

---

## 📝 完整命令摘要（使用 GPU 1）

```bash
# 1. Clone
git clone <repo-url> && cd 期末專案

# 2. 環境
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 3. 下載資料
./scripts/download_thetis.sh

# 4. 整理資料（手動或腳本）
# 將影片整理到 data/raw/tennis/各類別/

# 5. 預處理
python3 src/data/preprocess_videos.py --raw_dir data/raw/tennis --output_dir data/processed/tennis

# 6. 訓練（使用 GPU 1，背景執行）
CUDA_VISIBLE_DEVICES=1 nohup python3 src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    > training.log 2>&1 &

# 7. 監控
tail -f training.log

# 8. 完成後下載結果（本地）
scp -r user@server:~/projects/期末專案/weights/experiments/tennis_* ./weights/experiments/
```

---

## 🎯 預期時間

| 步驟 | 時間估算 |
|------|---------|
| 環境設置 | 5-10 分鐘 |
| 下載 THETIS | 10-30 分鐘（視網速）|
| 整理資料 | 30-60 分鐘（手動）|
| 預處理 | 10-30 分鐘（視影片數量）|
| 訓練 | 2-6 小時（GPU）/ 10-24 小時（CPU）|

**總計**：約 4-8 小時可完成第一次訓練

---

祝訓練順利！🚀
