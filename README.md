# 跨運動事件辨識：從網球到羽球 (Cross-Sport Event Recognition)

本專案旨在驗證 CNN-LSTM 架構在高速球拍運動（如羽球）中的適用性，並探討利用網球數據進行遷移學習（Transfer Learning）以提升羽球事件辨識的效果。

## 📂 專案架構 (Project Structure)

建議的專案目錄結構如下：

```
project_root/
├── data/                       # 資料存放區
│   ├── raw/                    # 原始影片檔
│   │   ├── tennis/             # 網球影片
│   │   └── badminton/          # 羽球影片
│   ├── processed/              # 預處理後的資料 (如 Frame 序列, 光流特徵)
│   │   ├── tennis/
│   │   └── badminton/
│   └── annotations/            # 標註檔 (JSON/CSV)
│       ├── tennis_labels.json
│       └── badminton_labels.json
├── src/                        # 核心程式碼
│   ├── models/                 # 模型定義
│   │   ├── cnn_lstm.py         # CNN-LSTM 主架構
│   │   └── backbones.py        # ResNet-50 等骨幹網路
│   ├── data/                   # 資料處理相關
│   │   ├── dataset.py          # PyTorch Dataset 定義
│   │   └── optical_flow.py     # 光流法提取工具
│   ├── utils/                  # 通用工具
│   │   ├── visualization.py    # Grad-CAM 與結果繪圖
│   │   └── metrics.py          # 評估指標計算
│   ├── train.py                # 訓練腳本
│   └── evaluate.py             # 測試與評估腳本
├── configs/                    # 設定檔
│   ├── config.yaml             # 全域參數設定 (路徑, Hyperparameters)
│   └── experiments/            # 不同實驗的設定 (e.g., baseline vs transfer)
├── notebooks/                  # Jupyter Notebooks (EDA, 測試用)
├── weights/                    # 訓練好的模型權重
├── results/                    # 輸出結果 (Log, 圖表, 預測結果)
├── requirements.txt            # Python 套件需求
├── workflow.md                 # 專案執行流程與進度
└── README.md                   # 專案說明文件
```

## 🚀 快速開始 (Quick Start)

### 1. 環境設定

建立並啟動虛擬環境：

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 資料集準備 (Dataset Preparation)

#### 方式 A：使用 THETIS 公開資料集

THETIS 是一個公開的網球資料集，使用 Git sparse checkout 下載：

```bash
# 建立暫存目錄並下載
mkdir -p ~/Downloads/thetis_rgb && cd ~/Downloads/thetis_rgb

# 使用 sparse checkout 只下載影片部分
git init
git config core.sparseCheckout true
echo "VIDEO_RGB" >> .git/info/sparse-checkout
git remote add origin https://github.com/THETIS-dataset/dataset.git
git pull origin main

# 將影片整理到專案的 data/raw/tennis/ 各類別資料夾中
```

#### 方式 B：使用自己的資料

請依照以下結構放置您的資料：

1.  **網球資料 (Source Domain)**：
    *   將原始影片按類別放入 `data/raw/tennis/` 的子資料夾中
    *   目前支援的類別：flat_service, slice_service, smash, forehand_flat, backhand, forehand_volley, backhand_volley
2.  **羽球資料 (Target Domain)**：
    *   將原始影片放入 `data/raw/badminton/` 的對應類別資料夾中

**預處理 (Preprocessing)**：
執行 RGB frame 提取（簡化版，不計算光流）：
```bash
python src/data/preprocess_videos.py \
    --raw_dir data/raw/tennis \
    --output_dir data/processed/tennis \
    --split_ratio 0.7 0.15 0.15
```

### 3. 訓練 (Training)

本專案支援三種訓練策略，請透過 `configs/` 中的設定檔或參數進行切換。

#### A. Baseline (僅使用羽球資料從頭訓練)
```bash
python src/train.py --config configs/experiments/baseline_badminton.yaml
```

#### B. Strategy A (凍結特徵層遷移學習)
先預訓練網球模型，或下載預訓練權重，然後凍結 Backbone 訓練羽球分類器：
```bash
# 1. 預訓練網球模型 (若無現成權重)
python src/train.py --config configs/experiments/pretrain_tennis.yaml

# 2. 遷移至羽球 (凍結 Backbone)
python src/train.py --config configs/experiments/transfer_frozen.yaml --pretrained_weights weights/tennis_best.pth
```

#### C. Strategy B (微調全模型)
載入網球權重，以較小 Learning Rate 微調整個網路：
```bash
python src/train.py --config configs/experiments/transfer_finetune.yaml --pretrained_weights weights/tennis_best.pth
```

### 4. 評估 (Evaluation)

評估模型並生成 Confusion Matrix 與 Grad-CAM 熱力圖：

```bash
python src/evaluate.py --model_path weights/badminton_best.pth --test_data data/processed/badminton/test
```

## 📊 事件定義 (Event Definitions)

為確保網球與羽球的語義對齊，我們採用 4 個核心動作類別：

* **Serve (發球):** 比賽開始的動作序列。
* **Smash (殺球/得分):** 造成直接得分的極端強力動作。
* **Rally (對打/過渡):** 比賽進行中，用於過渡和建立機會的一般回擊（長球、切球、挑球）。
* **Defense/Receive (防守/接發):** 處於被動狀態或網前快速反應的動作（擋小球、截擊）。

## 🛠 技術細節

*   **Input**: RGB Frames + Dense Optical Flow (6 channels)
*   **Backbone**: ResNet-50 (ImageNet Pre-trained)
*   **Temporal**: Bi-directional LSTM
*   **Framework**: PyTorch
