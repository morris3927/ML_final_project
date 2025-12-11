# 跨運動事件辨識：從網球到羽球 (Cross-Sport Event Recognition)

**深度學習期末專案 | Deep Learning Final Project**

本專案探討 CNN-LSTM 架構在跨運動事件辨識的適用性，透過遷移學習將網球動作識別知識遷移至羽球領域。

**專案狀態：** ✅ 已完成 | **最終報告：** [中文版](final_report_zh.md) | [English](final_report.md)

## 📊 專案成果

### 主要發現

- ✅ **網球基準模型**：達到 **79% 驗證準確率** (F1: 0.71)
- ⚠️ **羽球遷移學習**：5個模型達到 **35-39% 準確率**（低於預期）
- 📊 **類別不平衡挑戰**：少數類別（殺球、發球）召回率接近零
- 🔍 **類別權重效果有限**：改善幅度僅 ±3%
- 💡 **關鍵洞見**：領域差距大於預期，需要更多目標領域資料和進階適應技術

### 可用模型權重

| 模型 | 準確率 | F1 分數 | 策略 | 權重路徑 |
|------|--------|---------|------|----------|
| **Tennis Baseline** | 79.02% | 0.7065 | 凍結骨幹 | `weights/experiments/tennis_4event_baseline_20251126_072103/` |
| **Badminton Frozen v3** | 39.02% | 0.2311 | 凍結骨幹 + 類別權重 | `weights/experiments/badminton_4class_frozen_v3/` |
| **Badminton Finetune v2** | 35.55% | 0.2390 | 完整微調 + 類別權重 | `weights/experiments/badminton_4class_finetune_v2/` |

詳細評估結果請參閱：[羽球模型評估報告](docs/badminton_models_evaluation_report.md)

## 📂 專案架構

```
mis583-group16/
├── data/                          # 資料存放區
│   ├── processed/                 # 預處理後的資料（16幀序列）
│   │   ├── tennis/                # 網球資料（THETIS）
│   │   └── badminton/             # 羽球資料（ShuttleSet）
├── src/                           # 核心程式碼
│   ├── models/
│   │   ├── cnn_lstm.py            # CNN-LSTM 主架構
│   │   └── backbones.py           # ResNet-50 骨幹網路
│   ├── data/
│   │   ├── dataset.py             # PyTorch Dataset
│   │   └── preprocess_videos.py   # 影片預處理
│   ├── utils/
│   │   ├── metrics.py             # 評估指標
│   │   └── visualization.py       # 結果視覺化
│   ├── train.py                   # 訓練腳本
│   └── evaluate.py                # 評估腳本
├── configs/experiments/           # 實驗配置檔
│   ├── tennis_baseline.yaml       # 網球基準訓練
│   ├── badminton_transfer_frozen.yaml         # 羽球凍結骨幹
│   ├── badminton_transfer_frozen_v2.yaml      # 羽球凍結骨幹 v2（類別權重）
│   └── badminton_transfer_finetune.yaml       # 羽球完整微調
├── weights/experiments/           # 訓練好的模型權重
│   ├── tennis_4event_baseline_20251126_072103/
│   ├── badminton_4class_frozen_v3/
│   └── badminton_4class_finetune_v2/
├── results/                       # 實驗結果
│   ├── training_curves.png
│   └── badminton_comparison/      # 羽球模型對比
├── docs/                          # 專案文檔
│   ├── badminton_models_evaluation_report.md  # 羽球模型評估報告
│   ├── progress_update_report.md              # 期中進度報告
│   └── *.md                       # 其他技術文檔
├── final_report_zh.md             # 期末報告（中文）
├── final_report.md                # 期末報告（英文）
└── README.md                      # 本文件
```

## 🚀 快速開始

### 1. 環境設定

```bash
# 建立並啟動虛擬環境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 安裝依賴套件
pip install -r requirements.txt
```

**必要套件：**
- PyTorch >= 1.13
- torchvision
- opencv-python
- numpy, pandas
- scikit-learn
- matplotlib
- tqdm, pyyaml

### 2. 資料集準備

#### 網球資料（THETIS）

使用 Git sparse checkout 下載：

```bash
# 建立暫存目錄
mkdir -p ~/Downloads/thetis_rgb && cd ~/Downloads/thetis_rgb

# Sparse checkout 只下載影片
git init
git config core.sparseCheckout true
echo "VIDEO_RGB" >> .git/info/sparse-checkout
git remote add origin https://github.com/THETIS-dataset/dataset.git
git pull origin main
```

#### 羽球資料（ShuttleSet）

從 [ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet) 下載並整理到 `data/raw/badminton/`。

#### 預處理

```bash
# 網球資料預處理
python src/data/preprocess_videos.py \
    --raw_dir data/raw/tennis \
    --output_dir data/processed/tennis \
    --split_ratio 0.7 0.15 0.15 \
    --sport tennis

# 羽球資料預處理
python src/data/preprocess_videos.py \
    --raw_dir data/raw/badminton \
    --output_dir data/processed/badminton \
    --split_ratio 0.7 0.15 0.15 \
    --sport badminton
```

### 3. 訓練模型

#### A. 訓練網球基準模型

```bash
python src/train.py --config configs/experiments/tennis_baseline.yaml
```

**預期結果：** 約 79% 驗證準確率，訓練時間約 2.5 小時（RTX 2080 Ti）

#### B. 羽球遷移學習（凍結骨幹）

```bash
python src/train.py \
    --config configs/experiments/badminton_transfer_frozen_v2.yaml
```

**配置要點：**
- 凍結 ResNet-50 骨幹（`freeze_backbone: true`）
- 使用類別權重（`use_class_weights: true`）
- 載入網球預訓練權重

#### C. 羽球遷移學習（完整微調）

```bash
python src/train.py \
    --config configs/experiments/badminton_transfer_finetune.yaml
```

**配置要點：**
- 解凍整個網路（`freeze_backbone: false`）
- 較低學習率（`learning_rate: 0.00005`）
- 使用類別權重

### 4. 評估模型

#### 評估特定模型

```bash
python src/evaluate.py \
    --model_path weights/experiments/tennis_4event_baseline_20251126_072103/best_model.pth \
    --test_data data/processed/tennis/test \
    --config weights/experiments/tennis_4event_baseline_20251126_072103/config.yaml \
    --output_dir results/tennis_evaluation
```

#### 批量評估羽球模型

```bash
python test_all_badminton_models.py
```

這將評估所有羽球模型並生成對比報告。

## 📊 事件定義

為實現跨運動語義對齊，我們定義 4 個通用事件類別：

| 事件 ID | 事件名稱 | 網球動作 | 羽球動作 | 說明 |
|---------|---------|----------|----------|------|
| 0 | **Smash（殺球）** | smash | smash, wrist_smash | 高強度進攻擊球 |
| 1 | **Net Play（網前）** | forehand_volley, backhand_volley | net_shot, return_net, rush, push | 網前控制型擊球 |
| 2 | **Rally（對打）** | forehand_flat, backhand, forehand_slice | clear, lob, drive, drop | 底線/中場維持比賽 |
| 3 | **Serve（發球）** | flat_service | short_service, long_service | 回合開始動作 |

詳細映射規則請參閱：[事件標籤總結](docs/event_labels_summary.md)

## 🛠 技術細節

### 模型架構

```
輸入: 16幀 RGB 序列 (224×224×3)
  ↓
ResNet-50 (凍結，ImageNet預訓練)
  ↓ 提取空間特徵 (2048-dim/幀)
  ↓
3層雙向LSTM (hidden_size=512)
  ↓ 時序建模
  ↓
分類頭: FC → ReLU → Dropout → FC
  ↓
輸出: 4類事件機率
```

### 關鍵設定

- **輸入模態：** RGB-only（無光流）
- **序列長度：** 16 幀（約 0.5-1 秒）
- **骨幹網路：** ResNet-50（ImageNet 預訓練）
- **時序建模：** 3 層雙向 LSTM
- **無注意力機制：** 使用最後時間步輸出
- **框架：** PyTorch 1.13+

### 訓練策略

| 策略 | 網球基準 | 羽球凍結 | 羽球微調 |
|------|---------|---------|---------|
| **Freeze Backbone** | ✓ | ✓ | ✗ |
| **Learning Rate** | 1e-4 | 1e-4 | 5e-5 |
| **Batch Size** | 32 | 16 | 16 |
| **Class Weights** | ✗ | v2: ✓ | v2: ✓ |
| **Epochs** | 50 | 50 | 50 |

## 📈 實驗結果總結

### 網球基準模型

- **準確率：** 79.02%
- **F1 分數：** 0.7065
- **訓練時間：** 約 2.5 小時（50 epochs）
- **觀察：** 所有事件類別表現均衡，輕度過擬合

### 羽球遷移學習

| 模型 | 策略 | 準確率 | F1 分數 | 主要問題 |
|------|------|--------|---------|----------|
| frozen_v3 | 凍結+權重 | 39.02% | 0.2311 | Smash 完全無法預測 |
| finetune_v2 | 微調+權重 | 35.55% | 0.2390 | 整體準確率較低 |

**關鍵挑戰：**
1. **嚴重領域差距**（場地、物體、視覺外觀）
2. **目標領域資料不足**
3. **極端類別不平衡**（少數類別樣本極少）
4. **類別權重效果有限**（僅改善 ±3%）

詳細分析請參閱：[期末報告](final_report_zh.md)

## 📚 文檔

- **[事件分類說明](docs/event_classification.md)**：事件映射框架設計原理
- **[實驗管理指南](docs/experiment_management.md)**：如何管理和追蹤實驗
- **[遠端訓練指南](docs/remote_training_guide.md)**：在遠端伺服器上訓練模型

## 🔬 重現實驗結果

### 網球基準

```bash
# 1. 預處理資料（如果尚未完成）
python src/data/preprocess_videos.py \
    --raw_dir data/raw/tennis \
    --output_dir data/processed/tennis \
    --sport tennis

# 2. 訓練
python src/train.py --config configs/experiments/tennis_baseline.yaml

# 3. 評估
python src/evaluate.py \
    --model_path weights/experiments/tennis_4event_baseline_20251126_072103/best_model.pth \
    --test_data data/processed/tennis/test \
    --config weights/experiments/tennis_4event_baseline_20251126_072103/config.yaml
```

### 羽球遷移學習

```bash
# 使用預訓練的網球模型
python src/train.py --config configs/experiments/badminton_transfer_finetune.yaml
```

## 🤝 貢獻者

- **謝睿恩** (M144020038)
- **楊翊愷** (M144020057)

## 📄 授權與引用

本專案使用 MIT 授權。使用的資料集：

- **THETIS Dataset**: https://github.com/THETIS-dataset/dataset
- **ShuttleSet Dataset**: https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet


## 🔮 未來工作

基於實驗結果，我們建議：

**短期改進：**
- 資料增強與過採樣（針對少數類別）
- 光流整合（添加動作資訊）
- 時序注意力機制

**中期改進：**
- 領域適應技術（DANN、MMD）
- 多階段訓練策略
- Transformer-based 架構

**長期方向：**
- 多運動聯合預訓練
- 少樣本學習方法
- 弱監督學習

詳見：[期末報告 - 未來工作章節](final_report_zh.md#54-未來工作)

---

**專案倉庫：** https://github.com/morris3927/mis583-group16

**最後更新：** 2025年12月10日
