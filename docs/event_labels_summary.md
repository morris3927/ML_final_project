# 📊 事件標籤系統說明

## 概述

本專案使用**事件分類系統**（基於 Wang et al. 2025）：
- **4 個通用事件類別**：Smash, Net Play, Rally, Serve
- **目的**：提升跨運動遷移學習效果
- **優勢**：語意一致性、戰術意義

---

## ✅ 已完成的設定

### 1. 事件映射檔案
- **位置**: `configs/event_mapping.yaml`
- **內容**: 定義動作到事件的映射關係
- **支援**: 網球和羽球兩種運動

### 2. 資料整理工具
- **位置**: `scripts/organize_thetis.py`
- **功能**: 幫助整理 THETIS 資料集

### 3. 處理指南
- **位置**: `docs/THETIS_processing.md`
- **內容**: 完整的資料處理流程說明

---

## 🎯 使用方式

### THETIS 資料下載後的處理流程

#### 第 1 步：下載 THETIS
```bash
./scripts/download_thetis.sh
```

#### 第 2 步：檢查資料結構
```bash
python3 scripts/organize_thetis.py
```
這會顯示：
- THETIS 影片檔案清單
- 需要建立的類別資料夾
- 手動整理步驟說明

#### 第 3 步：整理影片
根據 THETIS 的標註檔案，將影片分類到：
- `data/raw/tennis/flat_service/`
- `data/raw/tennis/slice_service/`
- `data/raw/tennis/smash/`
- `data/raw/tennis/forehand_flat/`
- `data/raw/tennis/backhand/`
- `data/raw/tennis/forehand_volley/`
- `data/raw/tennis/backhand_volley/`

#### 第 4 步：預處理
```bash
python3 src/data/preprocess_videos.py \
    --raw_dir data/raw/tennis \
    --output_dir data/processed/tennis
```

---

## 🔄 遷移學習策略

### 當前架構（事件分類）
- **網球**: 4 類事件訓練 (Smash, Net Play, Rally, Serve)
- **優點**: 
  - 語意一致性高
  - 簡化模型複雜度
  - 更適合跨運動遷移
- **排除**: 3 個動作 (backhand2hands, kick_service, slice_service)

### 未來遷移（網球 → 羽球）
1. 用 4 類事件訓練網球模型
2. 使用網球模型作為預訓練權重
3. 在羽球上微調 4 類事件模型
4. 利用事件級別的語意相似性提升效果

---

## 📁 檔案結構

```
configs/
├── event_mapping.yaml          # ✅ 事件映射定義 (YAML)
└── experiments/
    ├── tennis_baseline.yaml    # ✅ 主配置（4類事件）
    └── tennis_colab.yaml       # ✅ Colab 配置

scripts/
├── download_thetis.sh          # ✅ THETIS 下載
└── organize_thetis.py          # ✅ 資料整理工具

docs/
├── THETIS_processing.md        # ✅ 處理指南
├── event_classification.md     # ✅ 事件分類說明
└── dataset_preparation.md      # 資料準備總覽
```

---

## 💡 下一步

1. **如果還沒下載 THETIS**:
   ```bash
   ./scripts/download_thetis.sh
   python3 scripts/organize_thetis.py
   ```

2. **如果已有影片資料**:
   - 按照類別放入 `data/raw/tennis/` 的對應資料夾
   - 運行預處理
   - 開始訓練

詳細步驟請參考：
- 📘 `docs/THETIS_processing.md` - THETIS 專用處理指南
- 🚀 `quickstart.md` - 完整訓練流程
