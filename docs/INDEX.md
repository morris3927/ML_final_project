# 📚 專案文檔索引

## 🎯 快速導航

根據你的需求，選擇對應的文檔：

### 🚀 剛開始使用
→ **[quickstart.md](/Users/morris/.gemini/antigravity/brain/7caf6d5a-983b-4dcd-b758-e8b5bdd983cb/quickstart.md)** - 完整的訓練流程指南
- 從資料準備到模型訓練的所有步驟
- 雲端部署說明
- 疑難排解

### ☁️ 遠端主機訓練
→ **[docs/remote_training_guide.md](docs/remote_training_guide.md)** - 實驗室遠端訓練完整流程 ⭐
- 從 Git clone 到訓練完成的 6 個步驟
- THETIS 資料下載與整理
- 背景執行與監控
- 常見問題排解

### 📓 Google Colab 訓練
→ **[colab_training.ipynb](colab_training.ipynb)** - Colab 專用 Notebook
- 一鍵掛載 Google Drive
- 自動安裝環境與訓練
- 適合沒有 GPU 的開發者

### 📦 準備資料
→ **[docs/dataset_preparation.md](docs/dataset_preparation.md)** - 資料集下載與整理
- THETIS 資料集下載方法
- 影片整理步驟
- 其他資料來源建議

### 🧪 實驗管理
→ **[docs/experiment_management.md](docs/experiment_management.md)** - 自動化實驗追蹤系統
- 時間戳資料夾自動管理
- CSV 訓練記錄
- 不覆蓋舊模型

### 🎯 事件分類架構
→ **[docs/event_classification.md](docs/event_classification.md)** - 事件分類說明（Wang et al. 2025）
- 4 大戰術事件定義
- 動作到事件的映射
- 跨運動遷移學習架構

### 📖 專案說明
→ **[README.md](README.md)** - 專案總覽
- 專案目標與架構
- 技術細節
- 訓練策略說明

### 🎓 提案文件
→ **[target.md](target.md)** - 研究提案
- 研究問題與動機
- 方法設計
- 實驗計畫

### 📝 開發流程
→ **[workflow.md](workflow.md)** - 詳細的開發步驟
- 資料處理流程
- 模型建置步驟
- 評估與視覺化

---

## 🛠 工具腳本

### 測試工具
- **`test_quick.py`** - 快速測試模型和資料集
  ```bash
  python3 test_quick.py
  ```

### 資料下載
- **`scripts/download_thetis.sh`** - 自動下載 THETIS 資料集
  ```bash
  ./scripts/download_thetis.sh
  ```

### 資料預處理
- **`src/data/preprocess_videos.py`** - 影片預處理腳本
  ```bash
  python3 src/data/preprocess_videos.py --help
  ```

### 訓練與評估
- **`src/train.py`** - 模型訓練
  ```bash
  python3 src/train.py --config configs/experiments/tennis_baseline.yaml
  ```
- **`src/evaluate.py`** - 模型評估
  ```bash
  python3 src/evaluate.py --model_path weights/best_model.pth
  ```

---

## 📂 配置文件

### 訓練配置
- **`configs/experiments/tennis_baseline.yaml`** - 網球事件分類（4 類）
- **`configs/experiments/tennis_colab.yaml`** - Colab 事件分類（4 類）
- **`configs/event_mapping.yaml`** - 動作到事件的映射配置
- **`configs/config.yaml`** - 全域配置（基礎範本）

---

## 🎓 論文與參考

### 主要參考文獻
1. **Wang, Y. (2025)**. *Research on Match Event Recognition Method Based on LSTM and CNN Fusion*. 2025 5th International Conference on Automation Control, Algorithm and Intelligent Bionics (ACAIB).

2. **THETIS Dataset**: [GitHub Repository](https://github.com/THETIS-dataset/dataset)

---

## 💡 使用建議

### 第一次使用？
1. 閱讀 **quickstart.md**
2. 運行 `test_quick.py` 驗證環境
3. 使用 `download_thetis.sh` 下載資料
4. 參考 **dataset_preparation.md** 整理資料
5. 開始訓練！

### 已經熟悉流程？
- 直接查閱 **README.md** 了解各訓練策略
- 參考 **workflow.md** 了解詳細步驟
- 調整 `configs/experiments/` 中的配置

### 遇到問題？
1. 查看 **quickstart.md** 的疑難排解章節
2. 檢查 `test_quick.py` 的測試結果
3. 確認配置文件格式是否正確

---

## 🔄 更新日誌

### 2025-11-25
- ✅ 建立 RGB-only 簡化版訓練流程
- ✅ 實作預處理腳本
- ✅ 更新 Dataset 支援 sliding window
- ✅ 添加 THETIS 資料集下載說明
- ✅ 創建快速測試工具

---

## 📞 聯絡資訊

**組員**：
- 楊翊愷 (M144020057)
- 謝睿恩 (M144020038)

**課程**: CSE544 深度學習期末專案
**學期**: 2024-2025
