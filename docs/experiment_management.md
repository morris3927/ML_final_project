# 實驗管理系統使用指南

## ✅ 已實作功能

### 自動化管理
- ✅ **自動建立時間戳資料夾** - 每次訓練都有獨立目錄
- ✅ **訓練記錄 CSV** - 所有實驗結果記錄在 `results/training_history.csv`
- ✅ **配置文件保存** - 每個實驗自動保存配置
- ✅ **不覆蓋舊模型** - 所有歷史模型都保留

---

## 📁 目錄結構

```
weights/experiments/
├── tennis_4event_20251125_104530/
│   ├── best_model.pth          # 最佳模型
│   └── config.yaml             # 訓練配置
├── tennis_4event_20251125_153000/
│   └── ...
└── badminton_transfer_20251126_090000/
    └── ...

results/
└── training_history.csv         # 📊 所有實驗記錄（版控）
```

---

## 🚀 使用方式

### 基本訓練

```bash
python src/train.py --config configs/experiments/tennis_baseline.yaml
```

**自動產生**：
- 實驗目錄：`weights/experiments/tennis_4event_{timestamp}/`
- CSV 記錄：自動添加一行

### 自訂實驗名稱

```bash
python src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name "tennis_high_lr"
```

**產生**：`weights/experiments/tennis_high_lr_{timestamp}/`

---

## 📊 訓練記錄 CSV

位置：`results/training_history.csv`

**字段**：
- `experiment_id` - 實驗 ID（唯一）
- `timestamp` - 訓練時間
- `config_file` - 使用的配置檔案
- `dataset` - 資料集名稱
- `num_classes` - 類別數
- `epochs`, `batch_size`, `learning_rate` - 訓練參數
- `use_pretrained`, `use_optical_flow` - 模型配置
- `best_train_acc`, `best_train_f1` - 訓練集最佳指標
- `best_val_acc`, `best_val_f1` - 驗證集最佳指標
- `model_path` - 模型路徑
- `notes` - 備註

**範例**：
```csv
experiment_id,timestamp,config_file,dataset,num_classes,best_val_f1,model_path
tennis_4event_20251125_104530,2025-11-25 10:45:30,tennis_baseline.yaml,tennis,4,0.8234,weights/experiments/tennis_4event_20251125_104530/best_model.pth
```

---

## 📈 查看訓練歷史

### 用 Excel/Google Sheets
直接開啟 `results/training_history.csv`

### 用 Python/Pandas
```python
import pandas as pd

df = pd.read_csv('results/training_history.csv')
print(df[['experiment_id', 'best_val_f1', 'model_path']])

# 找出最佳模型
best_exp = df.loc[df['best_val_f1'].idxmax()]
print(f"最佳實驗: {best_exp['experiment_id']}")
print(f"F1 分數: {best_exp['best_val_f1']}")
```

---

## 🔄 遷移學習時載入模型

```yaml
# configs/experiments/badminton_transfer.yaml
training:
  pretrained_weights: "weights/experiments/tennis_7class_20251125_104530/best_model.pth"
```

或命令行指定：
```bash
python src/train.py \
    --config configs/experiments/badminton_transfer.yaml \
    --pretrained_weights weights/experiments/tennis_4event_20251125_104530/best_model.pth
```

---

## 💡 最佳實踐

### 1. 訓練前添加備註
在配置文件中加入：
```yaml
notes: "測試較高學習率 lr=0.01"
```

### 2. 定期備份實驗目錄
```bash
# 重要實驗複製到 final/
cp -r weights/experiments/tennis_4event_20251125_104530 weights/final/tennis_best
```

### 3. Git 管理
```bash
# CSV 記錄跟著版控
git add results/training_history.csv
git commit -m "Add training results: tennis 4event baseline"

# 模型權重不上傳（已在 .gitignore）
```

---

## ✨ 優點

- ✅ **不會覆蓋** - 所有訓練都保留
- ✅ **易於追蹤** - CSV 記錄所有實驗
- ✅ **自動化** - 無需手動重命名
- ✅ **版本控制友好** - CSV 可以 git 追蹤
- ✅ **可重現** - 保存配置文件

---

## 🎯 實際工作流程

```bash
# 1. 訓練
python src/train.py --config configs/experiments/tennis_baseline.yaml

# 結果自動保存到:
# - weights/experiments/tennis_4event_20251125_104530/
# - results/training_history.csv (新增一行)

# 2. 查看結果
cat results/training_history.csv

# 3. 如果是最佳模型
cp -r weights/experiments/tennis_4event_20251125_104530 weights/final/tennis_final

# 4. 提交記錄
git add results/training_history.csv
git commit -m "Training: tennis 4event, F1=0.82"
```

---

## 📞 疑難排解

### Q: 如果訓練中斷怎麼辦？
A: 實驗目錄已創建，但模型可能不完整。可以刪除該實驗目錄並重新訓練。

### Q: CSV 記錄太多了？
A: 定期清理或歸檔舊的記錄，保留重要實驗的記錄。

### Q: 想要指定自己的實驗 ID？
A: 使用 `--experiment_name` 參數自訂名稱。
