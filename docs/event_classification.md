# 事件分類架構說明

## 概述

根據 Wang et al. (2025) 的研究，我們將細分動作聚合為 4 個具備戰術意義的**事件類別**，以提升跨運動遷移學習的效果。

## 四大事件定義

### Event 0: Smash (殺球/進攻事件)
**定義**: 試圖終結回合或創造機會的高強度進攻
- **特徵**: 手臂高舉過頭、極高的光流速度
- **網球動作**: `smash`
- **羽球動作**: `smash`, `wrist_smash`

### Event 1: Net Play (網前/小球事件)
**定義**: 發生在網前區域的控制型擊球
- **特徵**: 球員位置靠近網子、動作幅度較短促
- **網球動作**: `backhand_volley`, `forehand_volley`
- **羽球動作**: `net_shot`, `return_net`, `rush`, `push`, `cross_court_net_shot`

### Event 2: Rally (後場/對打事件)
**定義**: 比賽的常態過程，包含底線抽球、防守、過渡與調動
- **特徵**: 球員位於中後場且有完整揮拍軌跡
- **網球動作**: `forehand_flat`, `forehand_openstands`, `backhand`, `forehand_slice`, `backhand_slice`
- **羽球動作**: `clear`, `lob`, `defensive_return_lob`, `drive`, `driven_flight`, `back_court_drive`, `defensive_return_drive`, `drop`, `passive_drop`

### Event 3: Serve (發球事件)
**定義**: 回合開始的靜態啟動動作
- **網球動作**: `flat_service`
- **羽球動作**: `short_service`, `long_service`

## 排除的動作

為了提升跨運動遷移學習效果，以下動作被排除：

### 網球
- `backhand2hands`: 雙手反拍（不符合羽球動作特徵）
- `kick_service`: 上旋發球（複雜發球，不利遷移）
- `slice_service`: 切削發球（複雜發球，不利遷移）

## 使用方式

### 1. 訓練事件分類模型（地端）

```bash
# 啟動虛擬環境
source venv/bin/activate

# 訓練（使用主配置文件）
python src/train.py --config configs/experiments/tennis_baseline.yaml

# 或使用後台訓練
CUDA_VISIBLE_DEVICES=1 nohup python src/train.py \
    --config configs/experiments/tennis_baseline.yaml \
    --experiment_name "tennis_4event_baseline" \
    > training.log 2>&1 &
```

### 2. 訓練事件分類模型（Colab）

```python
# 在 Colab notebook 中
!python src/train.py --config configs/experiments/tennis_colab.yaml
```

### 3. 評估事件分類模型

```bash
python src/evaluate.py \
  --model_path weights/experiments/[實驗ID]/best_model.pth \
  --test_data data/processed/tennis/test \
  --config configs/experiments/tennis_baseline.yaml \
  --output_dir results/event_evaluation
```

## 配置文件

所有主配置文件已更新為事件分類模式（4類）：

| 配置文件 | 用途 | 類別數 | 訓練樣本數 |
|---------|------|-------|-----------|
| `tennis_baseline.yaml` | 地端訓練 | 4 | ~7,775 |
| `tennis_colab.yaml` | Colab 訓練 | 4 | ~7,775 |

## 數據統計

根據測試結果：

```
訓練集: 7,775 samples
驗證集: 882 samples  
測試集: 952 samples
總計: 9,609 samples
```

### 排除的樣本
以下動作的樣本被排除（約 2,000+ samples）:
- `backhand2hands`
- `kick_service`
- `slice_service`

## 事件映射配置

詳細映射定義請參考: [`configs/event_mapping.yaml`](file:///Users/morris/Documents/中山相關/深度學習/期末專案/configs/event_mapping.yaml)

## 遷移學習流程

1. **網球事件分類訓練** (當前階段)
   ```bash
   python src/train.py --config configs/experiments/tennis_baseline.yaml
   ```

2. **羽球事件分類訓練** (未來階段)
   - 使用網球訓練好的模型作為預訓練權重
   - 利用事件級別的語義相似性提升遷移效果
   
   ```bash
   python src/train.py \
     --config configs/experiments/badminton_baseline.yaml \
     --pretrained_weights weights/experiments/tennis_xxx/best_model.pth
   ```

## 優勢

1. **語義一致性**: 事件定義跨運動通用
2. **樣本平衡**: 減少細分類別的長尾分布問題
3. **遷移效果**: 提升跨運動域的特徵共享
4. **戰術意義**: 更符合實際比賽分析需求

## 參考文獻

Wang et al. (2025). Cross-Sport Event Recognition using Transfer Learning.
