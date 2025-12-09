# 跨運動事件辨識：以 CNN-LSTM 實現網球至羽球的遷移學習

**CSE544 深度學習期末專案 — 進度報告**

**組員與學號：**
M144020038 謝睿恩｜M144020057 楊翊愷

**報告日期：** 2025年11月26日

---

## 一、專案目標與範圍調整

### 1.1 原始提案目標

原提案計畫建立跨多種球類運動（籃球、羽球、桌球、排球、網球、棒球）的事件辨識系統，結合：
- CNN-LSTM 架構進行時序事件分類
- 弱標註學習降低標註成本
- 多任務學習實現跨運動共用表徵
- 可解釋性分析與網頁 Demo

### 1.2 修改後目標（聚焦網球→羽球遷移）

經過初步研究與資源評估，我們調整專案範圍為：

**核心目標：**
1. 在網球數據上訓練基準模型（Baseline）
2. 通過遷移學習將模型應用於羽球
3. 採用事件級別分類（4類通用事件）替代細粒度動作分類
4. 驗證跨運動遷移學習的可行性

**修改原因：**

| 修改項目 | 原提案 | 現行方案 | 原因 |
|---------|--------|---------|------|
| **運動範圍** | 6種運動 | 網球+羽球 | 聚焦深度而非廣度，確保高品質實作 |
| **標註策略** | 弱標註/自監督 | 標準監督學習 | 簡化流程，建立可靠基準 |
| **事件定義** | 運動特定動作 | 4類通用事件 | 提升跨運動語義一致性 |
| **輸入模態** | RGB + 光流 | RGB-only | 降低計算成本，階段性實驗 |
| **評估指標** | mAP@{5,10}秒 | Accuracy/F1 + mAP@1s | 匹配任務特性（片段分類） |

---

## 二、當前進度總結

### 2.1 已完成工作

#### ✅ 資料準備與預處理

**資料來源：**

1. **網球數據集：THETIS**
   - 來源：[THETIS-dataset/dataset](https://github.com/THETIS-dataset/dataset/tree/main/VIDEO_RGB)
   - 描述：包含職業網球比賽的高品質影片，涵蓋多種動作類別
   - 授權：研究使用

2. **羽球數據集：ShuttleSet**
   - 來源：[CoachAI-Projects/ShuttleSet](https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet)
   - 描述：羽球比賽影片與動作標註資料集
   - 授權：研究使用

**專案代碼庫：**
- GitHub: [morris3927/ML_final_project](https://github.com/morris3927/ML_final_project)

**原始標籤與事件映射：**

為了實現跨運動遷移，我們將細粒度動作標籤映射到4類戰術事件：

| Event ID | Event Name | 網球原始標籤 | 羽球原始標籤 |
|----------|-----------|------------|-------------|
| 0 | Smash | `smash` | `smash`, `wrist_smash` |
| 1 | Net Play | `forehand_volley`, `backhand_volley` | `net_shot`, `return_net`, `rush`, `push`, `cross_court_net_shot` |
| 2 | Rally | `forehand_flat`, `forehand_openstands`, `backhand`, `forehand_slice`, `backhand_slice` | `clear`, `lob`, `defensive_return_lob`, `drive`, `driven_flight`, `back_court_drive`, `defensive_return_drive`, `drop`, `passive_drop` |
| 3 | Serve | `flat_service` | `short_service`, `long_service` |

**排除的網球標籤：**
- `backhand2hands`（雙手反拍）：不符合羽球動作特徵
- `kick_service`, `slice_service`（複雜發球）：不利於跨運動遷移

**映射原理：**
1. **Smash（殺球/進攻）**：試圖終結回合的高強度進攻動作
2. **Net Play（網前/小球）**：發生在網前區域的控制型擊球
3. **Rally（後場/對打）**：比賽常態過程，包含底線抽球、防守、過渡
4. **Serve（發球）**：回合開始的靜態啟動動作

此設計基於 Wang et al. (2025) 的建議，將語義相似的動作聚合為通用事件，提升跨運動語義一致性。

**預處理流程：**
1. 影片抽幀（30 FPS）
2. 按原始動作類別整理
3. 應用事件映射策略
4. Train/Val/Test 切分（70%/15%/15%）
5. 滑動窗口生成序列樣本（16幀，8幀步長）

**資料統計：**
```
網球數據集統計：
- 訓練集：7,775 樣本（1,035 影片）
- 驗證集：882 樣本（216 影片）
- 測試集：952 樣本（234 影片）
- 總計：9,609 樣本（1,485 影片）

事件分布（訓練集）：
- Event 0 (Smash)：    839 樣本 (10.8%)
- Event 1 (Net Play)： 1,597 樣本 (20.5%)
- Event 2 (Rally)：    4,344 樣本 (55.9%)
- Event 3 (Serve)：    995 樣本 (12.8%)

原始動作分布（訓練集，115 videos each）：
- smash, flat_service → 各 115 影片
- forehand_volley, backhand_volley → 各 115 影片
- forehand_flat, forehand_openstands, backhand, 
  forehand_slice, backhand_slice → 各 115 影片
  
排除動作（未使用）：
- backhand2hands, kick_service, slice_service
```

**數據特點：**
- ✅ 類別平衡良好（原始動作每類115影片）
- ✅ Rally 事件佔主導（56%），符合比賽實際情況
- ✅ 通過滑動窗口增強（train: stride=8, val/test: stride=16）
- ✅ 排除3個不利遷移的動作類別

> **[待補充]** 請在此處補充實際數據分布圖表

#### ✅ 模型架構實現

**基於 Wang et al. (2025) 的 CNN-LSTM 架構：**

```
輸入: 視頻片段 (16 frames, 224×224, RGB)
  ↓
CNN Backbone: ResNet-50 (ImageNet預訓練)
  ↓ 提取空間特徵 (2048-dim)
  ↓
LSTM: 3層雙向LSTM (hidden_size=512)
  ↓ 時序建模
  ↓
分類層: FC → ReLU → Dropout → Softmax
  ↓
輸出: 4類事件機率分布
```

**關鍵設計決策：**
1. **凍結 CNN Backbone**：僅訓練 LSTM 層，加速收斂
2. **事件映射策略**：將12種網球動作映射到4類通用事件
3. **排除動作**：移除不利於跨運動遷移的動作（雙手反拍、複雜發球）

#### ✅ 訓練流程與優化

**訓練配置：**
```yaml
模型參數:
  - Backbone: ResNet-50 (凍結)
  - LSTM層數: 3
  - 隱藏單元: 512
  - Dropout: 0.3
  - 序列長度: 16幀

訓練參數:
  - Batch Size: 32
  - Learning Rate: 0.0001
  - Optimizer: Adam
  - Epochs: 50
  - 學習率調度: ReduceLROnPlateau
```

**實驗管理系統：**
- 自動時間戳實驗目錄
- CSV 訓練記錄追蹤
- 最佳模型自動保存（基於驗證F1）

#### ✅ 技術實現特色

**1. 事件分類架構**

根據 Wang et al. (2025) 的建議，我們實現了動作到事件的映射系統：

| Event | 網球動作 | 羽球動作（規劃） |
|-------|---------|----------------|
| **Smash** | smash | smash, wrist_smash |
| **Net Play** | forehand_volley, backhand_volley | net_shot, rush, push |
| **Rally** | forehand_flat, backhand, forehand_slice | clear, drive, drop |
| **Serve** | flat_service | short_service, long_service |

**2. 高效訓練流程**

使用單張 NVIDIA RTX 2080 Ti (11GB VRAM) 進行訓練：
```bash
python src/train.py --config configs/experiments/tennis_baseline.yaml
```

訓練優化：
- Batch size: 32（充分利用GPU記憶體）
- 凍結ResNet-50 backbone（加速收斂）
- ReduceLROnPlateau 學習率調度

**3. 自動化評估流程**

訓練結束後自動在測試集評估：
- 載入最佳驗證模型
- 計算測試集指標
- 記錄到訓練歷史

### 2.2 網球基準模型訓練結果

**訓練配置：**
- 實驗ID: `tennis_4event_baseline_20251126_072103`
- 總訓練時間: ~2.5小時 (50 epochs × 3 min/epoch)
- 最佳模型: Epoch 42 (基於驗證F1)

**訓練曲線：**

![Training and Validation Curves](training_curves.png)

**最終性能指標：**

| 指標 | 訓練集 | 驗證集 | 最佳驗證值 |
|------|--------|--------|-----------|
| **Accuracy** | 97.57% | 79.02% | 78.19% (Epoch 42) |
| **F1 Score** | 0.9691 | 0.7097 | **0.7065** (Epoch 42) |
| **Loss** | 0.0695 | 1.0760 | 0.9525 (Epoch 42) |

**訓練過程觀察：**

1. **收斂特性**：
   - 訓練集：快速收斂，第30 epoch後趨於穩定
   - 驗證集：在40-45 epoch達到最佳，之後輕微波動

2. **過擬合分析**：
   - 訓練準確率 97.57% vs 驗證準確率 79.02%
   - 差距約 18.5%，存在輕度過擬合
   - 驗證F1=0.7065 仍屬合理範圍

3. **最佳模型選擇**：
   - Epoch 42: Val F1 = 0.7065 (最佳)
   - 自動保存機制選取該權重

4. **驗證集震盪分析** ⚠️：
   - **觀察**：
     - 驗證損失未顯著下降（1.0775 → 1.0760）
     - 驗證準確率震盪範圍大（56%-79%）
   - **原因分析**：
     - 凍結 backbone 限制了特徵學習能力
     - 小驗證集（882樣本）導致統計不穩定
     - 訓練/驗證資料增強策略差異
   - **影響**：
     - LSTM 只能在固定CNN特徵空間中學習
     - 容易記憶訓練樣本，但泛化能力受限
   - **改進方向**：
     - 微調 backbone（作為消融實驗項目）
     - 調整學習率和訓練策略
     - 增強數據增強策略
   
   > **註**：此現象揭示了凍結backbone的局限性，為後續消融實驗提供了重要方向。作為快速baseline，當前結果仍具參考價值。

**訓練環境：**
- GPU: NVIDIA RTX 2080 Ti (11GB)
- 訓練時間: XX 小時
- 最佳模型 Epoch: XX
- Framework: PyTorch 1.x

### 2.3 實現的核心功能

**資料處理模組：**
- [x] 影片預處理與抽幀
- [x] 滑動窗口樣本生成
- [x] 資料增強（RandomFlip, ColorJitter）
- [x] 事件標籤映射系統

**模型訓練模組：**
- [x] CNN-LSTM 架構實現
- [x] 學習率調度
- [x] Early stopping（基於驗證F1）
- [x] 實驗自動化管理

**評估模組：**
- [x] 準確率、F1、精確率、召回率
- [x] 混淆矩陣生成
- [x] Per-class 性能報告
- [x] 自動測試集評估

---

## 三、與原提案的對比分析

### 3.1 保留的核心元素

| 項目 | 實作狀態 | 說明 |
|------|---------|------|
| CNN-LSTM 架構 | ✅ 完成 | 完全依照 Wang et al. (2025) |
| 跨運動遷移 | 🔄 進行中 | 網球→羽球 |
| 事件分類 | ✅ 完成 | 4類通用事件 |
| 時序建模 | ✅ 完成 | 3層雙向LSTM |

### 3.2 修改的技術選擇

#### 修改 1：簡化標註策略

**原提案：** 弱標註（MIL）+ 自監督學習  
**現行方案：** 標準監督學習（完整標註）

**原因：**
1. **建立可靠基準**：標準監督學習提供更清晰的性能上限
2. **資源限制**：弱標註需要更複雜的訓練流程和調參
3. **階段性策略**：先確保基本方法可行，再探索進階技術

#### 修改 2：聚焦雙運動遷移

**原提案：** 6種運動多任務學習  
**現行方案：** 網球→羽球單一遷移路徑

**原因：**
1. **深度優於廣度**：集中資源確保高品質實作
2. **語義相似性**：網球和羽球的事件語義相近，利於驗證方法
3. **資料可得性**：兩種運動都有公開資料集

#### 修改 3：階段性輸入模態實驗

**原提案：** RGB + 光流雙流網絡  
**現行方案：** 階段性實驗策略

**實驗計劃：**
1. **階段一（已完成）**：RGB-only baseline
   - 建立基準性能
   - 驗證架構可行性
   
2. **階段二（規劃中）**：RGB + 光流雙流網絡
   - 作為**消融實驗項目**
   - 對比 RGB-only 的性能提升
   - 評估計算成本與準確度的權衡

**原因：**
1. **漸進式驗證**：先確保基本方法可行
2. **資源規劃**：階段性分配計算資源
3. **科學對比**：建立清晰的消融實驗基準

#### 修改 4：調整評估指標

**原提案：** mAP@{5,10}秒（時序定位）  
**現行方案：** Accuracy/F1（訓練） + mAP@1s（應用評估）

**原因：**
1. **任務匹配**：片段分類任務更適合 Accuracy/F1
2. **時間尺度**：16幀 ≈ 0.5-1秒，mAP@1s 匹配模型能力
3. **階段性評估**：訓練時用分類指標，應用時可計算 mAP@1s

### 3.3 未實現的延伸功能（未來工作）

- [ ] 可解釋性分析（Grad-CAM、注意力熱圖）
- [ ] 網頁 Demo 系統
- [ ] 球體/關鍵點追蹤輔助
- [ ] 多運動多任務學習

---

## 四、下一階段計劃

### 4.1 短期目標（2週內）

**1. 完成羽球數據準備**
- 下載並整理 ShuttleSet 羽球資料集
- 應用事件標籤映射策略
- 預處理並生成訓練樣本（16幀序列）
- 驗證資料品質與標籤一致性

**2. 遷移學習實驗**
- 使用網球模型作為預訓練權重
- Fine-tune 羽球數據
- 對比從零訓練 vs 遷移學習的效果

**3. mAP@1s 評估實現**
- 實現滑動窗口推論
- 計算時序定位準確度
- 與基線方法比較

### 4.2 中期目標（期末前）

**1. 消融實驗**
- **LSTM 層數影響**（1層 vs 2層 vs 3層）
- **Backbone 策略對比** ⭐（重點）
  - 實驗A：凍結backbone（當前baseline，Val F1=0.706）
  - 實驗B：微調backbone（預期改善驗證震盪）
  - 實驗C：兩階段訓練（先凍結20 epochs，再微調30 epochs）
  - 對比：性能 vs 訓練時間 vs 過擬合程度
- **序列長度影響**（8幀 vs 16幀 vs 32幀）
- **輸入模態對比**：RGB-only vs RGB+光流雙流網絡
  - 實現光流提取（Farneback 或 TVL1）
  - 雙流架構訓練
  - 性能與效率權衡分析
- **數據增強策略對比**
  - 評估不同增強強度對驗證集泛化的影響

**2. 性能優化**
- 超參數調優
- 資料增強策略
- 類別平衡處理

**3. 結果分析**
- 錯誤案例分析
- 跨運動泛化能力評估
- 與相關工作對比

### 4.3 最終交付計劃

**技術報告：**
- 完整實驗結果與分析
- 方法論與架構詳述
- 與原提案對比說明
- 未來工作建議

**程式碼與模型：**
- 完整訓練/評估代碼
- 訓練好的模型權重
- 使用說明文檔
- 實驗配置文件

**（可選）Demo：**
- 簡單的命令行推論工具
- 視覺化預測結果

---

## 五、技術挑戰與解決方案

### 5.1 已解決的挑戰

**1. 動作到事件的映射**
- **挑戰**：如何將細粒度動作聚合為通用事件
- **解決**：參考 Wang et al. (2025)，基於戰術語義設計4類事件
- **效果**：提升跨運動語義一致性

**2. 訓練穩定性優化**
- **挑戰**：在 RTX 2080 Ti (11GB) 上訓練大批量
- **解決**：凍結 backbone、優化 batch size
- **效果**：batch_size=32 穩定訓練，無 OOM

**3. 類別不平衡**
- **挑戰**：某些事件類別樣本較少
- **解決**：排除不利遷移的動作，使用F1作為主要指標

### 5.2 當前面臨的挑戰

**1. 羽球資料品質**
- **挑戰**：公開羽球資料可能標註不一致
- **計劃**：仔細篩選與驗證，必要時人工校正

**2. 跨運動domain gap**
- **挑戰**：網球和羽球的視覺外觀差異
- **計劃**：實驗不同fine-tune策略，考慮domain adaptation

**3. 評估基準缺乏**
- **挑戰**：缺少公開的跨運動事件辨識基準
- **計劃**：與相關論文方法對比，建立內部基準

---

## 六、結論與展望

### 6.1 當前成果總結

我們成功建立了基於 CNN-LSTM 的網球事件辨識系統，實現了從資料準備、模型訓練到自動評估的完整流程。通過採用事件級別分類和聚焦單一遷移路徑，我們簡化了原提案的複雜度，同時保留了核心的跨運動遷移學習目標。

**關鍵貢獻：**
1. ✅ 實現完整的端到端事件辨識流程
2. ✅ 建立4類通用事件的語義框架
3. ✅ 證明網球數據上的基準性能
4. ✅ 建立可擴展的實驗管理系統

### 6.2 與原提案的對齊

雖然在具體技術選擇上有所調整，但我們的工作仍然緊扣原提案的核心目標：

- **跨運動遷移**：網球→羽球遷移學習（進行中）
- **CNN-LSTM 架構**：完全依照 Wang et al. (2025）
- **降低標註成本**：通過事件級別聚合減少類別數
- **實用性導向**：建立可重現、可擴展的系統

### 6.3 未來展望

**短期（本學期）：**
- 完成羽球遷移學習實驗
- 實現 mAP@1s 評估
- 撰寫完整技術報告

**長期（研究方向）：**
- 探索弱標註學習降低標註成本
- 擴展到更多運動（桌球、排球）
- 實現可解釋性分析
- 開發實用的Demo系統

---

## 七、參考文獻

[1] Wang, Y. et al. (2025). Research on Match Event Recognition Method Based on LSTM and CNN Fusion. 2025 5th International Conference on Automation Control, Algorithm and Intelligent Bionics (ACAIB).

[2] Naik, B. T. et al. (2022). A Comprehensive Review of Computer Vision in Sports: Open Issues, Future Trends and Research Directions. Applied Sciences.

[3] Mendes-Neves, T. et al. (2023). A Survey of Advanced Computer Vision Techniques for Sports. IEEE Transactions on Circuits and Systems for Video Technology.

[4] THETIS Dataset. https://github.com/THETIS-dataset/dataset

[5] He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.

---

## 附錄

### A. 訓練配置文件

完整配置見：`configs/experiments/tennis_baseline.yaml`

### B. 實驗管理系統

實驗記錄：`results/training_history.csv`  
模型權重：`weights/experiments/`

### C. 程式碼結構

```
期末專案/
├── configs/              # 配置文件
├── src/
│   ├── data/            # 資料處理
│   ├── models/          # 模型定義
│   ├── train.py         # 訓練腳本
│   └── evaluate.py      # 評估腳本
├── docs/                # 文檔
└── results/             # 訓練記錄
```

---

**報告結束**

> 此報告將持續更新，最終版本將於期末提交。
