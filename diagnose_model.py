#!/usr/bin/env python3
"""
診斷模型預測偏差問題
檢查模型輸出的 logits 分佈，找出為什麼只預測單一類別
"""

import sys
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.models.cnn_lstm import CNNLSTM
from src.data.dataset import VideoEventDataset


def diagnose_model(model_path, config_path, test_data_dir, num_samples=50):
    """
    診斷模型預測問題
    
    Args:
        model_path: 模型權重路徑
        config_path: 配置文件路徑
        test_data_dir: 測試資料目錄
        num_samples: 要檢查的樣本數
    """
    print(f"診斷模型: {model_path}")
    print("="*60)
    
    # 載入配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}\n")
    
    # 載入測試資料
    test_dataset = VideoEventDataset(
        processed_dir=test_data_dir,
        seq_length=config['model'].get('seq_length', 16),
        transform=None,
        use_event_labels=config['data'].get('use_event_labels', False),
        sport=config['data'].get('sport', 'badminton')
    )
    
    num_classes = len(test_dataset.class_to_idx)
    class_names = list(test_dataset.class_to_idx.keys())
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 一次處理一個樣本以便詳細分析
        shuffle=False,
        num_workers=0
    )
    
    print(f"類別: {class_names}")
    print(f"測試集大小: {len(test_dataset)}\n")
    
    # 載入模型
    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model'].get('num_lstm_layers', 2),
        pretrained=False,
        use_optical_flow=config['model'].get('use_optical_flow', False)
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint 資訊:")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'train_loss' in checkpoint:
            print(f"  訓練損失: {checkpoint['train_loss']:.4f}")
        if 'val_loss' in checkpoint:
            print(f"  驗證損失: {checkpoint['val_loss']:.4f}")
        if 'val_accuracy' in checkpoint:
            print(f"  驗證準確率: {checkpoint['val_accuracy']:.4f}")
        print()
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 收集預測資訊
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    
    print(f"分析前 {num_samples} 個樣本的預測...\n")
    
    with torch.no_grad():
        for idx, (frames, label) in enumerate(test_loader):
            if idx >= num_samples:
                break
            
            frames = frames.to(device)
            label = label.to(device)
            
            # 獲取 logits
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(pred.cpu().item())
            all_labels.append(label.cpu().item())
            
            # 顯示前 10 個樣本的詳細資訊
            if idx < 10:
                print(f"樣本 {idx+1}:")
                print(f"  真實標籤: {class_names[label.item()]}")
                print(f"  預測標籤: {class_names[pred.item()]}")
                print(f"  Logits: {logits.cpu().numpy()[0]}")
                print(f"  機率分佈: {probs.cpu().numpy()[0]}")
                print()
    
    # 統計分析
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("="*60)
    print("統計分析")
    print("="*60)
    
    # Logits 統計
    print("\nLogits 統計 (每個類別):")
    for i, class_name in enumerate(class_names):
        logit_mean = all_logits[:, i].mean()
        logit_std = all_logits[:, i].std()
        logit_min = all_logits[:, i].min()
        logit_max = all_logits[:, i].max()
        print(f"  {class_name:12s}: 平均={logit_mean:7.3f}, 標準差={logit_std:6.3f}, "
              f"範圍=[{logit_min:7.3f}, {logit_max:7.3f}]")
    
    # 機率統計
    print("\n平均預測機率 (每個類別):")
    for i, class_name in enumerate(class_names):
        prob_mean = all_probs[:, i].mean()
        print(f"  {class_name:12s}: {prob_mean:.4f}")
    
    # 預測分佈
    print("\n預測類別分佈:")
    for i, class_name in enumerate(class_names):
        count = (all_preds == i).sum()
        percentage = count / len(all_preds) * 100
        print(f"  {class_name:12s}: {count:3d} / {len(all_preds)} ({percentage:5.1f}%)")
    
    # 真實標籤分佈
    print("\n真實標籤分佈 (前 {} 個樣本):".format(num_samples))
    for i, class_name in enumerate(class_names):
        count = (all_labels == i).sum()
        percentage = count / len(all_labels) * 100
        print(f"  {class_name:12s}: {count:3d} / {len(all_labels)} ({percentage:5.1f}%)")
    
    # 檢查最終分類層的權重
    print("\n" + "="*60)
    print("最終分類層權重分析")
    print("="*60)
    
    # 獲取最後一層的權重和偏置
    final_layer = model.fc[-1]  # 最後一個 Linear 層
    weights = final_layer.weight.data.cpu().numpy()
    bias = final_layer.bias.data.cpu().numpy()
    
    print("\n偏置值 (Bias):")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:12s}: {bias[i]:8.4f}")
    
    print("\n權重範數 (Weight Norm) - 每個類別:")
    for i, class_name in enumerate(class_names):
        weight_norm = np.linalg.norm(weights[i])
        print(f"  {class_name:12s}: {weight_norm:8.4f}")
    
    print("\n" + "="*60)
    print("診斷結論")
    print("="*60)
    
    # 分析問題
    pred_counts = [(all_preds == i).sum() for i in range(num_classes)]
    max_pred_class = np.argmax(pred_counts)
    max_pred_ratio = pred_counts[max_pred_class] / len(all_preds)
    
    if max_pred_ratio > 0.9:
        print(f"\n⚠️  模型嚴重偏向預測 '{class_names[max_pred_class]}' ({max_pred_ratio*100:.1f}%)")
        print("\n可能原因:")
        
        # 檢查 logits 差異
        logit_means = all_logits.mean(axis=0)
        logit_diff = logit_means.max() - logit_means.min()
        if logit_diff > 5:
            print(f"  1. Logits 差異過大 ({logit_diff:.2f})，某個類別的 logit 顯著高於其他類別")
        
        # 檢查偏置
        bias_diff = bias.max() - bias.min()
        if bias_diff > 2:
            print(f"  2. 偏置值差異過大 ({bias_diff:.2f})，可能導致預測偏向")
        
        # 檢查訓練資料不平衡
        print("  3. 訓練資料可能嚴重不平衡，導致模型過度擬合到多數類別")
        print("  4. 學習率可能設定不當，導致模型收斂到次優解")
        
        print("\n建議:")
        print("  - 檢查訓練集的類別分佈是否平衡")
        print("  - 考慮使用類別權重 (class weights) 來平衡訓練")
        print("  - 降低學習率或調整優化器設定")
        print("  - 使用資料增強來平衡類別樣本數")
    else:
        print(f"\n✓ 模型預測分佈相對均勻")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='診斷模型預測偏差問題')
    parser.add_argument('--model_path', type=str, required=True, help='模型權重路徑')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--test_data', type=str, required=True, help='測試資料目錄')
    parser.add_argument('--num_samples', type=int, default=50, help='要檢查的樣本數')
    
    args = parser.parse_args()
    
    diagnose_model(args.model_path, args.config, args.test_data, args.num_samples)
