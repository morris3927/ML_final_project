import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import csv
import json

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 引入專案模組
from src.models.cnn_lstm import CNNLSTM
from src.data.dataset import BadmintonDataset
from src.utils.metrics import calculate_metrics

def create_experiment_dir(config, experiment_name=None):
    """
    創建實驗目錄，帶時間戳
    
    Returns:
        Path: 實驗目錄路徑
        str: 實驗 ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name is None:
        # 從配置推斷實驗名稱
        dataset = Path(config['data']['processed_path']).name
        num_classes = config['data'].get('num_classes', 'unknown')
        experiment_name = f"{dataset}_{num_classes}class"
    
    experiment_id = f"{experiment_name}_{timestamp}"
    experiment_dir = Path("weights/experiments") / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置到實驗目錄
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    return experiment_dir, experiment_id

def log_training_result(experiment_id, config, metrics_dict, model_path):
    """
    記錄訓練結果到 CSV
    
    Args:
        experiment_id: 實驗 ID
        config: 訓練配置
        metrics_dict: 包含訓練指標的字典
        model_path: 模型儲存路徑
    """
    csv_path = Path("results/training_history.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 準備記錄資料
    row = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config_file': config.get('_config_file', 'unknown'),
        'dataset': Path(config['data']['processed_path']).name,
        'num_classes': config['data'].get('num_classes', 0),
        'epochs': config['training']['epochs'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'use_pretrained': config['model'].get('use_pretrained', True),
        'use_optical_flow': config['model'].get('use_optical_flow', False),
        'best_train_acc': f"{metrics_dict.get('train_acc', 0):.4f}",
        'best_train_f1': f"{metrics_dict.get('train_f1', 0):.4f}",
        'best_val_acc': f"{metrics_dict.get('val_acc', 0):.4f}",
        'best_val_f1': f"{metrics_dict.get('val_f1', 0):.4f}",
        'best_test_acc': f"{metrics_dict.get('test_acc', 0):.4f}",
        'best_test_f1': f"{metrics_dict.get('test_f1', 0):.4f}",
        'model_path': str(model_path),
        'notes': config.get('notes', '')
    }
    
    # 寫入 CSV
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists or csv_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
    
    print(f"✓ 訓練記錄已保存到: {csv_path}")

def train(config, experiment_name=None):
    """
    訓練主函式。
    
    Args:
        config (dict): 設定參數字典。
    """
    # 0. 創建實驗目錄
    experiment_dir, experiment_id = create_experiment_dir(config, experiment_name)
    print(f"="*60)
    print(f"實驗 ID: {experiment_id}")
    print(f"實驗目錄: {experiment_dir}")
    print(f"="*60)
    
    # 1. 設定裝置 (Device Setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 準備資料 (Data Preparation)
    from src.data.dataset import get_dataloaders
    
    dataloaders_dict = get_dataloaders(
        data_root=config['data']['processed_path'],
        batch_size=config['training']['batch_size'],
        seq_length=config['model'].get('seq_length', 16),
        num_workers=config['training'].get('num_workers', 4),
        use_event_labels=config['data'].get('use_event_labels', False),
        sport=config['data'].get('sport', 'tennis')
    )
    
    train_loader = dataloaders_dict.get('train')
    val_loader = dataloaders_dict.get('val')
    num_classes = dataloaders_dict.get('num_classes', config['data'].get('num_classes', 4))
    
    if train_loader is None:
        raise ValueError(f"No training data found in {config['data']['processed_path']}/train")
    
    print(f"Train set: {len(train_loader.dataset)} samples")
    if val_loader:
        print(f"Val set: {len(val_loader.dataset)} samples")
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {dataloaders_dict.get('class_to_idx', {})}")

    # 3. 初始化模型 (Model Initialization)
    use_pretrained = config['model'].get('use_pretrained', True)
    weights_path = None
    
    # 處理 use_pretrained 參數
    if isinstance(use_pretrained, str):
        if use_pretrained.lower() == 'imagenet':
            use_pretrained = True
        elif use_pretrained.endswith('.pth') or os.path.exists(use_pretrained):
            weights_path = use_pretrained
            use_pretrained = False  # 使用本地權重，不從網路下載
            print(f"Using local backbone weights: {weights_path}")
        else:
            # 其他字串情況（如 'true'），嘗試轉為布林值
            use_pretrained = use_pretrained.lower() == 'true'

    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model'].get('num_lstm_layers', 2),
        pretrained=use_pretrained,
        use_optical_flow=config['model'].get('use_optical_flow', False),
        weights_path=weights_path
    ).to(device)
    
    # 如果有指定預訓練權重路徑 (例如 Transfer Learning)，則載入
    if 'pretrained_weights' in config['training'] and config['training']['pretrained_weights']:
        print(f"Loading pretrained weights from {config['training']['pretrained_weights']}")
        # 這裡需要根據儲存格式調整，如果是整個模型或 state_dict
        checkpoint = torch.load(config['training']['pretrained_weights'], map_location=device)
        # 處理可能的 key 不匹配問題 (例如多卡訓練的 'module.' 前綴)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
            
    # 凍結 Backbone (如果設定檔要求)
    if config['training'].get('freeze_backbone', False):
        print("Freezing backbone layers...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    # 4. 定義 Loss 與 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), # 只更新需要梯度的參數
        lr=float(config['training']['learning_rate'])
    )
    
    # 學習率排程器 (Optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # 5. 訓練迴圈 (Training Loop)
    best_val_f1 = 0.0
    best_metrics = {'train_acc': 0, 'train_f1': 0, 'val_acc': 0, 'val_f1': 0}
    save_dir = experiment_dir  # 使用實驗目錄
    
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            frames, labels = batch
            frames = frames.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 收集預測結果以計算指標
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': loss.item()})
            
        train_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        print(f"Epoch {epoch+1} Train: Loss={avg_train_loss:.4f}, Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}")
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                frames, labels = batch
                frames = frames.to(device)
                labels = labels.to(device)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_metrics = calculate_metrics(np.array(val_labels), np.array(val_preds))
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch+1} Val: Loss={avg_val_loss:.4f}, Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")
        
        # 更新 Scheduler
        scheduler.step(val_metrics['f1'])
        
        # 儲存最佳模型
        if val_loader and val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics.update({
                'train_acc': train_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1']
            })
            save_path = save_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'config': config
            }, save_path)
            print(f"New best model saved to {save_path} (F1: {best_val_f1:.4f})")
    
    # 訓練完成
    print("\n" + "="*60)
    print("訓練完成！")
    print(f"最佳驗證 F1: {best_val_f1:.4f}")
    print("="*60)
    
    # 在測試集上評估（如果存在）
    test_loader = dataloaders_dict.get('test')
    if test_loader is not None:
        print("\n" + "="*60)
        print("正在測試集上評估最佳模型...")
        print("="*60)
        
        # 載入最佳模型
        best_model_path = save_dir / "best_model.pth"
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 在測試集上推論
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc="Testing")
            for batch in test_pbar:
                frames, labels = batch
                frames = frames.to(device)
                labels = labels.to(device)
                
                outputs = model(frames)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        # 計算測試指標
        test_metrics = calculate_metrics(np.array(test_labels), np.array(test_preds))
        
        print(f"\n測試結果: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}")
        
        # 更新 best_metrics
        best_metrics.update({
            'test_acc': test_metrics['accuracy'],
            'test_f1': test_metrics['f1']
        })
    else:
        print("\n⚠️  未找到測試集，跳過測試評估")
    
    # 記錄到 CSV
    log_training_result(
        experiment_id=experiment_id,
        config=config,
        metrics_dict=best_metrics,
        model_path=save_dir / "best_model.pth"
    )
    
    return experiment_dir, best_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--pretrained_weights", type=str, help="Path to pretrained weights (override config)")
    parser.add_argument("--experiment_name", type=str, help="Custom experiment name (optional)")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # 記錄配置文件名稱
    config['_config_file'] = os.path.basename(args.config)
    
    # 允許從命令列覆蓋預訓練權重路徑
    if args.pretrained_weights:
        if 'training' not in config:
            config['training'] = {}
        config['training']['pretrained_weights'] = args.pretrained_weights
    
    # 執行訓練
    experiment_dir, metrics = train(config, experiment_name=args.experiment_name)
    
    print(f"\n✅ 訓練完成！")
    print(f"📁 實驗目錄: {experiment_dir}")
    print(f"📊 訓練記錄: results/training_history.csv")

