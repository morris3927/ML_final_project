import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import yaml

class VideoEventDataset(Dataset):
    """
    影片事件辨識資料集 - RGB Only 版本
    從預處理後的資料載入 frame 序列
    """
    
    def __init__(self, processed_dir, seq_length=16, transform=None, stride=8, 
                 use_event_labels=False, event_mapping_path=None, sport='tennis'):
        """
        Args:
            processed_dir: 處理後的資料目錄，如 data/processed/tennis/train
            seq_length: 序列長度（幀數），預設 16
            transform: 資料增強轉換
            stride: sliding window 的步長，預設 8（50% overlap）
            use_event_labels: 是否使用事件標籤（4類）而非動作標籤（12類）
            event_mapping_path: 事件映射配置檔路徑
            sport: 運動類型 ('tennis' 或 'badminton')
        """
        self.processed_dir = Path(processed_dir)
        self.seq_length = seq_length
        self.stride = stride
        self.use_event_labels = use_event_labels
        self.sport = sport
        
        # 載入事件映射
        self.action_to_event = None
        self.excluded_actions = []
        if use_event_labels:
            self._load_event_mapping(event_mapping_path)
        
        # 設定預設 transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # 掃描資料並建立樣本列表
        self.samples = []
        self.class_to_idx = {}
        self._scan_dataset()
        
        print(f"Dataset: {processed_dir}")
        print(f"  Mode: {'Event' if use_event_labels else 'Action'} Classification")
        print(f"  Classes: {len(self.class_to_idx)} - {list(self.class_to_idx.keys())}")
        print(f"  Total samples: {len(self.samples)}")
    
    def _load_event_mapping(self, mapping_path):
        """載入事件映射配置"""
        if mapping_path is None:
            mapping_path = Path(__file__).parent.parent.parent / "configs" / "event_mapping.yaml"
        
        mapping_path = Path(mapping_path)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Event mapping file not found: {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 根據運動類型選擇映射
        mapping_key = f"{self.sport}_action_to_event"
        if mapping_key not in config:
            raise ValueError(f"Mapping for sport '{self.sport}' not found in config")
        
        self.action_to_event = config[mapping_key]
        self.excluded_actions = config.get(f'excluded_{self.sport}_actions', [])
        
        print(f"Loaded event mapping for {self.sport}")
        print(f"  Excluded actions: {self.excluded_actions}")
    
    def _scan_dataset(self):
        """
        掃描資料集目錄，建立樣本列表
        結構: processed_dir/category/video_name/frame_xxxx.jpg
        """
        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed directory not found: {self.processed_dir}")
        
        # 掃描所有類別資料夾
        category_dirs = sorted([d for d in self.processed_dir.iterdir() if d.is_dir()])
        
        if len(category_dirs) == 0:
            raise ValueError(f"No category directories found in {self.processed_dir}")
        
        # 如果使用事件標籤，建立事件到索引的映射
        if self.use_event_labels:
            # 事件固定為 0-3
            self.class_to_idx = {
                'Smash': 0,
                'Net Play': 1,
                'Rally': 2,
                'Serve': 3
            }
        else:
            # 動作標籤：從資料夾名稱建立映射
            for idx, cat_dir in enumerate(category_dirs):
                action_name = cat_dir.name
                # 跳過排除的動作
                if action_name in self.excluded_actions:
                    continue
                self.class_to_idx[action_name] = idx
        
        # 掃描每個類別下的影片
        for cat_dir in category_dirs:
            action_name = cat_dir.name
            
            # 跳過排除的動作
            if action_name in self.excluded_actions:
                print(f"  Skipping excluded action: {action_name}")
                continue
            
            # 確定標籤
            if self.use_event_labels:
                if action_name not in self.action_to_event:
                    print(f"  Warning: Action '{action_name}' not in mapping, skipping...")
                    continue
                label = self.action_to_event[action_name]
            else:
                label = self.class_to_idx[action_name]
            
            # 掃描該類別下的所有影片資料夾
            video_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
            
            for video_dir in video_dirs:
                # 取得該影片的所有 frames
                frame_files = sorted(video_dir.glob("frame_*.jpg"))
                
                if len(frame_files) < self.seq_length:
                    # 影片太短，跳過
                    continue
                
                # 使用 sliding window 建立多個樣本
                num_frames = len(frame_files)
                for start_idx in range(0, num_frames - self.seq_length + 1, self.stride):
                    end_idx = start_idx + self.seq_length
                    sample = {
                        'frames': frame_files[start_idx:end_idx],
                        'label': label,
                        'action': action_name,
                        'video_name': video_dir.name
                    }
                    self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        回傳一個訓練樣本
        
        Returns:
            frames: Tensor of shape (seq_length, 3, H, W)
            label: int (event label if use_event_labels=True, else action label)
        """
        sample = self.samples[idx]
        frame_paths = sample['frames']
        label = sample['label']
        
        # 載入並轉換所有 frames
        frames = []
        for frame_path in frame_paths:
            # 使用 PIL 載入圖片
            img = Image.open(frame_path).convert('RGB')
            
            # 應用 transform
            if self.transform:
                img = self.transform(img)
            
            frames.append(img)
        
        # Stack 成 (seq_length, C, H, W)
        frames = torch.stack(frames, dim=0)
        
        return frames, label


# 為了向後兼容，保留原有的類別名稱
class BadmintonDataset(VideoEventDataset):
    """向後兼容的別名"""
    pass


def get_dataloaders(data_root, batch_size=8, seq_length=16, num_workers=4, 
                    use_event_labels=False, event_mapping_path=None, sport='tennis'):
    """
    便捷函數：建立 train/val/test dataloaders
    
    Args:
        data_root: 資料根目錄，如 data/processed/tennis
        batch_size: batch 大小
        seq_length: 序列長度
        num_workers: DataLoader 的工作執行緒數
        use_event_labels: 是否使用事件標籤（4類）
        event_mapping_path: 事件映射配置檔路徑
        sport: 運動類型 ('tennis' 或 'badminton')
    
    Returns:
        dict: {'train': train_loader, 'val': val_loader, 'test': test_loader, 
               'num_classes': int, 'class_to_idx': dict}
    """
    data_root = Path(data_root)
    
    # 建立資料增強
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 建立 datasets
    dataloaders = {}
    num_classes = None
    class_to_idx = None
    
    for split in ['train', 'val', 'test']:
        split_dir = data_root / split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping...")
            continue
        
        transform = train_transform if split == 'train' else val_transform
        
        dataset = VideoEventDataset(
            processed_dir=split_dir,
            seq_length=seq_length,
            transform=transform,
            stride=8 if split == 'train' else seq_length,  # test/val 不重疊
            use_event_labels=use_event_labels,
            event_mapping_path=event_mapping_path,
            sport=sport
        )
        
        if num_classes is None:
            num_classes = len(dataset.class_to_idx)
            class_to_idx = dataset.class_to_idx
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    dataloaders['num_classes'] = num_classes
    dataloaders['class_to_idx'] = class_to_idx
    
    return dataloaders
