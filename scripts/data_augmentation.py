import logging
import random
import time
import warnings
import shutil
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import deque, OrderedDict
import threading
import weakref

import numpy as np
import albumentations as A
from albumentations.augmentations.dropout.functional import cutout
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
import cv2
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
# warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
# warnings.filterwarnings("ignore", category=FutureWarning)


class CacheManager:
    def __init__(self, max_size_per_class: int = 200, max_memory_mb: int = 1024, 
                 enable_lru: bool = True, enable_stats: bool = True):

        self.max_size_per_class = max_size_per_class
        self.max_memory_mb = max_memory_mb
        self.enable_lru = enable_lru
        self.enable_stats = enable_stats
        
        self._cache: Dict[str, OrderedDict] = {}
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage_mb': 0,
            'total_requests': 0,
            'cache_size_by_class': {}
        }

        self._lock = threading.RLock()
        self._memory_threshold = max_memory_mb * 1024 * 1024
        
        logger.info(f"CacheManager initialized: max_size_per_class={max_size_per_class}, "
                   f"max_memory_mb={max_memory_mb}, LRU={enable_lru}")
    
    def add(self, image: np.ndarray, label: int, class_name: str, 
            metadata: Optional[Dict[str, Any]] = None) -> None:

        with self._lock:
            if class_name not in self._cache:
                self._cache[class_name] = OrderedDict()
            
            cache_item = {
                'image': image.copy(),
                'label': label,
                'metadata': metadata or {},
                'timestamp': time.time(),
                'access_count': 0
            }
            
            if self._should_evict_memory():
                self._evict_oldest_items()
            
            if (self.enable_lru and  len(self._cache[class_name]) >= self.max_size_per_class):
                self._evict_lru_item(class_name)
            
            cache_key = f"{class_name}_{len(self._cache[class_name])}"
            self._cache[class_name][cache_key] = cache_item
            
            if self.enable_stats:
                self._update_stats()
    
    def get(self, class_name: str) -> Optional[Tuple[np.ndarray, int]]:
        with self._lock:
            if class_name not in self._cache or not self._cache[class_name]:
                if self.enable_stats:
                    self.stats['misses'] += 1
                    self.stats['total_requests'] += 1
                return None
            
            cache_key = random.choice(list(self._cache[class_name].keys()))
            cache_item = self._cache[class_name][cache_key]
            
            if self.enable_lru:
                self._cache[class_name].move_to_end(cache_key)
                cache_item['access_count'] += 1
            
            if self.enable_stats:
                self.stats['hits'] += 1
                self.stats['total_requests'] += 1
            
            return cache_item['image'], cache_item['label']
    
    def get_by_class(self, class_name: str, count: int = 1) -> List[Tuple[np.ndarray, int]]:

        with self._lock:
            if class_name not in self._cache or not self._cache[class_name]:
                return []
            
            available_count = min(count, len(self._cache[class_name]))
            cache_keys = random.sample(list(self._cache[class_name].keys()), available_count)
            
            results = []
            for cache_key in cache_keys:
                cache_item = self._cache[class_name][cache_key]
                
                if self.enable_lru:
                    self._cache[class_name].move_to_end(cache_key)
                    cache_item['access_count'] += 1
                
                results.append((cache_item['image'], cache_item['label']))
            
            if self.enable_stats:
                self.stats['hits'] += len(results)
                self.stats['total_requests'] += len(results)
            
            return results
    
    def clear(self, class_name: Optional[str] = None) -> None:
        with self._lock:
            if class_name:
                if class_name in self._cache:
                    del self._cache[class_name]
            else:
                self._cache.clear()
            
            if self.enable_stats:
                self._update_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            stats = self.stats.copy()
            
            # Calculate hit rate
            if stats['total_requests'] > 0:
                stats['hit_rate'] = stats['hits'] / stats['total_requests']
            else:
                stats['hit_rate'] = 0.0
            
            # Calculate cache size by class
            stats['cache_size_by_class'] = {
                class_name: len(cache) 
                for class_name, cache in self._cache.items()
            }
            
            # Calculate total cache size
            stats['total_cached_items'] = sum(len(cache) for cache in self._cache.values())
            
            return stats
    
    def _should_evict_memory(self) -> bool:
        """Check if cache items need to be evicted due to memory shortage"""
        if not self.enable_stats:
            return False
        
        # Estimate memory usage (simplified calculation)
        total_items = sum(len(cache) for cache in self._cache.values())
        estimated_memory = total_items * 32 * 32 * 3 * 4  # Assume 32x32x3 float32 images
        
        return estimated_memory > self._memory_threshold
    
    def _evict_oldest_items(self, evict_ratio: float = 0.1) -> None:
        """Evict oldest cache items"""
        with self._lock:
            total_items = sum(len(cache) for cache in self._cache.values())
            items_to_evict = max(1, int(total_items * evict_ratio))
            
            evicted_count = 0
            for class_name in list(self._cache.keys()):
                if evicted_count >= items_to_evict:
                    break
                
                cache = self._cache[class_name]
                if not cache:
                    continue
                
                # Remove oldest item
                oldest_key = next(iter(cache))
                del cache[oldest_key]
                evicted_count += 1
                
                if self.enable_stats:
                    self.stats['evictions'] += 1
    
    def _evict_lru_item(self, class_name: str) -> None:
        """Evict least recently used item for specified class"""
        with self._lock:
            cache = self._cache[class_name]
            if cache:
                # Remove oldest item (first one)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
                
                if self.enable_stats:
                    self.stats['evictions'] += 1
    
    def _update_stats(self) -> None:
        """Update statistics"""
        self.stats['cache_size_by_class'] = {
            class_name: len(cache) 
            for class_name, cache in self._cache.items()
        }
        
        # Estimate memory usage
        total_items = sum(len(cache) for cache in self._cache.values())
        self.stats['memory_usage_mb'] = (total_items * 32 * 32 * 3 * 4) / (1024 * 1024)
    
    def __len__(self) -> int:
        """Return total number of items in cache"""
        with self._lock:
            return sum(len(cache) for cache in self._cache.values())
    
    def __repr__(self) -> str:
        """Return string representation of cache"""
        stats = self.get_stats()
        return (f"CacheManager(size={len(self)}, "
                f"classes={len(self._cache)}, "
                f"hit_rate={stats['hit_rate']:.2%})")

class CIFAR100:
    def __init__(self, cifar100_dict: dict, confusion_tuples_list: list):
        self.cifar100_dict = cifar100_dict
        self.cifar100_list = sorted([item for sublist in self.cifar100_dict.values() for item in sublist])
        self.class2superclass = {fine: coarse for coarse, fines in self.cifar100_dict.items() for fine in fines}
        self.confusion_tuples_list = confusion_tuples_list

    def get_cifar100_list(self) -> list[str]:
        return self.cifar100_list
    
    def get_label(self, class_name: str) -> int:
        return self.cifar100_list.index(class_name)
    
    def get_superclass_name(self, class_name: str) -> str:
        return self.class2superclass[class_name]
    
    def is_confusion_pair(self, class1: str, class2: str) -> bool:
        return (class1, class2) in self.confusion_tuples_list or (class2, class1) in self.confusion_tuples_list

class ClassSettings:
    def __init__(self, class_name: str, cifar100: CIFAR100 = None,
                 basic_pipeline = None, cache_manager=None, dataset_sampler=None,
                 p: float = 1.0):

        self.class_name = class_name
        self.cifar100 = cifar100
        self.basic_pipeline = basic_pipeline
        self.p = p
        self.multiplier = 1.0

        if self.cifar100:
            self.label = self.cifar100.get_label(class_name)
            self.class_techniques = self.get_class_techniques()
            self.superclass_name = self.cifar100.get_superclass_name(class_name)
            self.superclass_techniques = self.get_superclass_techniques()
        else:
            raise ValueError(f"CIFAR100 is not provided")

        self.pipeline = self.create_pipeline(cache_manager=cache_manager,
                                            dataset_sampler=dataset_sampler)
    
    def get_superclass_techniques(self) -> list[str]:
        superclass_techniques = []
        if self.superclass_name in ['people']:
            superclass_techniques.extend(['face_enhancement', 'pose_variation', 'clothing_diversity'])
            self.multiplier *= 2.0  # Reduced from 2.5
        elif self.superclass_name in ['trees', 'flowers']:
            superclass_techniques.extend(['texture_enhancement', 'shape_variation', 'environment_diversity'])
            self.multiplier *= 2.0  # Reduced from 2.5
        elif self.superclass_name in ['vehicles_1', 'vehicles_2']:
            # Vehicle classes: preserve structural features
            superclass_techniques.extend(['vehicle_structure_enhancement', 'environment_diversity'])
            self.multiplier *= 1.8  # Reduced from 2.0
        elif self.superclass_name in ['small_mammals']:
            # Small animals: enhance texture and shape
            superclass_techniques.extend(['texture_enhancement', 'shape_variation'])
            self.multiplier *= 1.8  # Reduced from 2.0
        elif self.superclass_name in ['fruit_and_vegetables']:
            # Fruits and vegetables: preserve color and shape
            superclass_techniques.extend(['texture_enhancement', 'shape_variation'])
            self.multiplier *= 1.8  # Reduced from 2.0
        elif self.superclass_name in ['insects']:
            # Insects: enhance detail features
            superclass_techniques.extend(['texture_enhancement', 'shape_variation'])
            self.multiplier *= 1.8  # Reduced from 2.0
        elif self.superclass_name in ['large_carnivores', 'large_omnivores_and_herbivores']:
            # Large animals: preserve overall features
            superclass_techniques.extend(['texture_enhancement'])
            self.multiplier *= 1.5
        else:
            # Other classes: use basic augmentation
            self.multiplier *= 1.4  # Slightly reduced
        return superclass_techniques

    def get_class_techniques(self) -> list[str]:
        class_techniques = []
        
        # 人物类别 - 增强区分性（针对困难类别优化）
        if self.class_name in ['boy', 'girl', 'baby']:
            class_techniques.extend(['face_enhancement', 'pose_variation', 'clothing_diversity', 'age_distinction', 'gender_distinction', 'person_context_enhancement'])
            self.multiplier *= 2.5  # 最高增强强度 - 最困难的类别
        elif self.class_name in ['man', 'woman']:
            class_techniques.extend(['face_enhancement', 'pose_variation', 'clothing_diversity', 'age_distinction', 'gender_distinction', 'adult_context_enhancement'])
            self.multiplier *= 2.3  # 高增强强度
            
        # 树木类别 - 增强纹理和结构区分
        elif self.class_name in ['oak_tree', 'maple_tree', 'willow_tree', 'pine_tree']:
            class_techniques.extend(['texture_enhancement', 'tree_structure_distinction', 'environment_diversity', 'leaf_pattern_enhancement', 'bark_texture_enhancement'])
            self.multiplier *= 2.2  # 显著增加增强强度
            
        # 花朵类别 - 增强花瓣和结构区分
        elif self.class_name in ['rose', 'tulip', 'sunflower', 'orchid', 'poppy']:
            class_techniques.extend(['flower_petal_enhancement', 'flower_structure_enhancement', 'environment_diversity', 'flower_color_enhancement', 'petal_texture_enhancement'])
            self.multiplier *= 2.0
            
        # 车辆类别 - 增强结构区分
        elif self.class_name in ['bus', 'streetcar', 'truck', 'pickup_truck']:
            class_techniques.extend(['vehicle_structure_enhancement', 'vehicle_structure_distinction', 'environment_diversity', 'vehicle_context_enhancement'])
            self.multiplier *= 2.0
            
        # 容器类别 - 增强形状和用途区分（最困难类别）
        elif self.class_name in ['bowl', 'plate']:
            class_techniques.extend(['container_shape_enhancement', 'container_usage_context', 'material_texture_enhancement', 'size_context_enhancement'])
            self.multiplier *= 3.0  # 最高增强强度 - 最困难的混淆对
        elif self.class_name in ['cup', 'bottle', 'can']:
            class_techniques.extend(['container_shape_enhancement', 'container_usage_context', 'material_texture_enhancement', 'size_context_enhancement'])
            self.multiplier *= 2.0  # 中等增强强度
            
        # 海洋动物 - 增强纹理和形状区分
        elif self.class_name in ['seal', 'otter', 'dolphin', 'whale', 'shark']:
            class_techniques.extend(['marine_animal_texture', 'aquatic_context_enhancement', 'animal_shape_distinction', 'water_environment_enhancement'])
            self.multiplier *= 2.2
            
        # 昆虫类别 - 增强细节特征
        elif self.class_name in ['bee', 'beetle', 'butterfly', 'cockroach', 'spider']:
            class_techniques.extend(['insect_detail_enhancement', 'insect_texture_enhancement', 'insect_shape_distinction', 'natural_environment_enhancement'])
            self.multiplier *= 2.0
            
        return class_techniques
    
    def create_pipeline(self, cache_manager=None, dataset_sampler=None) -> A.Compose:
        # Create base pipeline
        transforms_list = []
        
        # Add basic pipeline
        if self.basic_pipeline:
            # Get the transforms from the basic pipeline
            basic_transforms = self.basic_pipeline.get_pipeline().transforms
            transforms_list.extend(basic_transforms)
        
        # Add superclass techniques
        if self.superclass_techniques:
            transforms_list.append(ApplyTechniques(requirements=self.superclass_techniques,
                                                    class_name=self.class_name,
                                                    dataset_sampler=dataset_sampler,
                                                    p=0.7))
        # Add class-specific techniques
        if self.class_techniques:
            transforms_list.append(ApplyTechniques(requirements=self.class_techniques,
                                                    class_name=self.class_name,
                                                    dataset_sampler=dataset_sampler,
                                                    p=0.7))
        
        # Add smart rotation augmentation based on class semantics
        smart_rotation = SmartRotation(
            class_name=self.class_name,
            p=0.6
        )
        transforms_list.append(smart_rotation)

        # Add advanced augmentation techniques (optimized strength)
        if cache_manager is not None:
            # Cross-sample augmentation (intra-class) - lighter
            cross_sample = CrossSampleAugmentation(
                class_name=self.class_name,
                alpha=0.2,  # Reduced from 0.3 for less mixing
                p=0.3,  # Reduced from 0.3
                cache_manager=cache_manager,
                dataset_sampler=dataset_sampler
            )
            transforms_list.append(cross_sample)
            
            # Cross-class augmentation (inter-class) - lighter
            cross_class = CrossClassAugmentation(
                class_name=self.class_name,
                alpha=0.1,  # Reduced from 0.15
                p=0.15,  # Reduced from 0.2
                cifar100=self.cifar100,
                dataset_sampler=dataset_sampler,
                cache_manager=cache_manager
            )
            transforms_list.append(cross_class)
        
        else:
            cross_sample = CrossSampleAugmentation(
                class_name=self.class_name,
                alpha=0.2,  # Reduced from 0.3
                p=0.3,  # Reduced from 0.5
                cache_manager=None,
                dataset_sampler=dataset_sampler
            )
            transforms_list.append(cross_sample)
            
            cross_class = CrossClassAugmentation(
                class_name=self.class_name,
                alpha=0.1,  # Reduced from 0.15
                p=0.15,  # Reduced from 0.5
                cifar100=self.cifar100,
                dataset_sampler=dataset_sampler,
                cache_manager=None
            )
            transforms_list.append(cross_class)
        
        return A.Compose(transforms_list)

    def get_pipeline(self) -> A.Compose:
        return self.pipeline

    def get_augmentation_multiplier(self) -> float:
        return self.multiplier

class ImageInfo:
    def __init__(self, image_path: str, cifar100: CIFAR100 = None, basic_pipeline = None):
        self.image_path = image_path
        self.image_np = self._get_image_np()
        self.image_name = self._get_image_name()
        self.class_name = self._get_class_name()
        self.super_class_name = self._get_superclass_name()
        self.class_settings = ClassSettings(class_name=self.class_name, cifar100=cifar100, basic_pipeline=basic_pipeline)
        self.pipeline = self._get_pipeline()
        self.augmentation_multiplier = self._get_augmentation_multiplier()
        self.image_label = self._get_image_label()

    def _get_image_np(self) -> np.ndarray:
        """Get image as numpy array, handling virtual paths."""
        try:
            return np.array(Image.open(self.image_path).convert("RGB"))
        except (FileNotFoundError, OSError):
            # For virtual paths used in ClassSettings, return a dummy image
            return np.zeros((32, 32, 3), dtype=np.uint8)
    
    def _get_image_name(self) -> str:
        return Path(self.image_path).name

    def _get_class_name(self) -> str:
        return Path(self.image_path).parts[-2]
    
    def _get_superclass_name(self) -> str:
        if self.cifar100:
            return self.cifar100.get_superclass_name(self.class_name)
        return "unknown"

    def _get_pipeline(self) -> A.Compose:
        return self.class_settings.get_pipeline()

    def _get_augmentation_multiplier(self) -> float:
        return self.class_settings.get_augmentation_multiplier()

    def _get_image_label(self) -> int:
        if self.cifar100:
            return self.cifar100.get_label(self.class_name)
        return 0

class BasicPipeline(ImageOnlyTransform):
    """Create stable and comprehensive augmentation pipeline for CIFAR-100."""
    def __init__(self, p: float = 1.0):
        super().__init__(p=p)
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self) -> A.Compose:
        """Create the basic augmentation pipeline (optimized for 32x32)."""
        return A.Compose([
            # 1. CIFAR100-optimized Geometric augmentations (conservative for 32x32)
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # Reduced
                scale=(0.9, 1.1),  # Reduced
                rotate=(-15, 15),  # Reduced
                shear={"x": (-3, 3), "y": (-3, 3)},  # Reduced
                border_mode=0,
                p=0.7),
                
            # 2. Advanced Geometric augmentations (significantly reduced)
            A.OneOf([
                A.Perspective(scale=(0.02, 0.05), p=0.3),
                A.Transpose(p=0.3),
                A.SquareSymmetry(p=0.1),
                ], p=0.3),
        
            # 3. Light Crop/Pad augmentations (optimized for small images)
            A.OneOf([
                A.CropAndPad(px=8, fill=0, p=0.3),
                A.RandomSizedCrop(min_max_height=(26, 32), size=(32, 32), p=0.3)
                ], p=0.2),
            
            # 4. CIFAR100-optimized Color augmentations (preserve important color patterns)
            A.SomeOf([
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.7),  # Lighter
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.4),  # Lighter
                A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.4),  # Lighter
                A.ChannelShuffle(p=0.2),  # Reduced
                A.ToGray(p=0.05),  # Reduced
                A.ToSepia(p=0.05),  # Reduced
                ], n=2, p=0.4),
        
            # 5. Light Brightness and contrast
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.CLAHE(clip_limit=1.5, tile_grid_size=(4, 4), p=0.5),
                A.Equalize(p=0.3),
                A.Posterize(num_bits=6, p=0.2),
                ], p=0.2),
        
            # 6. Noise and blur effects (reduced intensity)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                ], p=0.2),
        
            # 7. Minimal Dropout (preserve object integrity)
            A.OneOf([
                A.Erasing(
                    scale=(0.02, 0.2),  # Reduced max scale
                    ratio=(0.33, 3.3),
                    p=0.5
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 4),  # Reduced holes
                    hole_height_range=(8, 12),  # Reduced size
                    hole_width_range=(8, 12),
                    fill=0,
                    p=0.3
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 4),  # Reduced holes
                    hole_height_range=(8, 12),  # Reduced size
                    hole_width_range=(8, 12),
                    fill='random',
                    p=0.2
                ),
                A.GridDropout(ratio=0.2, fill=0, p=0.15),  # Reduced
                ], p=0.3),  # Reduced from 0.4
        
            # 8. Advanced geometric transformations (minimal)
            A.OneOf([
                A.ElasticTransform(alpha=3, sigma=4, p=0.1),
                A.GridDistortion(num_steps=2, distort_limit=0.03, p=0.2),
                A.OpticalDistortion(distort_limit=(-0.05, 0.05), p=0.5),
                ], p=0.05),
        
            # 9. CIFAR100-specific enhancements (preserve fine details)
            A.OneOf([
                A.Sharpen(alpha=(0.05, 0.15), lightness=(0.8, 1.0), p=0.4),  # Lighter sharpening
                A.UnsharpMask(blur_limit=(3, 3), sigma_limit=(0.3, 0.7), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),  
                A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.0), p=0.2),  # Lighter emboss
                ], p=0.15),  # Reduced probability
            
            # 10. CIFAR100-specific: add light noise for robustness
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.05), p=0.1),  # Very light noise
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.2), p=0.1),
                ], p=0.1),
            ], p=0.2)  # Reduced overall probability
    
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the standard pipeline to the image."""
        result = self.pipeline(image=image)
        return result["image"]
    
    def get_pipeline(self) -> A.Compose:
        return self.pipeline


class DatasetSampler:
    """Dataset sampler for getting random images when cache is not available."""
    
    def __init__(self, input_dir: str, cifar100_classes: List[str]):
        self.input_dir = Path(input_dir)
        self.cifar100_classes = cifar100_classes
        self._class_image_paths = {}
        self._load_image_paths()
    
    def _load_image_paths(self):
        """Load all image paths organized by class."""
        for class_name in self.cifar100_classes:
            class_dir = self.input_dir / class_name
            if class_dir.exists():
                image_files = []
                for ext in ['.png', '.jpg', '.jpeg']:
                    image_files.extend(class_dir.glob(f'*{ext}'))
                self._class_image_paths[class_name] = image_files
            else:
                self._class_image_paths[class_name] = []
    
    def get_random_image(self, class_name: str) -> Optional[np.ndarray]:
        """Get a random image from specified class."""
        if class_name not in self._class_image_paths:
            return None
        
        image_files = self._class_image_paths[class_name]
        if not image_files:
            return None
        
        # Randomly select an image file
        selected_file = random.choice(image_files)
        
        try:
            image = Image.open(selected_file).convert("RGB")
            return np.array(image)
        except Exception as e:
            logger.warning(f"Failed to load image {selected_file}: {e}")
            return None
    
    def get_random_image_from_other_class(self, current_class: str) -> Optional[np.ndarray]:
        """Get a random image from a different class."""
        other_classes = [cls for cls in self.cifar100_classes if cls != current_class]
        if not other_classes:
            return None
        
        other_class = random.choice(other_classes)
        return self.get_random_image(other_class)


# Custom Albumentations Transforms
class Mixup(ImageOnlyTransform):
    """Custom Mixup transform for Albumentations pipeline."""
    
    def __init__(self, class_name: str, alpha: float = 0.2, p: float = 1.0, 
                 mix_type: str = "intra_class", dataset_sampler: Optional[DatasetSampler] = None):
        super().__init__(p=p)
        self.class_name = class_name
        self.alpha = alpha
        self.mix_type = mix_type  # "intra_class" or "inter_class"
        self.cache_manager: Optional[CacheManager] = None
        self.cifar100_classes: List[str] = []
        self.dataset_sampler: Optional[DatasetSampler] = dataset_sampler
    
    def set_cache_manager(self, cache_manager: CacheManager):
        """Set the cache manager for this transform."""
        self.cache_manager = cache_manager
    
    def set_cifar100_classes(self, cifar100_classes: List[str]):
        """Set the CIFAR100 classes for this transform."""
        self.cifar100_classes = cifar100_classes
    
    def set_dataset_sampler(self, dataset_sampler: DatasetSampler):
        """Set the dataset sampler for this transform."""
        self.dataset_sampler = dataset_sampler
    
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply mixup augmentation using cached images or random sampling."""
        class_name = self.class_name
        
        if self.cache_manager is not None:
            # Use cache if available
            if self.mix_type == "intra_class":
                sample = self.cache_manager.get(class_name)
            else:  # inter_class
                other_class = self._get_random_other_class(class_name)
                if other_class is None:
                    return image
                sample = self.cache_manager.get(other_class)
            
            if sample is None:
                return image
                
            another_image, _ = sample
        else:
            # No cache - use dataset sampler if available
            if self.dataset_sampler is not None:
                if self.mix_type == "intra_class":
                    another_image = self.dataset_sampler.get_random_image(class_name)
                else:  # inter_class
                    another_image = self.dataset_sampler.get_random_image_from_other_class(class_name)
            else:
                another_image = None
            
            if another_image is None:
                return image
        
        # Apply mixup augmentation
        lam = np.random.beta(self.alpha, self.alpha) * self.alpha
        mixed = (1 - lam) * image.astype(np.float32) + lam * another_image.astype(np.float32)
        return np.clip(mixed, 0, 255).astype(np.uint8)
    
    def _get_random_other_class(self, current_class: str) -> Optional[str]:
        """Get a random class different from current class."""
        if not self.cifar100_classes:
            return None
        available_classes = [cls for cls in self.cifar100_classes if cls != current_class]
        if not available_classes:
            return None
        return random.choice(available_classes)


class CutMix(ImageOnlyTransform):
    """Custom CutMix transform for Albumentations pipeline."""
    
    def __init__(self, class_name: str, alpha: float = 0.2, p: float = 1.0,
                 mix_type: str = "intra_class", dataset_sampler: Optional[DatasetSampler] = None):
        super().__init__(p=p)
        self.class_name = class_name
        self.alpha = alpha
        self.mix_type = mix_type  # "intra_class" or "inter_class"
        self.cache_manager: Optional[CacheManager] = None
        self.cifar100_classes: List[str] = []
        self.dataset_sampler: Optional[DatasetSampler] = dataset_sampler
    
    def set_cache_manager(self, cache_manager: CacheManager):
        """Set the cache manager for this transform."""
        self.cache_manager = cache_manager
    
    def set_cifar100_classes(self, cifar100_classes: List[str]):
        """Set the CIFAR100 classes for this transform."""
        self.cifar100_classes = cifar100_classes
    
    def set_dataset_sampler(self, dataset_sampler: DatasetSampler):
        """Set the dataset sampler for this transform."""
        self.dataset_sampler = dataset_sampler
    
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply cutmix augmentation using cached images or random sampling."""
        class_name = self.class_name
        
        if self.cache_manager is not None:
            # Use cache if available
            if self.mix_type == "intra_class":
                sample = self.cache_manager.get(class_name)
            else:  # inter_class
                other_class = self._get_random_other_class(class_name)
                if other_class is None:
                    return image
                sample = self.cache_manager.get(other_class)
            
            if sample is None:
                return image
                
            another_image, _ = sample
        else:
            # No cache - use dataset sampler if available
            if self.dataset_sampler is not None:
                if self.mix_type == "intra_class":
                    another_image = self.dataset_sampler.get_random_image(class_name)
                else:  # inter_class
                    another_image = self.dataset_sampler.get_random_image_from_other_class(class_name)
            else:
                another_image = None
            
            if another_image is None:
                return image
        
        h, w = image.shape[:2]
        
        # Generate random bounding box
        lam = np.random.beta(self.alpha, self.alpha) * self.alpha
        cut_rat = np.sqrt(lam)
        cut_w = int(w * cut_rat / 2)
        cut_h = int(h * cut_rat / 2)
        
        # Uniform sampling
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Mix images
        mixed_image = image.copy()
        mixed_image[bby1:bby2, bbx1:bbx2] = another_image[bby1:bby2, bbx1:bbx2]
        
        return mixed_image
    
    def _get_random_other_class(self, current_class: str) -> Optional[str]:
        """Get a random class different from current class."""
        if not self.cifar100_classes:
            return None
        available_classes = [cls for cls in self.cifar100_classes if cls != current_class]
        if not available_classes:
            return None
        return random.choice(available_classes)

class CrossSampleAugmentation(ImageOnlyTransform):
    """Cross-sample augmentation for Albumentations pipeline."""
    
    def __init__(self, 
                 class_name: str,
                 alpha: float = 0.1, p: float = 1.0,
                 cache_manager: Optional[CacheManager] = None,
                 dataset_sampler: Optional[DatasetSampler] = None):

        self.alpha = alpha
        self.class_name = class_name
        self.cache_manager = cache_manager
        self.dataset_sampler = dataset_sampler
        super().__init__(p=p)
    
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply cross-sample augmentation using cached images or random sampling."""
        class_name = self.class_name

        # Choose augmentation strategy
        mix_type = np.random.choice(["mixup", "cutmix"], p=[0.5, 0.5])
        alpha = self.alpha
        
        if mix_type == "mixup":
            mixup_transform = Mixup(class_name=class_name, alpha=alpha, p=1.0, mix_type="intra_class", dataset_sampler=self.dataset_sampler)
            if self.cache_manager is not None:
                mixup_transform.set_cache_manager(self.cache_manager)
            mixed_image = mixup_transform.apply(image)
        else:
            cutmix_transform = CutMix(class_name=class_name, alpha=alpha, p=1.0, mix_type="intra_class", dataset_sampler=self.dataset_sampler)
            if self.cache_manager is not None:
                cutmix_transform.set_cache_manager(self.cache_manager)
            mixed_image = cutmix_transform.apply(image)
        
        return np.clip(mixed_image, 0, 255).astype(np.uint8)


class CrossClassAugmentation(ImageOnlyTransform):
    """Cross-class augmentation for Albumentations pipeline."""
    
    def __init__(self, class_name: str,
                 alpha: float = 0.15, p: float = 1.0,
                 cifar100: CIFAR100 = None,
                 dataset_sampler: Optional[DatasetSampler] = None,
                 cache_manager: Optional[CacheManager] = None):

        self.class_name = class_name
        self.alpha = alpha
        self.cifar100 = cifar100
        self.cache_manager = cache_manager
        self.dataset_sampler = dataset_sampler
        super().__init__(p=p)
    
    
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply cross-class augmentation using cached images or random sampling."""
        class_name_1 = self.class_name
        class_name_2 = self._get_random_other_class(class_name_1)
        
        if class_name_2 is None:
            return image

        # Choose augmentation strategy based on confusion pairs
        if self.cifar100 and self.cifar100.is_confusion_pair(class_name_1, class_name_2):
            # Use lighter augmentation for confusion pairs
            alpha = self.alpha * 0.5  # Reduced intensity
        else:
            # Use normal augmentation for non-confusion pairs
            alpha = self.alpha

        mix_type = np.random.choice(["mixup", "cutmix"], p=[0.5, 0.5])
        if mix_type == "mixup":
            mixup_transform = Mixup(class_name=class_name_1, alpha=alpha, p=1.0, mix_type="inter_class", dataset_sampler=self.dataset_sampler)
            if self.cache_manager is not None:
                mixup_transform.set_cache_manager(self.cache_manager)
            mixed_image = mixup_transform.apply(image)
        else:
            cutmix_transform = CutMix(class_name=class_name_1, alpha=alpha, p=1.0, mix_type="inter_class", dataset_sampler=self.dataset_sampler)
            if self.cache_manager is not None:
                cutmix_transform.set_cache_manager(self.cache_manager)
            mixed_image = cutmix_transform.apply(image)
        
        return np.clip(mixed_image, 0, 255).astype(np.uint8)
    
    def _get_random_other_class(self, current_class: str) -> Optional[str]:
        """Get a random class different from current class."""
        if not self.cifar100 or not self.cifar100.cifar100_list:
            return None
        available_classes = [cls for cls in self.cifar100.cifar100_list if cls != current_class]
        if not available_classes:
            return None
        return random.choice(available_classes)
    
class ApplyTechniques(ImageOnlyTransform):
    """Apply techniques to the image based on the requirements."""

    def __init__(self, requirements: list[str], class_name: str = None, dataset_sampler=None, p: float = 1.0, 
                 selection_ratio: float = 0.6, min_selections: int = 1):
        super().__init__(p=p)
        self.requirements = requirements if requirements else []
        self.class_name = class_name
        self.dataset_sampler = dataset_sampler
        self.random_generator = np.random.default_rng(seed=42)
        self.selection_ratio = selection_ratio  # 选择比例，默认选择60%的技术
        self.min_selections = min_selections  # 最少选择的技术数量

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply randomly selected techniques to the image."""
        augmented = image.copy()
        
        # 如果没有requirements，直接返回原图
        if not self.requirements:
            return augmented
        
        # 随机选择要执行的技术
        selected_techniques = self._select_random_techniques()
        
        for requirement in selected_techniques:
            if requirement == 'face_enhancement':
                augmented = self._apply_face_enhancement(augmented)
            elif requirement == 'pose_variation':
                augmented = self._apply_pose_variation(augmented)
            elif requirement == 'clothing_diversity':
                augmented = self._apply_clothing_diversity(augmented)
            elif requirement == 'age_distinction':
                augmented = self._apply_age_distinction(augmented)
            elif requirement == 'texture_enhancement':
                augmented = self._apply_texture_enhancement(augmented)
            elif requirement == 'flower_petal_enhancement':
                augmented = self._apply_flower_petal_enhancement(augmented)
            elif requirement == 'vehicle_structure_enhancement':
                augmented = self._apply_vehicle_structure_enhancement(augmented)
            elif requirement == 'environment_diversity':
                augmented = self._apply_environment_diversity(augmented)
            elif requirement == 'shape_variation':
                augmented = self._apply_shape_variation(augmented)
            elif requirement == 'size_variation':
                augmented = self._apply_size_variation(augmented)
            elif requirement == 'gender_distinction':
                augmented = self._apply_gender_distinction(augmented)
            elif requirement == 'tree_structure_distinction':
                augmented = self._apply_tree_structure_distinction(augmented)
            elif requirement == 'flower_structure_enhancement':
                augmented = self._apply_flower_structure_enhancement(augmented)
            elif requirement == 'vehicle_structure_distinction':
                augmented = self._apply_vehicle_structure_distinction(augmented)
            elif requirement == 'person_context_enhancement':
                augmented = self._apply_person_context_enhancement(augmented)
            elif requirement == 'adult_context_enhancement':
                augmented = self._apply_adult_context_enhancement(augmented)
            elif requirement == 'leaf_pattern_enhancement':
                augmented = self._apply_leaf_pattern_enhancement(augmented)
            elif requirement == 'bark_texture_enhancement':
                augmented = self._apply_bark_texture_enhancement(augmented)
            elif requirement == 'flower_color_enhancement':
                augmented = self._apply_flower_color_enhancement(augmented)
            elif requirement == 'petal_texture_enhancement':
                augmented = self._apply_petal_texture_enhancement(augmented)
            elif requirement == 'vehicle_context_enhancement':
                augmented = self._apply_vehicle_context_enhancement(augmented)
            elif requirement == 'container_shape_enhancement':
                augmented = self._apply_container_shape_enhancement(augmented)
            elif requirement == 'container_usage_context':
                augmented = self._apply_container_usage_context(augmented)
            elif requirement == 'material_texture_enhancement':
                augmented = self._apply_material_texture_enhancement(augmented)
            elif requirement == 'size_context_enhancement':
                augmented = self._apply_size_context_enhancement(augmented)
            elif requirement == 'marine_animal_texture':
                augmented = self._apply_marine_animal_texture(augmented)
            elif requirement == 'aquatic_context_enhancement':
                augmented = self._apply_aquatic_context_enhancement(augmented)
            elif requirement == 'animal_shape_distinction':
                augmented = self._apply_animal_shape_distinction(augmented)
            elif requirement == 'water_environment_enhancement':
                augmented = self._apply_water_environment_enhancement(augmented)
            elif requirement == 'insect_detail_enhancement':
                augmented = self._apply_insect_detail_enhancement(augmented)
            elif requirement == 'insect_texture_enhancement':
                augmented = self._apply_insect_texture_enhancement(augmented)
            elif requirement == 'insect_shape_distinction':
                augmented = self._apply_insect_shape_distinction(augmented)
            elif requirement == 'natural_environment_enhancement':
                augmented = self._apply_natural_environment_enhancement(augmented)
        
        return augmented
    
    def _select_random_techniques(self) -> list[str]:
        """随机选择要执行的技术"""
        if not self.requirements:
            return []
        
        # 计算要选择的技术数量
        total_techniques = len(self.requirements)
        num_to_select = max(
            self.min_selections,
            min(total_techniques, int(total_techniques * self.selection_ratio))
        )
        
        # 随机选择技术
        selected = self.random_generator.choice(
            self.requirements, 
            size=num_to_select, 
            replace=False
        )
        
        return selected.tolist()
    
    def _apply_face_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance facial features using sharpening."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(image.astype(np.float32), -1, kernel)
        alpha = np.random.uniform(0.05, 0.1)  # Reduced from 0.1-0.2
        enhanced = np.clip(image * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)
        return enhanced
    
    def _apply_pose_variation(self, image: np.ndarray) -> np.ndarray:
        """Apply pose variations for human subjects."""
        h, w = image.shape[:2]
        
        # Random rotation (optimized for 32x32)
        angle = np.random.uniform(-3, 3)  # Further reduced for small images
        
        # Random shear (optimized for 32x32)
        shear_x = np.random.uniform(-0.03, 0.03)  # Further reduced
        shear_y = np.random.uniform(-0.03, 0.03)  # Further reduced
        
        # Create transformation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Add shear
        M[0, 1] += shear_x
        M[1, 0] += shear_y
        
        # Apply transformation
        transformed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return transformed
    
    def _apply_clothing_diversity(self, image: np.ndarray) -> np.ndarray:
        """Apply clothing diversity through color changes."""
        # Convert to HSV for clothing color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Random hue shift for clothing (reduced)
        hue_shift = np.random.uniform(-10, 10)  # Reduced from -20, 20
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Random saturation adjustment (reduced)
        sat_factor = np.random.uniform(0.9, 1.1)  # Reduced from 0.8, 1.2
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb
    
    def _apply_age_distinction(self, image: np.ndarray) -> np.ndarray:
        """Apply age-specific enhancements (optimized for 32x32)."""
        augmented = image.copy()
        
        # Apply subtle brightness and contrast adjustments
        if np.random.random() < 0.5:
            # Makes the image look younger - softer features
            blurred = cv2.GaussianBlur(augmented, (3, 3), sigmaX=0.5)  # Lighter blur for 32x32
            alpha = np.random.uniform(0.85, 0.95)  # More conservative mixing
            augmented = cv2.addWeighted(blurred, alpha, augmented, 1 - alpha, 0)
            augmented = cv2.convertScaleAbs(augmented, alpha=1.01, beta=3)  # Lighter adjustment
        else:
            # Makes the image look older - sharper features
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(augmented.astype(np.float32), -1, kernel)
            alpha = np.random.uniform(0.05, 0.1)  # Much lighter sharpening
            augmented = np.clip(augmented * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)
            
        return augmented

    def _apply_texture_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance texture details using edge detection."""
        edges = cv2.Canny(image, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        alpha = np.random.uniform(0.02, 0.08)  # Reduced from 0.05, 0.15
        enhanced = np.clip(image * (1 - alpha) + edges_colored * alpha, 0, 255).astype(np.uint8)
        return enhanced
    
    def _apply_flower_petal_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance flower petal details and colors."""
        # Convert to HSV for flower enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Enhance saturation for vibrant petals
        sat_factor = np.random.uniform(1.1, 1.3)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # Adjust brightness
        brightness_factor = np.random.uniform(0.95, 1.05)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb
    
    def _apply_vehicle_structure_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance vehicle structure and metallic surfaces."""
        # Enhance metallic surfaces
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(image.astype(np.float32), -1, kernel)
        alpha = np.random.uniform(0.1, 0.2)
        
        # Apply sharpening
        enhanced = np.clip(image * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)
        
        # Enhance contrast for metallic surfaces
        contrast_factor = np.random.uniform(1.05, 1.15)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_factor, beta=0)
        
        return enhanced
    
    def _apply_environment_diversity(self, image: np.ndarray) -> np.ndarray:
        """Apply environment diversity through lighting changes."""
        # Simulate different lighting conditions
        temp_factor = np.random.uniform(0.9, 1.1)
        atm_factor = np.random.uniform(0.95, 1.05)
        
        # Adjust color temperature (warm/cool)
        if temp_factor > 1.0:  # Warm lighting
            image[:, :, 0] = np.clip(image[:, :, 0] * temp_factor, 0, 255)  # Increase red
            image[:, :, 2] = np.clip(image[:, :, 2] / temp_factor, 0, 255)  # Decrease blue
        else:  # Cool lighting
            image[:, :, 0] = np.clip(image[:, :, 0] / temp_factor, 0, 255)  # Decrease red
            image[:, :, 2] = np.clip(image[:, :, 2] * temp_factor, 0, 255)  # Increase blue
        
        # Apply atmospheric effects
        image = cv2.convertScaleAbs(image, alpha=atm_factor, beta=0)
        
        return image
    
    def _apply_shape_variation(self, image: np.ndarray) -> np.ndarray:
        """Apply very subtle shape variations (optimized for 32x32)."""
        h, w = image.shape[:2]
        
        # Much lighter displacement for small images
        alpha = np.random.uniform(0.02, 0.05) * w  # 0.64-1.6 pixels only
        sigma = np.random.uniform(6, 8)  # Higher smoothing
        
        # Generate random displacement
        dx = np.random.normal(0, 1, (h, w))
        dy = np.random.normal(0, 1, (h, w))
        
        # Apply Gaussian blur to displacement field
        dx = cv2.GaussianBlur(dx, (5, 5), sigma) * alpha  # Smaller kernel
        dy = cv2.GaussianBlur(dy, (5, 5), sigma) * alpha
        
        # Create coordinate maps
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply elastic transformation
        transformed = cv2.remap(image, map_x, map_y, 
                              interpolation=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REFLECT)
        return transformed
    
    def _apply_size_variation(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle size variations (optimized for 32x32)."""
        scale_factor = np.random.uniform(0.97, 1.03)  # Very conservative for small images
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # Scale down/up then back to original size
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _apply_gender_distinction(self, image: np.ndarray) -> np.ndarray:
        """Apply gender-specific enhancements (optimized for 32x32)."""
        augmented = image.copy()
        
        if self.class_name in ['man', 'woman']:
            # Emphasize adult features with sharper details
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(augmented.astype(np.float32), -1, kernel)
            alpha = np.random.uniform(0.08, 0.15)  # Much lighter
            augmented = np.clip(augmented * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)
        
        elif self.class_name in ['boy', 'girl']:
            # Emphasize child features with very subtle blur
            augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=0.5)  # Lighter blur

        elif self.class_name in ['baby']:
            # Emphasize baby features with subtle blur
            augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=0.7)  # Slightly more blur
            
        return augmented
    
    def _apply_tree_structure_distinction(self, image: np.ndarray) -> np.ndarray:
        """Apply tree-specific enhancement (optimized for 32x32)."""
        augmented = image.copy()
        
        if self.class_name == 'oak_tree':
            # Enhance bark texture and leaf shape (lighter for small images)
            augmented = cv2.convertScaleAbs(augmented, alpha=1.1, beta=5)  # Reduced intensity
            # Use moderate sharpening instead of strong
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(augmented.astype(np.float32), -1, kernel)
            augmented = np.clip(augmented * 0.85 + sharpened * 0.15, 0, 255).astype(np.uint8)  # Lighter mix
            
        elif self.class_name == 'maple_tree':
            # Enhance maple leaf characteristics
            hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Lighter saturation boost
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            # Lighter edge enhancement
            edges = cv2.Canny(augmented, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            augmented = np.clip(augmented * 0.95 + edges_colored * 0.05, 0, 255).astype(np.uint8)
        
        return augmented
    
    def _apply_flower_structure_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply flower-specific enhancement (optimized for 32x32)."""
        augmented = image.copy()
        
        if self.class_name == 'rose':
            # Enhance rose characteristics - layered petals (lighter for 32x32)
            augmented = cv2.convertScaleAbs(augmented, alpha=1.15, beta=8)  # Reduced intensity
            # Enhance red/pink tones
            hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 1.25  # Lighter saturation boost
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        elif self.class_name == 'tulip':
            # Enhance tulip characteristics - cup shape (very subtle for small images)
            h, w = augmented.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([[2, 2], [w-2, 2], [0, h], [w, h]])  # Much smaller distortion
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            augmented = cv2.warpPerspective(augmented, matrix, (w, h))
            # Enhance brightness (lighter)
            augmented = cv2.convertScaleAbs(augmented, alpha=1.05, beta=10)  # Reduced
        
        return augmented
    
    def _apply_vehicle_structure_distinction(self, image: np.ndarray) -> np.ndarray:
        """Apply vehicle-specific enhancement for bus vs streetcar."""
        augmented = image.copy()
        
        if self.class_name == 'bus':
            # Enhance bus characteristics - rectangular shape, windows
            augmented = cv2.convertScaleAbs(augmented, alpha=1.2, beta=10)
            # Enhance edges to emphasize windows and doors
            edges = cv2.Canny(augmented, 30, 100)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            augmented = np.clip(augmented * 0.8 + edges_colored * 0.2, 0, 255).astype(np.uint8)
            
        elif self.class_name == 'streetcar':
            # Enhance streetcar characteristics - elongated shape, tracks
            h, w = augmented.shape[:2]
            # Slight horizontal stretching to emphasize length
            augmented = cv2.resize(augmented, (int(w * 1.1), h))
            augmented = cv2.resize(augmented, (w, h))
            # Enhance contrast for track details
            augmented = cv2.convertScaleAbs(augmented, alpha=1.15, beta=5)
        
        return augmented

    def _apply_person_context_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强人物上下文特征，帮助区分不同年龄段的人物"""
        augmented = image.copy()

        # 增强边缘特征以突出人物轮廓
        edges = cv2.Canny(augmented, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # 轻微混合边缘信息
        alpha = np.random.uniform(0.05, 0.1)
        augmented = cv2.addWeighted(augmented, 1 - alpha, edges_colored, alpha, 0)

        # 增强对比度以突出人物特征
        lab = cv2.cvtColor(augmented, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * np.random.uniform(1.02, 1.05), 0, 255)
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return augmented

    def _apply_adult_context_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强成人上下文特征"""
        augmented = image.copy()

        # 增强面部特征
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(augmented.astype(np.float32), -1, kernel)
        alpha = np.random.uniform(0.08, 0.12)
        augmented = np.clip(augmented * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)

        return augmented

    def _apply_leaf_pattern_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强叶子纹理模式，帮助区分不同树种"""
        augmented = image.copy()

        # 增强纹理特征
        gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.uint8(np.absolute(texture))
        texture_colored = cv2.cvtColor(texture, cv2.COLOR_GRAY2RGB)

        alpha = np.random.uniform(0.1, 0.2)
        augmented = cv2.addWeighted(augmented, 1 - alpha, texture_colored, alpha, 0)

        return augmented

    def _apply_bark_texture_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强树皮纹理特征"""
        augmented = image.copy()

        # 增强局部对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(augmented, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return augmented

    def _apply_flower_color_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强花朵颜色特征"""
        augmented = image.copy()

        # 增强饱和度
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(1.1, 1.3), 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return augmented

    def _apply_petal_texture_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强花瓣纹理"""
        augmented = image.copy()

        # 增强细节
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(augmented.astype(np.float32), -1, kernel)
        alpha = np.random.uniform(0.1, 0.15)
        augmented = np.clip(augmented * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)

        return augmented

    def _apply_vehicle_context_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强车辆上下文特征"""
        augmented = image.copy()

        # 增强边缘特征以突出车辆结构
        gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        alpha = np.random.uniform(0.08, 0.12)
        augmented = cv2.addWeighted(augmented, 1 - alpha, edges_colored, alpha, 0)

        return augmented

    def _apply_container_shape_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强容器形状特征，帮助区分bowl和plate"""
        augmented = image.copy()

        try:
            # 增强轮廓特征
            gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 创建轮廓掩码
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, contours, -1, 255, 2)
                mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

                alpha = np.random.uniform(0.1, 0.15)
                augmented = cv2.addWeighted(augmented, 1 - alpha, mask_colored, alpha, 0)
        except Exception:
            logger.info(f"Container shape enhancement failed for {self.class_name}")
            # 如果轮廓检测失败，使用简单的边缘增强
            edges = cv2.Canny(augmented, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            alpha = np.random.uniform(0.05, 0.1)
            augmented = cv2.addWeighted(augmented, 1 - alpha, edges_colored, alpha, 0)

        return augmented

    def _apply_container_usage_context(self, image: np.ndarray) -> np.ndarray:
        """增强容器使用上下文"""
        augmented = image.copy()

        # 增强对比度以突出容器特征
        lab = cv2.cvtColor(augmented, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * np.random.uniform(1.05, 1.1), 0, 255)
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return augmented

    def _apply_material_texture_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强材质纹理"""
        augmented = image.copy()

        # 增强局部纹理
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        lab = cv2.cvtColor(augmented, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return augmented

    def _apply_size_context_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强尺寸上下文特征"""
        augmented = image.copy()

        # 轻微调整亮度和对比度以模拟不同尺寸的视觉效果
        alpha = np.random.uniform(0.95, 1.05)
        beta = np.random.uniform(-5, 5)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

        return augmented

    def _apply_marine_animal_texture(self, image: np.ndarray) -> np.ndarray:
        """增强海洋动物纹理特征"""
        augmented = image.copy()

        # 增强水波纹效果
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(augmented.astype(np.float32), -1, kernel)
        alpha = np.random.uniform(0.1, 0.15)
        augmented = np.clip(augmented * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)

        return augmented

    def _apply_aquatic_context_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强水生环境上下文"""
        augmented = image.copy()

        # 增强蓝色调以模拟水生环境
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-5, 5)) % 180  # 轻微色调调整
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return augmented

    def _apply_animal_shape_distinction(self, image: np.ndarray) -> np.ndarray:
        """增强动物形状区分"""
        augmented = image.copy()

        # 增强边缘特征
        gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        alpha = np.random.uniform(0.05, 0.1)
        augmented = cv2.addWeighted(augmented, 1 - alpha, edges_colored, alpha, 0)

        return augmented

    def _apply_water_environment_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强水环境特征"""
        augmented = image.copy()

        # 轻微调整色调以模拟水环境
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(1.05, 1.1), 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return augmented

    def _apply_insect_detail_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强昆虫细节特征"""
        augmented = image.copy()

        # 增强细节
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(augmented.astype(np.float32), -1, kernel)
        alpha = np.random.uniform(0.1, 0.15)
        augmented = np.clip(augmented * (1 - alpha) + sharpened * alpha, 0, 255).astype(np.uint8)

        return augmented

    def _apply_insect_texture_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强昆虫纹理"""
        augmented = image.copy()

        # 增强局部对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab = cv2.cvtColor(augmented, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        augmented = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return augmented

    def _apply_insect_shape_distinction(self, image: np.ndarray) -> np.ndarray:
        """增强昆虫形状区分"""
        augmented = image.copy()

        # 增强轮廓特征
        gray = cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        alpha = np.random.uniform(0.08, 0.12)
        augmented = cv2.addWeighted(augmented, 1 - alpha, edges_colored, alpha, 0)

        return augmented

    def _apply_natural_environment_enhancement(self, image: np.ndarray) -> np.ndarray:
        """增强自然环境特征"""
        augmented = image.copy()

        # 增强自然色调
        hsv = cv2.cvtColor(augmented, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(1.05, 1.1), 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return augmented
        
class SmartRotation(ImageOnlyTransform):
    def __init__(self, class_name: str, p: float = 0.5):
        super().__init__(p=p)
        self.class_name = class_name
        
        # 创建CIFAR100实例来获取superclass信息
        cifar100_dict = {
                "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
                "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
                "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
                "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
                "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
                "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
                "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
                "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
                "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
                "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
                "people": ["baby", "boy", "girl", "man", "woman"],
                "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]}
        confusion_tuples_list = []
        cifar100 = CIFAR100(cifar100_dict, confusion_tuples_list)
        self.superclass_name = cifar100.get_superclass_name(class_name)
        
        # 可任意旋转类别
        whatever = [
            'fish', 'flowers', 'fruit_and_vegetables',
            'non_insect_invertebrates'
        ]

        # 小角度或180度可接受
        constrained = [
            'aquatic_mammals', 'insects',
            'reptiles', 'small_mammals'
        ]

        # 禁止额外旋转
        forbidden = [
            'large_man_made_outdoor_things', 'people',
            'vehicles_1', 'vehicles_2', 'trees',
            'household_electrical_devices', 'household_furniture',
            'large_carnivores', 'large_omnivores_and_herbivores',
            'large_natural_outdoor_scenes', 'food_containers',
            'medium_sized_mammals'
        ]

        self.transform_list = None

        if self.superclass_name in whatever:
            self.transform_list = A.Rotate(180, p=0.3)
        elif self.superclass_name in constrained:
            self.transform_list = A.OneOf([A.RandomRotate90(p=0.8),
                                           A.Rotate(180, p=0.2)])
        elif self.superclass_name in forbidden:
            self.transform_list = None
        else:
            raise ValueError(f"Invalid superclass name: {self.superclass_name}")
    
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """应用智能旋转增强"""
        if self.transform_list is None:
            return image  # 无旋转策略
            
        # 应用Albumentations变换
        result = self.transform_list(image=image)
        return result['image']
        

class ImageAugmenter:
    def __init__(
        self,
        augmentations_per_image: int = 16,
        seed: int = 42,
        random_generator: np.random.Generator = np.random.default_rng(42),
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        enable_cross_sample: bool = True,
        cross_sample_prob: float = 0.3,
        enable_cross_class: bool = True,
        cross_class_prob: float = 0.1,
        cache_size: int = 500,
        max_memory_mb: int = 500,
        batch_processing: bool = True,
        enable_cache: bool = True,  # New parameter to control cache usage
        cifar100: CIFAR100 = None,
        basic_pipeline = None,
    ):
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.random_generator = random_generator
        self.save_original = save_original
        self.image_extensions = image_extensions
        self.enable_cross_sample = enable_cross_sample
        self.cross_sample_prob = cross_sample_prob
        self.enable_cross_class = enable_cross_class
        self.cross_class_prob = cross_class_prob
        self.cache_size = cache_size
        self.max_memory_mb = max_memory_mb
        self.batch_processing = batch_processing
        self.enable_cache = enable_cache  # Store cache enable flag
        self.cifar100 = cifar100
        self.basic_pipeline = basic_pipeline

        self._set_seed()
        self._init_cache_manager()

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.random_generator = np.random.default_rng(seed=self.seed)

    def _init_cache_manager(self) -> None:
        """Initialize cache manager based on enable_cache flag"""
        if self.enable_cache:
            self.cache_manager = CacheManager(
                max_size_per_class=self.cache_size,
                max_memory_mb=self.max_memory_mb,
                enable_lru=True,
                enable_stats=True
            )
            logger.info(f"CacheManager initialized with max_size_per_class={self.cache_size}, "
                       f"max_memory_mb={self.max_memory_mb}")
        else:
            self.cache_manager = None
            logger.info("Cache disabled - no cache manager initialized")

    def _init_dataset_sampler(self, input_dir: str) -> None:
        """Initialize dataset sampler for non-cache augmentation."""
        if not self.enable_cache and self.cifar100 is not None:
            self.dataset_sampler = DatasetSampler(input_dir, self.cifar100.cifar100_list)
            logger.info(f"DatasetSampler initialized for {len(self.cifar100.cifar100_list)} classes")
        else:
            self.dataset_sampler = None
            logger.info("DatasetSampler disabled (cache enabled)")

    def _add_to_cache(self, image_np: np.ndarray, label: int, class_name: str) -> None:
        """Add image to cache if cache is enabled"""
        if self.cache_manager is not None:
            self.cache_manager.add(image_np, label, class_name)

    def _get_from_cache(self, class_name: str) -> Optional[Tuple[np.ndarray, int]]:
        """Get image from cache if cache is enabled"""
        if self.cache_manager is not None:
            return self.cache_manager.get(class_name)
        return None

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all images in directory with optimized batch processing.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Check if output directory already exists and has content
        if output_path.exists() and output_path.is_dir():
            # Check if directory has any image files
            existing_images = list(output_path.rglob("*.png")) + list(output_path.rglob("*.jpg")) + list(output_path.rglob("*.jpeg"))
            if existing_images:
                logger.info(f"📁 Output directory '{output_path}' already exists with {len(existing_images)} images.")
                logger.info(f"✅ Skipping augmentation and using existing augmented data...")
                return
            else:
                logger.info(f"📁 Output directory '{output_path}' exists but is empty. Proceeding with augmentation...")
                # Remove empty directory
                try:
                    output_path.rmdir()
                except OSError:
                    pass  # Directory not empty or other error, continue anyway
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset sampler for non-cache augmentation
        self._init_dataset_sampler(input_dir)
        
        image_files = self._find_image_files(input_path)
        logger.info(f"Found {len(image_files)} images to augment.")
        
        saved_count = 0
        
        # Create progress bar with better description
        progress_bar = tqdm(image_files, desc="Processing images", leave=False)
        
        for img_path in progress_bar:
            try:
                image = Image.open(img_path).convert("RGB")
                image_np = np.array(image)
                
                # Determine output directory and extract class name
                rel_dir = img_path.parent.relative_to(input_path)
                target_dir = output_path / rel_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                
                if rel_dir.parts:
                    class_name = rel_dir.parts[-1]  # Get the last part of the path
                else:
                    path_parts = img_path.parts
                    if len(path_parts) >= 2:
                        class_name = path_parts[-2]  # Second to last part (parent directory)
                    else:
                        class_name = None
                
                label = self.cifar100.get_label(class_name)
                
                # Save original if requested
                if self.save_original:
                    orig_name = f"orig_{img_path.name}"
                    self._save_image(image, target_dir / orig_name)
                
                # Generate augmented versions with special handling for difficult classes
                class_settings = ClassSettings(
                    class_name=class_name, 
                    cifar100=self.cifar100, 
                    basic_pipeline=self.basic_pipeline,
                    cache_manager=self.cache_manager,
                    dataset_sampler=self.dataset_sampler
                )
                multiplier = class_settings.get_augmentation_multiplier()
                pipeline = class_settings.create_pipeline(
                    cache_manager=self.cache_manager,
                    dataset_sampler=self.dataset_sampler
                )
                counts = int(multiplier * self.augmentations_per_image)
                
                # Add current image to cache for cross-sample and cross-class augmentation
                self._add_to_cache(image_np, label, class_name)
                
                
                for i in range(counts):
                    try:
                        # Apply augmentation pipeline
                        augmented_image = pipeline(image=image_np, class_name=class_name)
                        aug_name = f"aug_{i}_{img_path.name}"
                        
                        if self._save_image(augmented_image, target_dir / aug_name):
                            saved_count += 1
                            
                    except Exception as aug_error:
                        logger.warning(f"Failed to augment {img_path.name} (aug {i}): {aug_error}")
                        continue
                
                # Update progress with useful statistics
                if self.cache_manager is not None:
                    cache_stats = self.cache_manager.get_stats()
                    progress_bar.set_postfix({
                        "Saved": saved_count,
                        "Memory": f"{cache_stats['memory_usage_mb']:.1f}MB",
                        "Cached": len(self.cache_manager)
                    })
                else:
                    progress_bar.set_postfix({
                        "Saved": saved_count,
                        "Cache": "Disabled"
                    })
                
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                continue
        
        # Final statistics with detailed cache information
        total_images = len(image_files)
        total_augmented = saved_count
        
        logger.info(f"✅ Augmentation completed!")
        logger.info(f"   📊 Processed: {total_images} images")
        logger.info(f"   🖼️  Generated: {total_augmented} augmented images")
        logger.info(f"   📁 Output directory: {output_dir}")
        
        if self.cache_manager is not None:
            final_cache_stats = self.cache_manager.get_stats()
            logger.info(f"   🎯 Cache hit rate: {final_cache_stats['hit_rate']:.1%}")
            logger.info(f"   📦 Cache size: {final_cache_stats['total_cached_items']} items")
            logger.info(f"   💾 Memory usage: {final_cache_stats['memory_usage_mb']:.1f} MB")
            logger.info(f"   🔄 Cache evictions: {final_cache_stats['evictions']}")
        else:
            logger.info(f"   🚫 Cache: Disabled")

    def _find_image_files(self, root: Path) -> List[Path]:
        """Find all image files recursively."""
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files

    def _save_image(self, image, file_path: Path) -> bool:
        """Save image efficiently."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Handle Albumentations pipeline output (dictionary)
            if isinstance(image, dict):
                if 'image' in image:
                    image = image['image']
                else:
                    logger.warning(f"Invalid image dictionary format: {image.keys()}")
                    return False
            
            # Handle both PIL Image and NumPy array
            if isinstance(image, np.ndarray):
                # Convert NumPy array to PIL Image
                if image.dtype != np.uint8:
                    image = np.clip(image, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
                pil_image.save(file_path, 'PNG', quality=95, optimize=True)
            else:
                # Handle PIL Image
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(file_path, 'PNG', quality=95, optimize=True)
            
            return True
        except Exception as e:
            logger.warning(f"Failed to save {file_path.name}: {e}")
            return False

def clean_corrupted_images(directory: str) -> None:
    """Clean corrupted images from directory."""
    directory_path = Path(directory)
    corrupted_count = 0
    
    logger.info(f"🧹 Cleaning corrupted images in {directory}")
    
    for img_path in directory_path.rglob("*.png"):
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            logger.warning(f"Removing corrupted image: {img_path} - {e}")
            img_path.unlink(missing_ok=True)
            corrupted_count += 1
    
    logger.info(f"✅ Cleaned {corrupted_count} corrupted images")


def force_remove_directory(directory_path: Path) -> bool:
    """
    Forcefully remove a directory and all its contents.
    
    Args:
        directory_path: Path to the directory to remove
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if directory_path.exists():
            logger.info(f"🗑️  Removing directory: {directory_path}")
            try:
                shutil.rmtree(directory_path)
                logger.info(f"✅ Successfully removed: {directory_path}")
                return True
            except Exception as rmtree_error:
                logger.warning(f"shutil.rmtree failed: {rmtree_error}")
                raise rmtree_error
        else:
            logger.info(f"Directory does not exist: {directory_path}")
            return True
    except OSError as e:
        logger.warning(f"Failed to remove directory {directory_path}: {e}")
        logger.info("Attempting alternative removal method...")
        
        # Alternative method: remove files individually
        try:
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        file_path.unlink()
                    except Exception as file_error:
                        logger.warning(f"Failed to remove file {file_path}: {file_error}")
                
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    try:
                        dir_path.rmdir()
                    except Exception as dir_error:
                        logger.warning(f"Failed to remove directory {dir_path}: {dir_error}")
            
            # Try to remove the root directory
            directory_path.rmdir()
            logger.info(f"✅ Successfully removed directory using alternative method: {directory_path}")
            return True
            
        except Exception as alt_error:
            logger.error(f"Failed to remove directory using alternative method: {alt_error}")
            return False
    
    return True


cifar100_dict = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]}
confusion_tuples_list = [
    ('oak_tree', 'maple_tree'),
    ('girl', 'boy'),
    ('baby', 'boy'),
    ('boy', 'man'),
    ('woman', 'boy'),
    ('streetcar', 'bus'),
    ('boy', 'baby'),
    ('man', 'boy'),
    ('rose', 'tulip'),
    ('tulip', 'rose'),
    ('bowl', 'plate'),
    ('snake', 'worm'),
    ('seal', 'otter'),
    ('maple_tree', 'willow_tree'),
    ('girl', 'baby'),
    ('boy', 'girl'),
    ('man', 'woman'),
]


def offline_augment_dataset(
    input_dir: str,
    output_dir: str,
    offline_aug_count: int = 10,
    seed: int = 42,
    enable_cross_sample: bool = True,
    cross_sample_prob: float = 0.3,
    enable_cross_class: bool = True,
    cross_class_prob: float = 0.1,
    cache_size: int = 256,
    max_memory_mb: int = 1024,
    enable_cache: bool = True,
) -> None:
    """
    Enhanced augmentation function with optional cache management.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        augmentations_per_image: Number of augmentations per image
        seed: Random seed
        enable_cross_sample: Whether to enable cross-sample augmentation
        cross_sample_prob: Cross-sample augmentation probability
        enable_cross_class: Whether to enable cross-class augmentation
        cross_class_prob: Cross-class augmentation probability
        cache_size: Cache size per class
        max_memory_mb: Maximum memory usage in MB
        enable_cache: Whether to enable cache management
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=offline_aug_count,
        seed=seed,
        save_original=True,
        enable_cross_sample=enable_cross_sample,
        cross_sample_prob=cross_sample_prob,
        enable_cross_class=enable_cross_class,
        cross_class_prob=cross_class_prob,
        cache_size=cache_size,
        max_memory_mb=max_memory_mb,
        enable_cache=enable_cache,
        cifar100=CIFAR100(cifar100_dict, confusion_tuples_list),
        basic_pipeline=BasicPipeline(),
    )
    augmenter.process_directory(input_dir, output_dir)