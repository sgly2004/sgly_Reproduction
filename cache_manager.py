"""
缓存管理模块 - 处理数据集和模型的缓存机制
"""

import os
import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset, DatasetDict


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 子目录
        self.datasets_cache_dir = self.cache_dir / "datasets"
        self.models_cache_dir = self.cache_dir / "models"
        self.tokenizers_cache_dir = self.cache_dir / "tokenizers"
        
        # 创建子目录
        self.datasets_cache_dir.mkdir(exist_ok=True)
        self.models_cache_dir.mkdir(exist_ok=True)
        self.tokenizers_cache_dir.mkdir(exist_ok=True)
        
        # 缓存索引文件
        self.index_file = self.cache_dir / "cache_index.json"
        self.load_index()
    
    def load_index(self):
        """加载缓存索引"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
            except:
                self.index = {}
        else:
            self.index = {}
    
    def save_index(self):
        """保存缓存索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)
    
    def _generate_hash(self, *args, **kwargs) -> str:
        """生成缓存键的哈希值"""
        # 创建一个唯一的字符串表示
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, cache_key: str) -> Path:
        """获取缓存文件路径"""
        if cache_type == "dataset":
            return self.datasets_cache_dir / f"{cache_key}.pkl"
        elif cache_type == "model":
            return self.models_cache_dir / cache_key
        elif cache_type == "tokenizer":
            return self.tokenizers_cache_dir / cache_key
        else:
            raise ValueError(f"未知的缓存类型: {cache_type}")
    
    def cache_dataset(self, dataset: Union[Dataset, Dict[str, Dataset]], 
                     dataset_name: str, **kwargs) -> str:
        """
        缓存数据集
        
        Args:
            dataset: 要缓存的数据集
            dataset_name: 数据集名称
            **kwargs: 数据集加载参数
        
        Returns:
            缓存键
        """
        cache_key = self._generate_hash(dataset_name, **kwargs)
        cache_path = self._get_cache_path("dataset", cache_key)
        
        # 保存数据集
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        # 更新索引
        self.index[cache_key] = {
            "type": "dataset",
            "name": dataset_name,
            "params": kwargs,
            "file": str(cache_path),
            "size": len(dataset) if hasattr(dataset, '__len__') else "unknown"
        }
        self.save_index()
        
        print(f"数据集 {dataset_name} 已缓存: {cache_key}")
        return cache_key
    
    def load_dataset(self, dataset_name: str, **kwargs) -> Optional[Union[Dataset, Dict[str, Dataset]]]:
        """
        加载缓存的数据集
        
        Args:
            dataset_name: 数据集名称
            **kwargs: 数据集加载参数
        
        Returns:
            缓存的数据集，如果不存在则返回None
        """
        cache_key = self._generate_hash(dataset_name, **kwargs)
        
        if cache_key in self.index and self.index[cache_key]["type"] == "dataset":
            cache_path = Path(self.index[cache_key]["file"])
            
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        dataset = pickle.load(f)
                    print(f"从缓存加载数据集 {dataset_name}: {cache_key}")
                    return dataset
                except Exception as e:
                    print(f"加载缓存数据集失败: {e}")
                    # 删除损坏的缓存
                    self.remove_cache(cache_key)
        
        return None
    
    def cache_model(self, model: PreTrainedModel, model_name: str, **kwargs) -> str:
        """
        缓存模型
        
        Args:
            model: 要缓存的模型
            model_name: 模型名称
            **kwargs: 模型加载参数
        
        Returns:
            缓存键
        """
        cache_key = self._generate_hash(model_name, **kwargs)
        cache_path = self._get_cache_path("model", cache_key)
        
        # 保存模型
        model.save_pretrained(cache_path)
        
        # 更新索引
        self.index[cache_key] = {
            "type": "model",
            "name": model_name,
            "params": kwargs,
            "path": str(cache_path)
        }
        self.save_index()
        
        print(f"模型 {model_name} 已缓存: {cache_key}")
        return cache_key
    
    def load_model(self, model_class, model_name: str, **kwargs) -> Optional[PreTrainedModel]:
        """
        加载缓存的模型
        
        Args:
            model_class: 模型类
            model_name: 模型名称
            **kwargs: 模型加载参数
        
        Returns:
            缓存的模型，如果不存在则返回None
        """
        cache_key = self._generate_hash(model_name, **kwargs)
        
        if cache_key in self.index and self.index[cache_key]["type"] == "model":
            cache_path = Path(self.index[cache_key]["path"])
            
            if cache_path.exists():
                try:
                    model = model_class.from_pretrained(cache_path)
                    print(f"从缓存加载模型 {model_name}: {cache_key}")
                    return model
                except Exception as e:
                    print(f"加载缓存模型失败: {e}")
                    # 删除损坏的缓存
                    self.remove_cache(cache_key)
        
        return None
    
    def cache_tokenizer(self, tokenizer: PreTrainedTokenizer, tokenizer_name: str, **kwargs) -> str:
        """
        缓存分词器
        
        Args:
            tokenizer: 要缓存的分词器
            tokenizer_name: 分词器名称
            **kwargs: 分词器加载参数
        
        Returns:
            缓存键
        """
        cache_key = self._generate_hash(tokenizer_name, **kwargs)
        cache_path = self._get_cache_path("tokenizer", cache_key)
        
        # 保存分词器
        tokenizer.save_pretrained(cache_path)
        
        # 更新索引
        self.index[cache_key] = {
            "type": "tokenizer",
            "name": tokenizer_name,
            "params": kwargs,
            "path": str(cache_path)
        }
        self.save_index()
        
        print(f"分词器 {tokenizer_name} 已缓存: {cache_key}")
        return cache_key
    
    def load_tokenizer(self, tokenizer_class, tokenizer_name: str, **kwargs) -> Optional[PreTrainedTokenizer]:
        """
        加载缓存的分词器
        
        Args:
            tokenizer_class: 分词器类
            tokenizer_name: 分词器名称
            **kwargs: 分词器加载参数
        
        Returns:
            缓存的分词器，如果不存在则返回None
        """
        cache_key = self._generate_hash(tokenizer_name, **kwargs)
        
        if cache_key in self.index and self.index[cache_key]["type"] == "tokenizer":
            cache_path = Path(self.index[cache_key]["path"])
            
            if cache_path.exists():
                try:
                    tokenizer = tokenizer_class.from_pretrained(cache_path)
                    print(f"从缓存加载分词器 {tokenizer_name}: {cache_key}")
                    return tokenizer
                except Exception as e:
                    print(f"加载缓存分词器失败: {e}")
                    # 删除损坏的缓存
                    self.remove_cache(cache_key)
        
        return None
    
    def remove_cache(self, cache_key: str):
        """删除指定的缓存"""
        if cache_key in self.index:
            cache_info = self.index[cache_key]
            
            if cache_info["type"] == "dataset":
                cache_path = Path(cache_info["file"])
                if cache_path.exists():
                    cache_path.unlink()
            else:
                cache_path = Path(cache_info["path"])
                if cache_path.exists():
                    # 删除目录及其内容
                    import shutil
                    shutil.rmtree(cache_path)
            
            del self.index[cache_key]
            self.save_index()
            print(f"已删除缓存: {cache_key}")
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        清空缓存
        
        Args:
            cache_type: 缓存类型 ("dataset", "model", "tokenizer")，None表示清空所有
        """
        keys_to_remove = []
        
        for key, info in self.index.items():
            if cache_type is None or info["type"] == cache_type:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self.remove_cache(key)
        
        print(f"已清空{'所有' if cache_type is None else cache_type}缓存")
    
    def list_cache(self, cache_type: Optional[str] = None) -> Dict[str, Any]:
        """
        列出缓存信息
        
        Args:
            cache_type: 缓存类型 ("dataset", "model", "tokenizer")，None表示列出所有
        
        Returns:
            缓存信息字典
        """
        result = {}
        
        for key, info in self.index.items():
            if cache_type is None or info["type"] == cache_type:
                result[key] = info
        
        return result
    
    def get_cache_size(self) -> Dict[str, int]:
        """获取缓存大小统计"""
        def get_dir_size(path):
            total = 0
            if path.exists():
                if path.is_file():
                    total = path.stat().st_size
                else:
                    for file_path in path.rglob('*'):
                        if file_path.is_file():
                            total += file_path.stat().st_size
            return total
        
        return {
            "datasets": get_dir_size(self.datasets_cache_dir),
            "models": get_dir_size(self.models_cache_dir),
            "tokenizers": get_dir_size(self.tokenizers_cache_dir),
            "total": get_dir_size(self.cache_dir)
        }


# 全局缓存管理器实例
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def clear_all_cache():
    """清空所有缓存"""
    cache_manager = get_cache_manager()
    cache_manager.clear_cache()


def print_cache_info():
    """打印缓存信息"""
    cache_manager = get_cache_manager()
    
    print("=" * 50)
    print("缓存信息")
    print("=" * 50)
    
    # 缓存大小
    sizes = cache_manager.get_cache_size()
    print(f"缓存大小:")
    print(f"  数据集: {sizes['datasets'] / (1024*1024):.2f} MB")
    print(f"  模型: {sizes['models'] / (1024*1024):.2f} MB")
    print(f"  分词器: {sizes['tokenizers'] / (1024*1024):.2f} MB")
    print(f"  总计: {sizes['total'] / (1024*1024):.2f} MB")
    
    # 缓存项目
    for cache_type in ["dataset", "model", "tokenizer"]:
        items = cache_manager.list_cache(cache_type)
        if items:
            print(f"\n{cache_type.title()} 缓存:")
            for key, info in items.items():
                print(f"  {key[:8]}... - {info['name']}")


if __name__ == "__main__":
    print_cache_info()