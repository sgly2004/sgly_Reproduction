"""
CLoRA训练主脚本 - 集成正交正则化的LoRA微调
"""

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import argparse
import json
import sys
from datetime import datetime

# 导入日志系统
from logger_utils import setup_logger, redirect_print_to_logger, restore_print

# 导入自定义模块
from data_processor import get_train_data, get_eval_datasets, get_in_domain_eval_dataset
from model_setup import setup_lora_model
from clora_loss import compute_orthogonal_loss, generate_regularization_matrices
from utils.gradient_processing import get_projected_gradients


class CLoRATrainer(Trainer):
    """
    自定义CLoRA训练器，集成正交正则化损失
    
    修复了损失计算逻辑，添加了详细的调试和验证功能
    """
    
    def __init__(self, regularization_matrices=None, orthogonal_weight=0.1, use_clora=True, 
                 debug_mode=False, logger=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularization_matrices = regularization_matrices
        self.orthogonal_weight = orthogonal_weight
        self.use_clora = use_clora
        self.debug_mode = debug_mode
        self.logger = logger
        
        # 验证正则化矩阵
        if self.use_clora and self.regularization_matrices is not None:
            self._validate_regularization_matrices()
        
        # 初始化统计信息
        self.loss_stats = {
            "total_steps": 0,
            "classification_losses": [],
            "orthogonal_losses": [],
            "total_losses": [],
            "layer_contributions": {}
        }
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写损失计算，添加正交正则化损失
        
        修复了设备兼容性、数值稳定性和损失累加逻辑
        
        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出
        
        Returns:
            损失值（如果return_outputs=True，则返回(loss, outputs)）
        """
        
        # 获取设备信息
        device = next(model.parameters()).device
        
        # 获取标签
        labels = inputs.get("labels")
        
        # 前向传播
        outputs = model(**inputs)
        
        # 计算标准分类损失
        if labels is not None:
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("Model outputs do not contain 'logits'")
            classification_loss = F.cross_entropy(logits, labels)
        else:
            classification_loss = outputs.loss
            if classification_loss is None:
                raise ValueError("Model outputs do not contain loss and no labels provided")
        
        # 确保分类损失在正确的设备上
        classification_loss = classification_loss.to(device)
        
        # 计算总损失
        if self.use_clora and self.regularization_matrices is not None:
            # CLoRA模式：添加正交正则化损失
            orthogonal_loss, layer_details = self._compute_orthogonal_regularization(model)
            
            # 确保正交损失在正确的设备上并且需要梯度
            if not isinstance(orthogonal_loss, torch.Tensor):
                orthogonal_loss = torch.tensor(orthogonal_loss, device=device, requires_grad=True)
            else:
                orthogonal_loss = orthogonal_loss.to(device)
                if not orthogonal_loss.requires_grad:
                    orthogonal_loss.requires_grad_(True)
            
            # 计算总损失
            total_loss = classification_loss + self.orthogonal_weight * orthogonal_loss
            
            # 调试信息
            if self.debug_mode:
                self._log_debug_info(classification_loss, orthogonal_loss, total_loss, layer_details)
            
            # 记录各部分损失（用于监控）
            self._log_training_metrics({
                "classification_loss": classification_loss.item(),
                "orthogonal_loss": orthogonal_loss.item(),
                "orthogonal_weight": self.orthogonal_weight,
                "weighted_orthogonal_loss": (self.orthogonal_weight * orthogonal_loss).item(),
                "total_loss": total_loss.item(),
                "mode": "CLoRA",
                "num_lora_layers": len(layer_details) if layer_details else 0
            })
            
        else:
            # 标准LoRA模式：只使用分类损失
            total_loss = classification_loss
            orthogonal_loss = torch.tensor(0.0, device=device)
            
            # 记录损失（用于监控）
            self._log_training_metrics({
                "classification_loss": classification_loss.item(),
                "orthogonal_loss": 0.0,
                "total_loss": total_loss.item(),
                "mode": "LoRA"
            })
        
        # 更新统计信息
        self._update_loss_stats(classification_loss, orthogonal_loss, total_loss)
        
        # 验证损失的合理性
        if self.debug_mode:
            self._validate_loss_values(classification_loss, orthogonal_loss, total_loss)
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        自定义训练步骤，集成PCA梯度更新机制
        
        如果启用了PCA梯度更新，将在反向传播后对LoRA参数的梯度进行PCA投影处理
        
        Args:
            model: 训练中的模型
            inputs: 输入数据
            num_items_in_batch: 批次中的样本数量
        """
        # 设置模型为训练模式
        model.train()
        
        # 前向传播和损失计算
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # 反向传播计算梯度
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        # PCA梯度更新机制
        if hasattr(self.args, 'use_pca_grad') and self.args.use_pca_grad:
            self._apply_pca_gradient_update(model)
        
        return loss.detach()
    
    def _apply_pca_gradient_update(self, model):
        """
        应用PCA梯度更新机制
        
        Args:
            model: 训练中的模型
        """
        try:
            # 获取当前设备
            device = next(model.parameters()).device
            
            # 获取PCA投影后的梯度
            n_components = getattr(self.args, 'pca_components', 1)
            projected_gradients = get_projected_gradients(
                model=model,
                n_components=n_components,
                device=device
            )
            
            if self.debug_mode and projected_gradients:
                print(f"PCA梯度更新: 处理了 {len(projected_gradients)} 个LoRA参数")
            
            # 将投影后的梯度应用回模型参数
            self._apply_projected_gradients_to_model(model, projected_gradients)
            
        except Exception as e:
            if self.debug_mode:
                print(f"PCA梯度更新失败: {e}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
            # 如果PCA处理失败，继续使用原始梯度
            pass
    
    def _apply_projected_gradients_to_model(self, model, projected_gradients):
        """
        将投影后的梯度应用回模型的LoRA参数
        
        Args:
            model: 训练中的模型
            projected_gradients: get_projected_gradients的输出结果
        """
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # 处理lora_A参数
                if 'default' in module.lora_A:
                    lora_a_key = f"{name}.lora_A"
                    if lora_a_key in projected_gradients:
                        projected_grad = projected_gradients[lora_a_key]['projected']
                        if projected_grad is not None:
                            # 将投影后的梯度赋值给参数的.grad属性
                            module.lora_A['default'].weight.grad = projected_grad.clone()
                
                # 处理lora_B参数
                if 'default' in module.lora_B:
                    lora_b_key = f"{name}.lora_B"
                    if lora_b_key in projected_gradients:
                        projected_grad = projected_gradients[lora_b_key]['projected']
                        if projected_grad is not None:
                            # 将投影后的梯度赋值给参数的.grad属性
                            module.lora_B['default'].weight.grad = projected_grad.clone()
    
    def _compute_orthogonal_regularization(self, model):
        """
        计算所有LoRA层的正交正则化损失
        
        修复了设备兼容性、数值稳定性和累加逻辑
        
        Args:
            model: PEFT模型
        
        Returns:
            tuple: (总的正交正则化损失, 层级详细信息)
        """
        device = next(model.parameters()).device
        total_orthogonal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        layer_details = []
        num_lora_layers = 0
        
        # 遍历所有LoRA层
        for name, module in model.named_modules():
            # 查找LoRA的A和B矩阵
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # 获取A和B矩阵的权重
                for adapter_name in module.lora_A.keys():
                    try:
                        lora_a = module.lora_A[adapter_name].weight  # (r, input_dim)
                        lora_b = module.lora_B[adapter_name].weight  # (output_dim, r)
                        
                        # 确保矩阵在正确的设备上
                        lora_a = lora_a.to(device)
                        lora_b = lora_b.to(device)
                        
                        # 转置以匹配我们的损失函数格式
                        lora_a_transposed = lora_a.T  # (input_dim, r)
                        lora_b_transposed = lora_b.T  # (r, output_dim)
                        
                        # 计算当前层的正交损失
                        if self.regularization_matrices is not None:
                            layer_orthogonal_loss = compute_orthogonal_loss(
                                lora_a_transposed, 
                                lora_b_transposed, 
                                self.regularization_matrices
                            )
                            
                            # 确保损失值是张量并在正确设备上
                            if not isinstance(layer_orthogonal_loss, torch.Tensor):
                                layer_orthogonal_loss = torch.tensor(
                                    layer_orthogonal_loss, device=device, requires_grad=True
                                )
                            else:
                                layer_orthogonal_loss = layer_orthogonal_loss.to(device)
                                if not layer_orthogonal_loss.requires_grad:
                                    layer_orthogonal_loss.requires_grad_(True)
                            
                            # 累加损失
                            total_orthogonal_loss = total_orthogonal_loss + layer_orthogonal_loss
                            num_lora_layers += 1
                            
                            # 记录层级详细信息
                            layer_info = {
                                "layer_name": f"{name}.{adapter_name}",
                                "lora_a_shape": list(lora_a.shape),
                                "lora_b_shape": list(lora_b.shape),
                                "loss_value": layer_orthogonal_loss.item(),
                                "lora_a_norm": torch.norm(lora_a).item(),
                                "lora_b_norm": torch.norm(lora_b).item()
                            }
                            layer_details.append(layer_info)
                            
                            if self.debug_mode:
                                print(f"Layer {layer_info['layer_name']}: "
                                      f"Loss={layer_orthogonal_loss.item():.6f}, "
                                      f"A_norm={layer_info['lora_a_norm']:.6f}, "
                                      f"B_norm={layer_info['lora_b_norm']:.6f}")
                        
                    except Exception as e:
                        print(f"Error processing LoRA layer {name}.{adapter_name}: {e}")
                        continue
        
        # 如果没有找到LoRA层，返回零损失
        if num_lora_layers == 0:
            if self.debug_mode:
                print("Warning: No LoRA layers found for orthogonal regularization")
            return torch.tensor(0.0, device=device, requires_grad=True), []
        
        # 平均化损失
        averaged_loss = total_orthogonal_loss / num_lora_layers
        
        if self.debug_mode:
            print(f"Orthogonal loss summary: {num_lora_layers} layers, "
                  f"total={total_orthogonal_loss.item():.6f}, "
                  f"averaged={averaged_loss.item():.6f}")
        
        return averaged_loss, layer_details
    
    def _validate_regularization_matrices(self):
        """验证正则化矩阵的有效性"""
        if self.regularization_matrices is None:
            raise ValueError("Regularization matrices cannot be None when use_clora=True")
        
        required_keys = ['P_A', 'P_B']
        for key in required_keys:
            if key not in self.regularization_matrices:
                raise ValueError(f"Missing regularization matrix: {key}")
            
            matrix = self.regularization_matrices[key]
            if not isinstance(matrix, torch.Tensor):
                raise ValueError(f"Regularization matrix {key} must be a torch.Tensor")
            
            if matrix.dim() != 2:
                raise ValueError(f"Regularization matrix {key} must be 2D")
            
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Regularization matrix {key} must be square")
        
        if self.debug_mode:
            print("Regularization matrices validation passed")
            for key, matrix in self.regularization_matrices.items():
                print(f"  {key}: shape={matrix.shape}, dtype={matrix.dtype}")
    
    def _log_training_metrics(self, metrics):
        """记录训练指标"""
        if hasattr(self, 'log'):
            self.log(metrics)
    
    def _log_debug_info(self, classification_loss, orthogonal_loss, total_loss, layer_details):
        """记录详细的调试信息"""
        debug_info = f"\n Debug Info [Step {self.loss_stats['total_steps']}]:"
        debug_info += f"\n  Classification Loss: {classification_loss.item():.6f}"
        debug_info += f"\n  Orthogonal Loss: {orthogonal_loss.item():.6f}"
        debug_info += f"\n  Orthogonal Weight: {self.orthogonal_weight}"
        debug_info += f"\n  Weighted Orthogonal: {(self.orthogonal_weight * orthogonal_loss).item():.6f}"
        debug_info += f"\n  Total Loss: {total_loss.item():.6f}"
        debug_info += f"\n  Number of LoRA layers: {len(layer_details)}"
        
        if layer_details and len(layer_details) <= 5:  # 只显示前5层的详细信息
            debug_info += f"\n  Layer contributions:"
            for detail in layer_details:
                debug_info += f"\n    {detail['layer_name']}: {detail['loss_value']:.6f}"
        
        # 输出到控制台和日志
        print(debug_info)
        if self.logger:
            self.logger.debug(debug_info)
    
    def _update_loss_stats(self, classification_loss, orthogonal_loss, total_loss):
        """更新损失统计信息"""
        self.loss_stats["total_steps"] += 1
        self.loss_stats["classification_losses"].append(classification_loss.item())
        self.loss_stats["orthogonal_losses"].append(orthogonal_loss.item())
        self.loss_stats["total_losses"].append(total_loss.item())
        
        # 保持列表长度在合理范围内
        max_history = 1000
        for key in ["classification_losses", "orthogonal_losses", "total_losses"]:
            if len(self.loss_stats[key]) > max_history:
                self.loss_stats[key] = self.loss_stats[key][-max_history:]
    
    def _validate_loss_values(self, classification_loss, orthogonal_loss, total_loss):
        """验证损失值的合理性"""
        def is_valid_loss(loss_tensor, name):
            if torch.isnan(loss_tensor):
                raise ValueError(f"{name} contains NaN values")
            if torch.isinf(loss_tensor):
                raise ValueError(f"{name} contains infinite values")
            if loss_tensor.item() < 0:
                print(f"  Warning: {name} is negative: {loss_tensor.item()}")
            if loss_tensor.item() > 1000:
                print(f" Warning: {name} is very large: {loss_tensor.item()}")
        
        is_valid_loss(classification_loss, "Classification loss")
        is_valid_loss(orthogonal_loss, "Orthogonal loss")
        is_valid_loss(total_loss, "Total loss")
        
        # 验证损失关系
        expected_total = classification_loss + self.orthogonal_weight * orthogonal_loss
        if abs(total_loss.item() - expected_total.item()) > 1e-6:
            print(f" Warning: Loss calculation mismatch. "
                  f"Expected: {expected_total.item():.6f}, Got: {total_loss.item():.6f}")
    
    def get_loss_statistics(self):
        """获取损失统计信息"""
        if not self.loss_stats["total_losses"]:
            return {}
        
        import numpy as np
        return {
            "total_steps": self.loss_stats["total_steps"],
            "avg_classification_loss": np.mean(self.loss_stats["classification_losses"]),
            "avg_orthogonal_loss": np.mean(self.loss_stats["orthogonal_losses"]),
            "avg_total_loss": np.mean(self.loss_stats["total_losses"]),
            "recent_classification_loss": self.loss_stats["classification_losses"][-1] if self.loss_stats["classification_losses"] else 0,
            "recent_orthogonal_loss": self.loss_stats["orthogonal_losses"][-1] if self.loss_stats["orthogonal_losses"] else 0,
            "recent_total_loss": self.loss_stats["total_losses"][-1] if self.loss_stats["total_losses"] else 0
        }


def compute_metrics(eval_pred):
    """
    计算评估指标
    
    Args:
        eval_pred: 预测结果
    
    Returns:
        dict: 评估指标
    """
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def setup_training_args(output_dir="./clora_results", **kwargs):
    """
    设置训练参数
    
    Args:
        output_dir: 输出目录
        **kwargs: 其他训练参数
    
    Returns:
        TrainingArguments: 训练参数对象
    """
    
    default_args = {
        "output_dir": output_dir,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-4,
        "warmup_steps": 100,
        "logging_steps": 100,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 3,
        "report_to": None,  # 禁用wandb等第三方日志
        "dataloader_pin_memory": False,
    }
    
    # 更新默认参数
    default_args.update(kwargs)
    
    return TrainingArguments(**default_args)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description="CLoRA vs LoRA 对比训练脚本")
    
    # 训练模式参数
    parser.add_argument(
        "--use_clora", 
        action="store_true",
        default=True,
        help="是否使用CLoRA训练（默认: True）"
    )
    
    parser.add_argument(
        "--no_clora",
        action="store_true", 
        help="使用纯LoRA训练（等价于 --use_clora=False）"
    )
    
    # 正交损失权重参数
    parser.add_argument(
        "--lambda_param",
        type=float,
        default=0.1,
        help="正交正则化损失权重 λ（默认: 0.1）"
    )
    
    # PCA梯度更新机制参数
    parser.add_argument(
        "--use_pca_grad",
        action="store_true",
        help="启用实验性的PCA梯度更新机制"
    )
    
    parser.add_argument(
        "--pca_components",
        type=int,
        default=1,
        help="PCA梯度投影保留的主成分数量（默认: 1）"
    )
    
    # 输出目录参数
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="训练输出目录（默认自动根据训练模式设置）"
    )
    
    # 训练参数
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数（默认: 3）"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="训练批次大小（默认: 8）"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率（默认: 2e-4）"
    )
    
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=10000,
        help="最大训练样本数（默认: 10000）"
    )
    
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=1000,
        help="最大评估样本数（默认: 1000）"
    )
    
    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备（默认: auto）"
    )
    
    # 调试参数
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="启用详细的调试输出"
    )
    
    args = parser.parse_args()
    
    # 处理互斥参数
    if args.no_clora:
        args.use_clora = False
    
    # 设置默认输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.use_clora:
            args.output_dir = f"./logs/results_clora_{timestamp}"
        else:
            args.output_dir = f"./logs/results_lora_{timestamp}"
    
    return args


def save_training_config(args, output_dir):
    """
    保存训练配置到文件
    
    Args:
        args: 命令行参数
        output_dir: 输出目录
    """
    config = {
        "training_mode": "CLoRA" if args.use_clora else "LoRA",
        "use_clora": args.use_clora,
        "lambda_param": args.lambda_param,
        "use_pca_grad": args.use_pca_grad,
        "pca_components": args.pca_components if args.use_pca_grad else None,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "timestamp": datetime.now().isoformat(),
        "output_dir": output_dir
    }
    
    os.makedirs(output_dir, exist_ok=True)
    config_file = os.path.join(output_dir, "training_config.json")
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"训练配置已保存到: {config_file}")


def main(args=None):
    """主训练函数"""
    
    # 解析命令行参数
    if args is None:
        args = parse_arguments()
    
    # 设置日志系统
    logger_name = f"{'CLoRA' if args.use_clora else 'LoRA'}_training"
    log_dir = os.path.join(args.output_dir, "logs")
    logger = setup_logger(
        name=logger_name,
        log_dir=log_dir,
        console_output=True,
        command_args=sys.argv,
        script_name="train_clora.py"
    )
    
    # 重定向print输出到日志（保持控制台输出）
    original_stdout = redirect_print_to_logger(logger, also_print=False)
    
    try:
        # 记录训练配置
        config_info = {
            "训练模式": f"{'CLoRA (带正交正则化)' if args.use_clora else 'LoRA (标准)'}",
            "输出目录": args.output_dir,
            "训练轮数": args.num_epochs,
            "批次大小": args.batch_size,
            "学习率": args.learning_rate,
            "最大训练样本": args.max_train_samples,
            "最大评估样本": args.max_eval_samples,
            "调试模式": args.debug_mode
        }
        
        if args.use_clora:
            config_info["正交损失权重 (λ)"] = args.lambda_param
        
        if args.use_pca_grad:
            config_info["PCA梯度更新"] = f"启用 (主成分数量: {args.pca_components})"
        
        logger.log_section(f"{'CLoRA' if args.use_clora else 'LoRA'} 训练开始")
        logger.log_config(config_info, "训练配置")
        
        # 保存训练配置
        save_training_config(args, args.output_dir)
        logger.info(f"训练配置已保存到: {args.output_dir}/training_config.json")
    
        # 1. 处理数据集
        logger.log_step(1, "处理数据集")
        
        # 获取训练数据（CommonsenseQA训练集）
        train_dataset, tokenizer = get_train_data(max_samples=args.max_train_samples)
        
        # 获取验证数据集（CommonsenseQA验证集）
        val_dataset = get_in_domain_eval_dataset(tokenizer, max_samples=args.max_eval_samples)
        
        if val_dataset is None:
            raise ValueError("无法加载CommonsenseQA验证数据集，请检查网络连接或数据集可用性")
        
        # 预加载域外评估数据集（用于最终测试）
        out_of_domain_datasets = get_eval_datasets(tokenizer, max_samples_per_dataset=args.max_eval_samples)
        
        dataset_info = {
            "训练集": f"CommonsenseQA train ({len(train_dataset)} 样本)",
            "验证集": f"CommonsenseQA validation ({len(val_dataset)} 样本)",
            "域外测试数据集": list(out_of_domain_datasets.keys())
        }
        logger.log_config(dataset_info, "数据集信息")
    
        # 2. 设置模型
        logger.log_step(2, "设置LoRA模型")
        model = setup_lora_model()
        logger.info("LoRA模型设置完成")
        
        # 3. 生成正则化矩阵（仅当使用CLoRA时）
        regularization_matrices = None
        if args.use_clora:
            logger.log_step(3, "生成正则化矩阵")
            lora_rank = 8  # 与model_setup.py中的rank保持一致
            regularization_matrices = generate_regularization_matrices(
                lora_rank, method='orthogonal'
            )
            matrix_info = {
                "P_A形状": str(regularization_matrices['P_A'].shape),
                "P_B形状": str(regularization_matrices['P_B'].shape),
                "生成方法": "orthogonal",
                "LoRA rank": lora_rank
            }
            logger.log_config(matrix_info, "正则化矩阵信息")
        else:
            logger.log_step(3, "跳过正则化矩阵生成（标准LoRA模式）")
        
        # 4. 设置训练参数
        logger.log_step(4, "设置训练参数")
        training_args = setup_training_args(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size * 2,
            learning_rate=args.learning_rate,
        )
        training_args_info = {
            "训练轮数": args.num_epochs,
            "训练批次大小": args.batch_size,
            "评估批次大小": args.batch_size * 2,
            "学习率": args.learning_rate,
            "输出目录": args.output_dir
        }
        logger.log_config(training_args_info, "训练参数配置")
    
        # 5. 创建数据整理器
        logger.log_step(5, "创建数据整理器")
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt"
        )
        logger.info("数据整理器创建完成")
        
        # 6. 创建自定义训练器
        trainer_name = "CLoRA训练器" if args.use_clora else "LoRA训练器"
        logger.log_step(6, f"创建{trainer_name}")
        
        # 添加调试模式参数
        debug_mode = getattr(args, 'debug_mode', False)
        
        trainer = CLoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            regularization_matrices=regularization_matrices,
            orthogonal_weight=args.lambda_param,
            use_clora=args.use_clora,
            debug_mode=debug_mode,
            logger=logger
        )
        
        trainer_info = {
            "训练器类型": trainer_name,
            "正交损失权重": args.lambda_param if args.use_clora else "N/A",
            "调试模式": debug_mode,
            "训练数据集大小": len(train_dataset),
            "验证数据集大小": len(val_dataset)
        }
        logger.log_config(trainer_info, "训练器配置")
    
        # 7. 开始训练
        logger.log_section("开始训练阶段")
        logger.info("正在启动训练过程...")
        
        # 训练模型
        trainer.train()
        logger.info("训练过程完成")
        
        # 8. 评估模型
        logger.log_section("模型评估阶段")
        
        # 8.1 域内验证评估（CommonsenseQA validation）
        logger.log_step("8.1", "域内验证评估（CommonsenseQA）")
        in_domain_results = trainer.evaluate()
        
        in_domain_metrics = {}
        for key, value in in_domain_results.items():
            if isinstance(value, float):
                in_domain_metrics[key] = f"{value:.4f}"
            else:
                in_domain_metrics[key] = str(value)
        
        logger.log_results(in_domain_metrics, "域内验证结果（CommonsenseQA）")
        
        # 8.2 域外测试评估（BoolQ, ARC, HellaSwag等）
        logger.log_step("8.2", "域外测试评估")
        out_of_domain_results = {}
        
        for dataset_name, dataset in out_of_domain_datasets.items():
            logger.info(f"正在评估 {dataset_name.upper()} 数据集...")
            
            # 临时更换评估数据集
            trainer.eval_dataset = dataset
            ood_results = trainer.evaluate()
            
            # 格式化结果
            ood_metrics = {}
            for key, value in ood_results.items():
                if isinstance(value, float):
                    ood_metrics[key] = f"{value:.4f}"
                else:
                    ood_metrics[key] = str(value)
            
            out_of_domain_results[dataset_name] = ood_metrics
            logger.log_results(ood_metrics, f"域外测试结果（{dataset_name.upper()}）")
        
        # 恢复原始验证数据集
        trainer.eval_dataset = val_dataset
        
        # 汇总所有评估结果
        eval_results = {
            "in_domain": in_domain_results,
            "out_of_domain": out_of_domain_results
        }
    
        # 记录训练损失统计信息
        loss_stats = trainer.get_loss_statistics()
        if loss_stats:
            loss_metrics = {
                "总训练步数": loss_stats['total_steps'],
                "平均分类损失": f"{loss_stats['avg_classification_loss']:.6f}",
                "平均总损失": f"{loss_stats['avg_total_loss']:.6f}",
                "最终分类损失": f"{loss_stats['recent_classification_loss']:.6f}",
                "最终总损失": f"{loss_stats['recent_total_loss']:.6f}"
            }
            
            if args.use_clora:
                loss_metrics.update({
                    "平均正交损失": f"{loss_stats['avg_orthogonal_loss']:.6f}",
                    "最终正交损失": f"{loss_stats['recent_orthogonal_loss']:.6f}"
                })
            
            logger.log_metrics(loss_metrics, "训练损失统计")
    
        # 9. 保存模型
        logger.log_step(7, "保存模型")
        final_model_dir = os.path.join(args.output_dir, "final_model")
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        logger.info(f"模型已保存到: {final_model_dir}")
    
        # 保存训练总结
        training_summary = {
            "training_mode": "CLoRA" if args.use_clora else "LoRA",
            "in_domain_eval_results": {key: value for key, value in in_domain_results.items() if not key.startswith('eval_')},
            "out_of_domain_eval_results": {
                dataset_name: {key: value for key, value in results.items() if not key.startswith('eval_')}
                for dataset_name, results in out_of_domain_results.items()
            },
            "total_training_time": "训练完成",
            "model_path": final_model_dir,
            "config": {
                "use_clora": args.use_clora,
                "lambda_param": args.lambda_param if args.use_clora else None,
                "use_pca_grad": args.use_pca_grad,
                "pca_components": args.pca_components if args.use_pca_grad else None,
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate
            }
        }
        
        summary_file = os.path.join(args.output_dir, "training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(training_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练总结已保存到: {summary_file}")
        
        # 记录训练完成
        logger.log_section(f"{'CLoRA' if args.use_clora else 'LoRA'} 训练完成", "🎉")
        
        completion_summary = {
            "训练模式": "CLoRA" if args.use_clora else "LoRA",
            "模型路径": final_model_dir,
            "总结文件": summary_file,
            "日志文件": logger.get_log_file_path()
        }
        logger.log_config(completion_summary, "训练完成总结")
    
        return {
            "model_path": final_model_dir,
            "eval_results": eval_results,
            "config": training_summary["config"]
        }
        
    except KeyboardInterrupt:
        logger.error("训练被用户中断")
        raise
    except Exception as e:
        logger.log_exception(e, "训练过程中")
        raise
    finally:
        # 恢复原始stdout并关闭日志器
        restore_print(original_stdout)
        logger.close()


if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    
    # 运行主训练函数
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        raise