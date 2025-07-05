"""
CLoRA模型评估脚本 - 评估训练好的CLoRA模型性能
集成了日志系统
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from peft import PeftModel
import os
import sys
import argparse
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional
import warnings
import json
from datetime import datetime

# 导入日志系统
from logger_utils import setup_logger, redirect_print_to_logger, restore_print

# 导入自定义模块
from data_processor import get_eval_datasets, load_tokenizer
from model_setup import setup_lora_model, load_base_model


def load_trained_model(checkpoint_path, base_model=None, logger=None):
    """
    加载训练好的CLoRA模型
    
    Args:
        checkpoint_path: 模型检查点路径
        base_model: 基础模型（可选）
        logger: 日志器
    
    Returns:
        tuple: (加载的模型, 分词器)
    """
    
    if logger:
        logger.info(f"正在从 {checkpoint_path} 加载训练好的模型...")
    else:
        print(f"正在从 {checkpoint_path} 加载训练好的模型...")
    
    # 如果没有提供基础模型，则加载一个新的
    if base_model is None:
        if logger:
            logger.info("加载基础模型...")
        else:
            print("加载基础模型...")
        base_model = load_base_model("gpt2", num_labels=2)
    
    # 加载PEFT模型
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    # 加载分词器
    tokenizer = load_tokenizer()
    
    if logger:
        logger.info("模型加载完成!")
    else:
        print("模型加载完成!")
    return model, tokenizer


def evaluate_model(model, tokenizer, eval_dataset, batch_size=16, device=None, logger=None):
    """
    评估模型性能
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        eval_dataset: 评估数据集
        batch_size: 批次大小
        device: 设备
        logger: 日志器
    
    Returns:
        dict: 评估结果字典
    """
    
    # 设置设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if logger:
        logger.info(f"使用设备: {device}")
    
    model.to(device)
    model.eval()
    
    # 创建数据加载器
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # 存储预测结果和真实标签
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    if logger:
        logger.info("开始评估...")
        logger.info(f"评估数据集大小: {len(eval_dataset)}")
        logger.info(f"批次大小: {batch_size}")
    else:
        print("开始评估...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估进度"):
            # 将数据移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            
            # 获取预测结果
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            # 累计损失
            if hasattr(outputs, 'loss'):
                total_loss += outputs.loss.item()
            num_batches += 1
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # 生成分类报告
    class_report = classification_report(
        all_labels, all_predictions,
        target_names=['False', 'True'],
        output_dict=True
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_loss': avg_loss,
        'classification_report': class_report,
        'predictions': all_predictions,
        'labels': all_labels,
        'num_samples': len(all_labels)
    }


def print_evaluation_results(results, logger=None):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
        logger: 日志器
    """
    
    if logger:
        logger.log_section("评估结果")
    else:
        print("\n" + "=" * 50)
        print("评估结果")
        print("=" * 50)
    
    # 主要指标
    main_metrics = {
        "准确率 (Accuracy)": f"{results['accuracy']:.4f}",
        "精确率 (Precision)": f"{results['precision']:.4f}",
        "召回率 (Recall)": f"{results['recall']:.4f}",
        "F1分数": f"{results['f1']:.4f}",
        "平均损失": f"{results['avg_loss']:.4f}"
    }
    
    if logger:
        logger.log_metrics(main_metrics, "主要指标")
    else:
        for metric, value in main_metrics.items():
            print(f"{metric}: {value}")
    
    # 详细分类报告
    class_report = results['classification_report']
    
    if logger:
        logger.log_section("详细分类报告", "-")
        
        # 每个类别的指标
        for class_name in ['False', 'True']:
            if class_name in class_report:
                metrics = class_report[class_name]
                class_metrics = {
                    "精确率": f"{metrics['precision']:.4f}",
                    "召回率": f"{metrics['recall']:.4f}",
                    "F1分数": f"{metrics['f1-score']:.4f}",
                    "支持数": str(metrics['support'])
                }
                logger.log_metrics(class_metrics, f"{class_name}类指标")
    else:
        print("\n" + "=" * 30)
        print("详细分类报告")
        print("=" * 30)
        
        # 打印每个类别的指标
        for class_name in ['False', 'True']:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"\n{class_name}类:")
                print(f"  精确率: {metrics['precision']:.4f}")
                print(f"  召回率: {metrics['recall']:.4f}")
                print(f"  F1分数: {metrics['f1-score']:.4f}")
                print(f"  支持数: {metrics['support']}")
    
    # 宏平均和加权平均
    if logger:
        if 'macro avg' in class_report:
            macro_avg = class_report['macro avg']
            macro_metrics = {
                "精确率": f"{macro_avg['precision']:.4f}",
                "召回率": f"{macro_avg['recall']:.4f}",
                "F1分数": f"{macro_avg['f1-score']:.4f}"
            }
            logger.log_metrics(macro_metrics, "宏平均")
        
        if 'weighted avg' in class_report:
            weighted_avg = class_report['weighted avg']
            weighted_metrics = {
                "精确率": f"{weighted_avg['precision']:.4f}",
                "召回率": f"{weighted_avg['recall']:.4f}",
                "F1分数": f"{weighted_avg['f1-score']:.4f}"
            }
            logger.log_metrics(weighted_metrics, "加权平均")
    else:
        if 'macro avg' in class_report:
            macro_avg = class_report['macro avg']
            print(f"\n宏平均:")
            print(f"  精确率: {macro_avg['precision']:.4f}")
            print(f"  召回率: {macro_avg['recall']:.4f}")
            print(f"  F1分数: {macro_avg['f1-score']:.4f}")
        
        if 'weighted avg' in class_report:
            weighted_avg = class_report['weighted avg']
            print(f"\n加权平均:")
            print(f"  精确率: {weighted_avg['precision']:.4f}")
            print(f"  召回率: {weighted_avg['recall']:.4f}")
            print(f"  F1分数: {weighted_avg['f1-score']:.4f}")


def compute_model_updating_capacity(model: torch.nn.Module, logger=None) -> float:
    """
    计算模型更新容量
    
    根据论文公式：对每个LoRA层计算 ΔW = B @ A，然后计算 ||ΔW||_2
    返回所有LoRA层的平均更新容量
    
    Args:
        model: PEFT模型或基线模型
        logger: 日志器
    
    Returns:
        float: 平均模型更新容量（基线模型返回0.0）
    """
    
    updating_capacities = []
    
    try:
        # 检查是否为基线模型（没有LoRA层）
        has_lora = False
        
        # 遍历所有模块寻找LoRA层
        for name, module in model.named_modules():
            # 查找LoRA的A和B矩阵
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                has_lora = True
                for adapter_name in module.lora_A.keys():
                    # 获取A和B矩阵
                    lora_a = module.lora_A[adapter_name].weight  # (r, input_dim)
                    lora_b = module.lora_B[adapter_name].weight  # (output_dim, r)
                    
                    # 计算 ΔW = B @ A
                    delta_w = torch.mm(lora_b, lora_a)  # (output_dim, input_dim)
                    
                    # 计算L2范数（最大奇异值）
                    l2_norm = torch.norm(delta_w, p=2).item()
                    updating_capacities.append(l2_norm)
        
        if not has_lora:
            msg = "基线模型（无LoRA层），更新容量为0"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return 0.0
                    
        if not updating_capacities:
            msg = "警告: 未找到任何LoRA层"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return 0.0
            
        # 返回平均更新容量
        avg_capacity = np.mean(updating_capacities)
        
        capacity_info = {
            "LoRA层数量": len(updating_capacities),
            "更新容量范围": f"{min(updating_capacities):.6f} - {max(updating_capacities):.6f}",
            "平均更新容量": f"{avg_capacity:.6f}"
        }
        
        if logger:
            logger.log_metrics(capacity_info, "模型更新容量")
        else:
            print(f"找到 {len(updating_capacities)} 个LoRA层")
            print(f"更新容量范围: {min(updating_capacities):.6f} - {max(updating_capacities):.6f}")
            print(f"平均更新容量: {avg_capacity:.6f}")
        
        return avg_capacity
        
    except Exception as e:
        error_msg = f"计算模型更新容量时出错: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return 0.0


def compute_relative_output_change(model: torch.nn.Module, 
                                 sample_inputs: List[torch.Tensor],
                                 device: Optional[torch.device] = None,
                                 logger=None) -> float:
    """
    计算相对输出变化
    
    实现论文中的公式：F_Δ(ΔW, x) = ||ΔW @ x|| / ||x||
    
    Args:
        model: PEFT模型或基线模型
        sample_inputs: 样本输入张量列表
        device: 计算设备
        logger: 日志器
    
    Returns:
        float: 平均相对输出变化（基线模型返回0.0）
    """
    
    if device is None:
        device = next(model.parameters()).device
    
    relative_changes = []
    
    try:
        model.eval()
        
        # 收集所有LoRA层的ΔW矩阵
        delta_w_matrices = []
        has_lora = False
        
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                has_lora = True
                for adapter_name in module.lora_A.keys():
                    lora_a = module.lora_A[adapter_name].weight
                    lora_b = module.lora_B[adapter_name].weight
                    delta_w = torch.mm(lora_b, lora_a)
                    delta_w_matrices.append(delta_w)
        
        if not has_lora:
            msg = "基线模型（无LoRA层），相对输出变化为0"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return 0.0
        
        if not delta_w_matrices:
            msg = "警告: 未找到任何LoRA层用于计算相对输出变化"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return 0.0
        
        # 对每个样本输入计算相对输出变化
        with torch.no_grad():
            for input_tensor in sample_inputs[:100]:  # 限制样本数量以提高效率
                input_tensor = input_tensor.to(device)
                
                # 确保输入是2D张量
                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(0)
                
                input_norm = torch.norm(input_tensor, p=2)
                
                if input_norm > 1e-8:  # 避免除零
                    layer_changes = []
                    
                    for delta_w in delta_w_matrices:
                        # 调整维度以确保矩阵乘法可行
                        if delta_w.shape[1] == input_tensor.shape[-1]:
                            output_change = torch.mm(delta_w, input_tensor.T)
                            output_change_norm = torch.norm(output_change, p=2)
                            relative_change = (output_change_norm / input_norm).item()
                            layer_changes.append(relative_change)
                    
                    if layer_changes:
                        # 对所有层的相对变化取平均
                        avg_change = np.mean(layer_changes)
                        relative_changes.append(avg_change)
        
        if not relative_changes:
            msg = "警告: 无法计算相对输出变化"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return 0.0
        
        avg_relative_change = np.mean(relative_changes)
        
        change_info = {
            "样本数量": len(relative_changes),
            "平均相对输出变化": f"{avg_relative_change:.6f}"
        }
        
        if logger:
            logger.log_metrics(change_info, "相对输出变化")
        else:
            print(f"计算了 {len(relative_changes)} 个样本的相对输出变化")
            print(f"平均相对输出变化: {avg_relative_change:.6f}")
        
        return avg_relative_change
        
    except Exception as e:
        error_msg = f"计算相对输出变化时出错: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        return 0.0


def extract_sample_inputs(eval_dataset, tokenizer, num_samples: int = 50) -> List[torch.Tensor]:
    """
    从评估数据集中提取样本输入
    
    Args:
        eval_dataset: 评估数据集
        tokenizer: 分词器
        num_samples: 提取的样本数量
    
    Returns:
        List[torch.Tensor]: 样本输入张量列表
    """
    
    sample_inputs = []
    
    try:
        # 随机选择样本
        indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))
        
        for idx in indices:
            sample = eval_dataset[idx]
            input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
            sample_inputs.append(input_ids)
            
    except Exception as e:
        print(f"提取样本输入时出错: {e}")
        
    return sample_inputs


def calculate_task_accuracies(all_results: Dict[str, Dict]) -> Tuple[float, float, Dict[str, float]]:
    """
    计算任务准确率
    
    Args:
        all_results: 所有数据集的评估结果
    
    Returns:
        tuple: (域内平均准确率, 域外平均准确率, 详细准确率字典)
    """
    
    # 定义域内和域外数据集
    # 域内：CommonsenseQA (训练数据集同类型)
    in_domain_datasets = ['commonsense_qa']
    # 域外：其他所有数据集
    out_domain_datasets = ['boolq', 'piqa', 'winogrande', 'hellaswag']
    
    in_domain_accuracies = []
    out_domain_accuracies = []
    detailed_accuracies = {}
    
    for dataset_name, results in all_results.items():
        accuracy = results['accuracy']
        detailed_accuracies[dataset_name] = accuracy
        
        if dataset_name in in_domain_datasets:
            in_domain_accuracies.append(accuracy)
        elif dataset_name in out_domain_datasets:
            out_domain_accuracies.append(accuracy)
        # 其他未定义的数据集默认为域外
        else:
            out_domain_accuracies.append(accuracy)
    
    # 计算平均值
    avg_in_domain = np.mean(in_domain_accuracies) if in_domain_accuracies else 0.0
    avg_out_domain = np.mean(out_domain_accuracies) if out_domain_accuracies else 0.0
    
    return avg_in_domain, avg_out_domain, detailed_accuracies


def generate_evaluation_report(model: torch.nn.Module, 
                             tokenizer, 
                             eval_datasets: Dict[str, any],
                             batch_size: int = 16,
                             logger=None) -> Dict[str, any]:
    """
    生成综合评估报告
    
    Args:
        model: CLoRA模型
        tokenizer: 分词器
        eval_datasets: 评估数据集字典
        batch_size: 评估批次大小
        logger: 日志器
    
    Returns:
        dict: 综合评估报告
    """
    
    if logger:
        logger.log_section("综合评估报告生成")
    else:
        print("\n" + "=" * 40)
        print("综合评估报告生成")
        print("=" * 40)
    
    # 评估所有数据集
    all_results = {}
    for dataset_name, dataset in eval_datasets.items():
        if logger:
            logger.info(f"评估数据集: {dataset_name}")
        else:
            print(f"\n评估数据集: {dataset_name}")
        
        results = evaluate_model(model, tokenizer, dataset, batch_size=batch_size, logger=logger)
        all_results[dataset_name] = results
        
        if logger:
            metrics = {
                "准确率": f"{results['accuracy']:.4f}",
                "F1分数": f"{results['f1']:.4f}",
                "样本数": results['num_samples']
            }
            logger.log_metrics(metrics, f"{dataset_name} 结果")
    
    # 计算任务准确率
    avg_in_domain, avg_out_domain, detailed_accuracies = calculate_task_accuracies(all_results)
    
    # 计算模型更新容量
    if logger:
        logger.info("计算模型更新容量...")
    model_updating_capacity = compute_model_updating_capacity(model, logger=logger)
    
    # 获取样本输入用于相对输出变化计算
    sample_inputs = []
    for dataset_name, dataset in eval_datasets.items():
        inputs = extract_sample_inputs(dataset, tokenizer, num_samples=10)
        sample_inputs.extend(inputs)
        if len(sample_inputs) >= 50:  # 限制总数
            break
    
    # 计算相对输出变化
    if logger:
        logger.info("计算相对输出变化...")
    relative_output_change = compute_relative_output_change(model, sample_inputs, logger=logger)
    
    # 生成报告
    evaluation_report = {
        'task_accuracies': detailed_accuracies,
        'avg_in_domain_accuracy': avg_in_domain,
        'avg_out_domain_accuracy': avg_out_domain,
        'model_updating_capacity': model_updating_capacity,
        'relative_output_change': relative_output_change,
        'detailed_results': all_results
    }
    
    # 记录综合结果
    if logger:
        summary_metrics = {
            "域内平均准确率": f"{avg_in_domain:.4f}",
            "域外平均准确率": f"{avg_out_domain:.4f}",
            "模型更新容量": f"{model_updating_capacity:.6f}",
            "相对输出变化": f"{relative_output_change:.6f}"
        }
        logger.log_results(summary_metrics, "综合评估结果")
    
    return evaluation_report


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CLoRA模型评估脚本")
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="指定训练好的CLoRA模型路径（PEFT适配器目录）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="评估时的批次大小 (默认: 16)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="每个数据集的最大样本数 (默认: 2000)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help="评估结果输出目录 (默认: ./)"
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="使用原始GPT2模型进行基线测试（不使用LoRA微调）"
    )
    
    return parser.parse_args()


def main():
    """主评估函数"""
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志系统
    log_dir = "./logs"
    logger = setup_logger(
        name="CLoRA_evaluation",
        log_dir=log_dir,
        console_output=True,
        command_args=sys.argv,
        script_name="evaluate_clora.py"
    )
    
    # 重定向print输出到日志（保持控制台输出）
    original_stdout = redirect_print_to_logger(logger, also_print=False)
    
    try:
        logger.log_section("CLoRA 模型评估")
        
        # 1. 处理数据集
        logger.log_step(1, "加载评估数据集")
        
        # 加载分词器
        from data_processor import load_tokenizer, get_in_domain_eval_dataset
        tokenizer = load_tokenizer()
        
        # 获取域内评估数据集（CommonsenseQA验证集）
        in_domain_dataset = get_in_domain_eval_dataset(tokenizer, max_samples=args.max_samples)
        
        # 获取域外评估数据集
        out_domain_datasets = get_eval_datasets(tokenizer, max_samples_per_dataset=args.max_samples)
        
        # 合并所有评估数据集
        eval_datasets = {"commonsense_qa": in_domain_dataset}
        eval_datasets.update(out_domain_datasets)
        
        dataset_info = {
            "数据集数量": len(eval_datasets)
        }
        for name, dataset in eval_datasets.items():
            dataset_info[f"{name}数据集"] = f"{len(dataset)} 样本"
        
        logger.log_config(dataset_info, "评估数据集信息")
        
        # 2. 加载模型
        if args.baseline:
            logger.log_step(2, "加载基线模型（原始GPT2）")
            
            # 使用原始GPT2模型进行基线测试
            from model_setup import load_base_model
            model = load_base_model("gpt2", num_labels=2)
            logger.info("已加载原始GPT2模型作为基线")
            checkpoint_path = "baseline_gpt2"
            
        else:
            logger.log_step(2, "确定CLoRA模型路径")
            
            checkpoint_path = None
            
            # 优先使用用户指定的模型路径
            if args.model_path:
                if os.path.exists(args.model_path):
                    # 验证路径是否包含有效的PEFT模型文件
                    if os.path.isdir(args.model_path):
                        required_files = ['adapter_config.json']
                        model_files = [f for f in os.listdir(args.model_path) 
                                     if f.endswith('.bin') or f.endswith('.safetensors')]
                        
                        if any(f in os.listdir(args.model_path) for f in required_files) and model_files:
                            checkpoint_path = args.model_path
                            logger.info(f"使用用户指定的模型路径: {checkpoint_path}")
                        else:
                            logger.error(f"指定路径 {args.model_path} 不包含有效的PEFT模型文件")
                            logger.info("需要包含 adapter_config.json 和模型权重文件 (.bin 或 .safetensors)")
                            return
                    else:
                        logger.error(f"指定的模型路径不是目录: {args.model_path}")
                        return
                else:
                    logger.error(f"指定的模型路径不存在: {args.model_path}")
                    return
        
            # 如果用户没有指定路径，则搜索默认位置
            if checkpoint_path is None:
                logger.info("未指定模型路径，搜索默认位置...")
                
                # 可能的模型路径
                possible_paths = [
                    "./clora_final_model",
                    "./clora_results", 
                    "./results",
                    "./final_model",
                    "./"
                ]
                
                # 查找检查点目录
                checkpoint_dirs = []
                for path in possible_paths:
                    if os.path.exists(path):
                        if os.path.isdir(path):
                            # 检查是否包含模型文件
                            if any(f.endswith('.bin') or f.endswith('.safetensors') or f == 'adapter_config.json' 
                                   for f in os.listdir(path)):
                                checkpoint_dirs.append(path)
                            else:
                                # 查找子目录中的检查点
                                for subdir in os.listdir(path):
                                    subpath = os.path.join(path, subdir)
                                    if os.path.isdir(subpath) and 'checkpoint' in subdir:
                                        checkpoint_dirs.append(subpath)
                
                if not checkpoint_dirs:
                    error_msg = "未找到训练好的模型检查点"
                    logger.error(error_msg)
                    logger.info("请执行以下操作之一:")
                    logger.info("1. 运行 train_clora.py 训练模型")
                    logger.info("2. 使用 --model-path 参数指定模型路径")
                    logger.info("3. 使用 --baseline 参数进行基线测试")
                    logger.info("示例: python evaluate_clora.py --model-path ./path/to/your/model")
                    return
                
                # 使用找到的第一个检查点
                checkpoint_path = checkpoint_dirs[0]
                logger.info(f"找到模型检查点: {checkpoint_path}")
            
            # 3. 加载训练好的CLoRA模型
            logger.log_step(3, "加载训练好的CLoRA模型")
            
            try:
                model, tokenizer = load_trained_model(checkpoint_path, logger=logger)
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                logger.info("尝试设置基础模型并重新加载...")
                from model_setup import setup_lora_model
                base_model = setup_lora_model()
                model, tokenizer = load_trained_model(checkpoint_path, base_model.base_model, logger=logger)
        
        # 4. 使用综合评估功能
        logger.log_step(4, "开始CLoRA综合评估")
        
        # 调用新的综合评估报告生成函数
        evaluation_report = generate_evaluation_report(model, tokenizer, eval_datasets, batch_size=args.batch_size, logger=logger)
        
        # 保存评估结果到文件（可选）
        logger.log_step(5, "保存评估结果")
        
        
        
        # 根据模型类型选择报告文件名
        if args.baseline:
            # report_file = os.path.join(args.model_path, args.output_dir, "baseline_evaluation_report.json")
            report_file = os.path.normpath(os.path.join(args.model_path, args.output_dir, "baseline_evaluation_report.json"))
            report_file = f"./{report_file}"
        else:
            report_file = os.path.normpath(os.path.join(args.model_path, args.output_dir, "clora_evaluation_report.json"))
            report_file = f"./{report_file}"
            # report_file = os.path.join(args.model_path, args.output_dir, "clora_evaluation_report.json")
        try:
            # 确保输出目录存在
            os.makedirs(report_file, exist_ok=True)
            # 将报告保存为JSON文件
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'task_accuracies': evaluation_report['task_accuracies'],
                    'avg_in_domain_accuracy': evaluation_report['avg_in_domain_accuracy'],
                    'avg_out_domain_accuracy': evaluation_report['avg_out_domain_accuracy'],
                    'model_updating_capacity': evaluation_report['model_updating_capacity'],
                    'relative_output_change': evaluation_report['relative_output_change'],
                    'timestamp': datetime.now().isoformat(),
                    'checkpoint_path': checkpoint_path
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"评估报告已保存到: {report_file}")
        except Exception as e:
            logger.error(f"保存评估报告失败: {e}")
        
        # 记录评估完成
        logger.log_section("评估完成", "🎉")
        
        completion_info = {
            "评估报告文件": report_file,
            "日志文件": logger.get_log_file_path(),
            "模型检查点": checkpoint_path
        }
        logger.log_config(completion_info, "评估完成总结")
        
        return evaluation_report
        
    except KeyboardInterrupt:
        logger.error("评估被用户中断")
        raise
    except Exception as e:
        logger.log_exception(e, "评估过程中")
        raise
    finally:
        # 恢复原始stdout并关闭日志器
        restore_print(original_stdout)
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n评估被用户中断")
    except Exception as e:
        print(f"\n评估过程中出现错误: {e}")
        raise