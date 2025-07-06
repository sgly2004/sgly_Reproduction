"""
梯度处理工具模块

本模块提供对LoRA模型参数梯度进行处理的工具函数，主要用于梯度投影分析。
"""

import torch


def get_projected_gradients(model, n_components=1, device='cpu'):
    """
    对模型参数的梯度进行处理和投影分析
    
    该函数遍历模型中的所有LoRA层，提取lora_A和lora_B参数的梯度，
    并使用SVD分解对梯度矩阵进行分析处理。
    
    Args:
        model: 待分析的PEFT模型，包含LoRA层
        n_components (int): SVD分解中要保留的主成分数量，默认为1
        device (str): 计算设备，默认为'cpu'
        
    Returns:
        dict: 包含所有投影梯度的字典，格式为:
              {
                  'layer_name.lora_A': {
                      'gradient': 梯度矩阵,
                      'U': SVD分解的U矩阵,
                      'S': SVD分解的奇异值,
                      'V': SVD分解的V矩阵,
                      'projected': 投影后的梯度
                  },
                  'layer_name.lora_B': { ... }
              }
    """
    # 初始化存储投影梯度的字典
    projected_gradients = {}
    
    # 遍历模型的所有模块，同时获取模块名称和模块对象
    for name, module in model.named_modules():
        # 检查当前模块是否为LoRA层
        # LoRA层的特征是同时具有lora_A和lora_B属性
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            
            # 处理lora_A参数的梯度
            if 'default' in module.lora_A:
                # 获取lora_A的权重梯度
                lora_a_grad = module.lora_A['default'].weight.grad
                
                # 检查梯度是否存在，如果不存在则跳过
                if lora_a_grad is not None:
                    # 将梯度移动到指定设备
                    lora_a_grad = lora_a_grad.to(device)
                    
                    # 使用SVD分解对梯度矩阵进行分析
                    # lora_a_grad是一个二维矩阵，形状为(rank, input_dim)
                    U, S, V = torch.linalg.svd(lora_a_grad, full_matrices=False)
                    
                    # 根据n_components截取主要成分
                    U_reduced = U[:, :n_components]
                    S_reduced = S[:n_components]
                    V_reduced = V[:n_components, :]
                    
                    # 重构投影后的梯度矩阵
                    projected_grad = U_reduced @ torch.diag(S_reduced) @ V_reduced
                    
                    # 存储分析结果
                    projected_gradients[f"{name}.lora_A"] = {
                        'gradient': lora_a_grad,
                        'U': U,
                        'S': S,
                        'V': V,
                        'projected': projected_grad
                    }
            
            # 处理lora_B参数的梯度
            if 'default' in module.lora_B:
                # 获取lora_B的权重梯度
                lora_b_grad = module.lora_B['default'].weight.grad
                
                # 检查梯度是否存在，如果不存在则跳过
                if lora_b_grad is not None:
                    # 将梯度移动到指定设备
                    lora_b_grad = lora_b_grad.to(device)
                    
                    # 使用SVD分解对梯度矩阵进行分析
                    # lora_b_grad是一个二维矩阵，形状为(output_dim, rank)
                    U, S, V = torch.linalg.svd(lora_b_grad, full_matrices=False)
                    
                    # 根据n_components截取主要成分
                    U_reduced = U[:, :n_components]
                    S_reduced = S[:n_components]
                    V_reduced = V[:n_components, :]
                    
                    # 重构投影后的梯度矩阵
                    projected_grad = U_reduced @ torch.diag(S_reduced) @ V_reduced
                    
                    # 存储分析结果
                    projected_gradients[f"{name}.lora_B"] = {
                        'gradient': lora_b_grad,
                        'U': U,
                        'S': S,
                        'V': V,
                        'projected': projected_grad
                    }
    
    # 返回包含所有投影梯度的字典
    return projected_gradients


def analyze_gradient_distribution(projected_gradients):
    """
    分析投影梯度的分布特性
    
    Args:
        projected_gradients (dict): get_projected_gradients函数的输出结果
        
    Returns:
        dict: 梯度分布分析结果
    """
    analysis_results = {}
    
    for layer_name, grad_info in projected_gradients.items():
        gradient = grad_info['gradient']
        singular_values = grad_info['S']
        
        # 计算梯度的基本统计信息
        analysis_results[layer_name] = {
            'gradient_norm': torch.norm(gradient).item(),
            'gradient_mean': torch.mean(gradient).item(),
            'gradient_std': torch.std(gradient).item(),
            'singular_values': singular_values.cpu().numpy(),
            'rank': len(singular_values),
            'effective_rank': (singular_values.sum() / singular_values.max()).item()
        }
    
    return analysis_results