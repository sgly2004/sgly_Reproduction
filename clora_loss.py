"""
CLoRA核心模块 - 正交正则化损失函数
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_orthogonal_loss(lora_a, lora_b, regularization_matrices):
    """
    计算CLoRA的正交正则化损失
    
    Args:
        lora_a: LoRA的A矩阵 (torch.Tensor) - 形状: (input_dim, r)
        lora_b: LoRA的B矩阵 (torch.Tensor) - 形状: (r, output_dim)
        regularization_matrices: 包含正则化矩阵的字典
            - 'P_A': A矩阵的正则化矩阵 (r, r)
            - 'P_B': B矩阵的正则化矩阵 (r, r)
    
    Returns:
        torch.Tensor: 总的正交损失 (Loss_A + Loss_B)
    """
    
    P_A = regularization_matrices['P_A']
    P_B = regularization_matrices['P_B']
    
    # 确保矩阵在同一设备上
    device = lora_a.device
    P_A = P_A.to(device)
    P_B = P_B.to(device)
    
    # 计算 Loss_A = ||lora_a @ P_A.T||_F^2
    # lora_a: (input_dim, r), P_A.T: (r, r) -> result: (input_dim, r)
    loss_a_matrix = torch.mm(lora_a, P_A.T)
    loss_a = torch.norm(loss_a_matrix, 'fro') ** 2
    
    # 计算 Loss_B = ||lora_b.T @ P_B.T||_F^2
    # lora_b.T: (output_dim, r), P_B.T: (r, r) -> result: (output_dim, r)
    loss_b_matrix = torch.mm(lora_b.T, P_B.T)
    loss_b = torch.norm(loss_b_matrix, 'fro') ** 2
    
    # 返回总损失
    total_loss = loss_a + loss_b
    
    return total_loss


def generate_regularization_matrices(rank, method='random'):
    """
    生成正则化矩阵P_A和P_B
    
    Args:
        rank: LoRA的秩
        method: 生成方法 ('random', 'orthogonal', 'identity')
    
    Returns:
        dict: 包含P_A和P_B的字典
    """
    
    if method == 'random':
        # 随机生成正则化矩阵
        P_A = torch.randn(rank, rank)
        P_B = torch.randn(rank, rank)
        
    elif method == 'orthogonal':
        # 生成正交矩阵
        # 使用QR分解来获得正交矩阵
        Q_A, _ = torch.linalg.qr(torch.randn(rank, rank))
        Q_B, _ = torch.linalg.qr(torch.randn(rank, rank))
        P_A = Q_A
        P_B = Q_B
        
    elif method == 'identity':
        # 使用单位矩阵（用于测试）
        P_A = torch.eye(rank)
        P_B = torch.eye(rank)
        
    else:
        raise ValueError(f"不支持的生成方法: {method}")
    
    return {'P_A': P_A, 'P_B': P_B}


def create_sample_lora_matrices(input_dim, output_dim, rank):
    """
    创建示例LoRA矩阵
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        rank: LoRA秩
    
    Returns:
        tuple: (lora_a, lora_b)
    """
    
    # 按照LoRA的初始化方法
    # A矩阵使用正态分布初始化
    lora_a = torch.randn(input_dim, rank) * 0.01
    
    # B矩阵初始化为零
    lora_b = torch.zeros(rank, output_dim)
    
    return lora_a, lora_b


def test_orthogonal_loss_properties():
    """
    测试正交损失的性质
    """
    print("\n" + "=" * 40)
    print("正交损失性质测试")
    print("=" * 40)
    
    rank = 8
    input_dim = 100
    output_dim = 200
    
    # 创建测试矩阵
    lora_a, lora_b = create_sample_lora_matrices(input_dim, output_dim, rank)
    
    # 测试不同的正则化矩阵
    methods = ['random', 'orthogonal', 'identity']
    
    for method in methods:
        reg_matrices = generate_regularization_matrices(rank, method)
        loss = compute_orthogonal_loss(lora_a, lora_b, reg_matrices)
        print(f"{method.capitalize()} 正则化矩阵 - 损失值: {loss.item():.6f}")
    
    # 测试零矩阵的情况
    zero_a = torch.zeros_like(lora_a)
    zero_b = torch.zeros_like(lora_b)
    reg_matrices = generate_regularization_matrices(rank, 'random')
    zero_loss = compute_orthogonal_loss(zero_a, zero_b, reg_matrices)
    print(f"零矩阵损失: {zero_loss.item():.6f}")


def main():
    """测试函数"""
    print("=" * 50)
    print("CLoRA正交正则化损失测试")
    print("=" * 50)
    
    # 设置参数
    rank = 8
    input_dim = 100
    output_dim = 200
    
    print(f"LoRA参数:")
    print(f"  秩 (r): {rank}")
    print(f"  输入维度: {input_dim}")
    print(f"  输出维度: {output_dim}")
    
    # 1. 创建示例LoRA矩阵
    print(f"\n正在创建LoRA矩阵...")
    lora_a, lora_b = create_sample_lora_matrices(input_dim, output_dim, rank)
    
    print(f"LoRA A矩阵形状: {lora_a.shape}")
    print(f"LoRA B矩阵形状: {lora_b.shape}")
    print(f"LoRA A矩阵范数: {torch.norm(lora_a).item():.6f}")
    print(f"LoRA B矩阵范数: {torch.norm(lora_b).item():.6f}")
    
    # 2. 生成正则化矩阵
    print(f"\n正在生成正则化矩阵...")
    regularization_matrices = generate_regularization_matrices(rank, method='random')
    
    P_A = regularization_matrices['P_A']
    P_B = regularization_matrices['P_B']
    
    print(f"P_A矩阵形状: {P_A.shape}")
    print(f"P_B矩阵形状: {P_B.shape}")
    print(f"P_A矩阵范数: {torch.norm(P_A).item():.6f}")
    print(f"P_B矩阵范数: {torch.norm(P_B).item():.6f}")
    
    # 3. 计算正交损失
    print(f"\n正在计算正交损失...")
    orthogonal_loss = compute_orthogonal_loss(lora_a, lora_b, regularization_matrices)
    
    print(f"正交正则化损失: {orthogonal_loss.item():.6f}")
    
    # 4. 分别计算Loss_A和Loss_B
    loss_a_matrix = torch.mm(lora_a, P_A.T)
    loss_a = torch.norm(loss_a_matrix, 'fro') ** 2
    
    loss_b_matrix = torch.mm(lora_b.T, P_B.T)
    loss_b = torch.norm(loss_b_matrix, 'fro') ** 2
    
    print(f"Loss_A: {loss_a.item():.6f}")
    print(f"Loss_B: {loss_b.item():.6f}")
    print(f"总损失 (Loss_A + Loss_B): {(loss_a + loss_b).item():.6f}")
    
    # 5. 测试损失的性质
    test_orthogonal_loss_properties()
    
    print(f"\n正交损失计算测试完成!")


if __name__ == "__main__":
    main()