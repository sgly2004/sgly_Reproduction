#!/bin/bash

# CLoRA自动化实验脚本（增强版）
# 用于探索不同k值对CLoRA性能的影响
# 自动捕获模型路径进行后续评估

echo "========================================"
echo "开始CLoRA自动化实验（增强版）"
echo "========================================"

# 创建输出目录
mkdir -p ./output
mkdir -p ./eval_logs

# 定义k值数组
k_values=(1 4 8 16 32 64 128 256 512)

echo "将要测试的k值: ${k_values[@]}"
echo ""

# 用于存储所有模型路径和实验信息的数组
declare -a experiments

# 函数：从训练输出中提取模型路径
extract_model_path() {
    local log_file="$1"
    local model_path=$(grep "MODEL_PATH:" "$log_file" | tail -1 | cut -d' ' -f2-)
    echo "$model_path"
}

# 函数：运行训练并记录结果
run_training() {
    local experiment_name="$1"
    local output_dir="$2"
    shift 2
    local train_cmd=("$@")
    
    echo "----------------------------------------"
    echo "运行实验: $experiment_name"
    echo "----------------------------------------"
    
    mkdir -p "$output_dir"
    local log_file="${output_dir}/training_log.txt"
    
    echo "输出目录: $output_dir"
    echo "训练命令: ${train_cmd[@]}"
    echo "开始训练..."
    
    # 运行训练命令
    "${train_cmd[@]}" > "$log_file" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # 提取模型路径
        local model_path=$(extract_model_path "$log_file")
        if [ -n "$model_path" ] && [ -d "$model_path" ]; then
            echo "✓ $experiment_name 训练完成"
            echo "  模型路径: $model_path"
            experiments+=("$experiment_name|$model_path")
        else
            echo "✓ $experiment_name 训练完成，但无法提取模型路径"
            echo "  查看日志: $log_file"
            # 假设模型在标准位置
            local fallback_path="${output_dir}/final_model"
            if [ -d "$fallback_path" ]; then
                experiments+=("$experiment_name|$fallback_path")
            fi
        fi
    else
        echo "✗ $experiment_name 训练失败"
        echo "  查看日志: $log_file"
    fi
    
    echo ""
}

# ========================================
# 第一部分：不同k值的CLoRA实验
# ========================================
echo "第一部分：开始CLoRA k值实验..."
echo ""

for k in "${k_values[@]}"; do
    experiment_name="CLoRA_k${k}"
    output_dir="./output/clora_k_${k}"
    
    run_training "$experiment_name" "$output_dir" \
        python train_clora.py \
        --use_clora \
        --clora_k "$k" \
        --lambda_param 0.1 \
        --num_epochs 20 \
        --batch_size 32 \
        --learning_rate 2e-4 \
        --output_dir "$output_dir"
done

echo "CLoRA k值实验完成！"
echo ""

# ========================================
# 第二部分：基准实验（标准LoRA）
# ========================================
echo "第二部分：开始基准实验..."
echo ""

experiment_name="Baseline_LoRA"
output_dir="./output/baseline_lora"

run_training "$experiment_name" "$output_dir" \
    python train_clora.py \
    --no_clora \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --output_dir "$output_dir"

# ========================================
# 第三部分：PCA梯度实验
# ========================================
echo "第三部分：开始PCA梯度实验..."
echo ""

experiment_name="PCA_Grad_LoRA"
output_dir="./output/pca_grad_lora"

run_training "$experiment_name" "$output_dir" \
    python train_clora.py \
    --use_pca_grad \
    --pca_components 2 \
    --no_clora \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --output_dir "$output_dir"

# ========================================
# 第四部分：模型评估
# ========================================
echo "第四部分：开始模型评估..."
echo ""

eval_counter=1
for experiment_info in "${experiments[@]}"; do
    # 分割实验名称和模型路径
    IFS='|' read -r experiment_name model_path <<< "$experiment_info"
    
    if [ -d "$model_path" ]; then
        echo "----------------------------------------"
        echo "评估模型 ${eval_counter}/${#experiments[@]}: $experiment_name"
        echo "----------------------------------------"
        
        eval_output_dir="./eval_logs/eval_${experiment_name}"
        mkdir -p "$eval_output_dir"
        
        echo "模型路径: $model_path"
        echo "评估输出目录: $eval_output_dir"
        
        python evaluate_clora.py \
            --model-path "$model_path" \
            --output-dir "$eval_output_dir" \
            > "${eval_output_dir}/evaluation_log.txt" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "successfully！ 模型评估完成: $experiment_name"
        else
            echo "error！ 模型评估失败: $experiment_name"
            echo "  查看日志: ${eval_output_dir}/evaluation_log.txt"
        fi
        
        echo ""
    else
        echo "模型路径不存在: $model_path (实验: $experiment_name)"
    fi
    
    ((eval_counter++))
done

# ========================================
# 生成实验报告
# ========================================
echo "========================================"
echo "生成实验报告..."
echo "========================================"

report_file="./output/experiment_report.txt"
{
    echo "CLoRA自动化实验报告"
    echo "生成时间: $(date)"
    echo ""
    echo "实验配置："
    echo "- k值范围: ${k_values[@]}"
    echo "- 基准实验: 标准LoRA"
    echo "- 特殊实验: PCA梯度LoRA"
    echo ""
    echo "实验结果："
    for experiment_info in "${experiments[@]}"; do
        IFS='|' read -r experiment_name model_path <<< "$experiment_info"
        echo "- $experiment_name: $model_path"
    done
    echo ""
    echo "评估结果位置: ./eval_logs/"
} > "$report_file"

echo "实验报告已生成: $report_file"
echo ""

# ========================================
# 生成综合总结报告
# ========================================
echo "========================================"
echo "生成综合总结报告..."
echo "========================================"

# 创建实验报告目录
mkdir -p ./experiment_report

# 生成时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
comprehensive_report="./experiment_report/comprehensive_report_${timestamp}.txt"

{
    echo "========================================"
    echo "CLoRA自动化实验综合报告"
    echo "========================================"
    echo "生成时间: $(date)"
    echo "脚本版本: Enhanced Version"
    echo ""
    
    # 实验配置概览
    echo "========================================"
    echo "实验配置概览"
    echo "========================================"
    echo "k值范围: ${k_values[@]}"
    echo "总实验数量: $((${#k_values[@]} + 2))"
    echo "成功完成的实验: ${#experiments[@]}"
    echo ""
    
    # 所有实验的训练总结
    echo "========================================"
    echo "训练总结报告"
    echo "========================================"
    echo ""
    
    for experiment_info in "${experiments[@]}"; do
        IFS='|' read -r experiment_name model_path <<< "$experiment_info"
        
        echo "----------------------------------------"
        echo "实验: $experiment_name"
        echo "----------------------------------------"
        echo "模型路径: $model_path"
        
        # 查找对应的训练总结文件
        output_dir=$(dirname "$model_path")
        summary_file="${output_dir}/training_summary.json"
        
        if [ -f "$summary_file" ]; then
            echo ""
            echo "=== 训练总结文件内容 ==="
            cat "$summary_file"
            echo ""
            echo "=== 训练总结文件结束 ==="
        else
            echo "训练总结文件未找到: $summary_file"
        fi
        
        echo ""
        echo ""
    done
    
    # 所有实验的评估报告
    echo "========================================"
    echo "评估报告汇总"
    echo "========================================"
    echo ""
    
    for experiment_info in "${experiments[@]}"; do
        IFS='|' read -r experiment_name model_path <<< "$experiment_info"
        
        echo "----------------------------------------"
        echo "实验评估: $experiment_name"
        echo "----------------------------------------"
        
        # 查找对应的评估报告文件
        eval_output_dir="./eval_logs/eval_${experiment_name}"
        eval_report_file="${eval_output_dir}/evaluation_report.json"
        
        if [ -f "$eval_report_file" ]; then
            echo ""
            echo "=== 评估报告文件内容 ==="
            cat "$eval_report_file"
            echo ""
            echo "=== 评估报告文件结束 ==="
        else
            echo "评估报告文件未找到: $eval_report_file"
            
            # 尝试查找其他可能的评估文件
            if [ -d "$eval_output_dir" ]; then
                echo "评估目录内容:"
                ls -la "$eval_output_dir"
                
                # 查找任何JSON文件
                json_files=$(find "$eval_output_dir" -name "*.json" 2>/dev/null)
                if [ -n "$json_files" ]; then
                    echo ""
                    echo "找到的JSON文件:"
                    for json_file in $json_files; do
                        echo "=== $(basename $json_file) ==="
                        cat "$json_file"
                        echo ""
                    done
                fi
            fi
        fi
        
        echo ""
        echo ""
    done
    
    # 添加基础实验报告内容
    echo "========================================"
    echo "基础实验报告"
    echo "========================================"
    if [ -f "$report_file" ]; then
        cat "$report_file"
    else
        echo "基础实验报告文件未找到: $report_file"
    fi
    echo ""
    
    # 实验结论和建议
    echo "========================================"
    echo "实验结论和建议"
    echo "========================================"
    echo "1. 总共尝试了 ${#k_values[@]} 个不同的k值进行CLoRA实验"
    echo "2. 成功完成了 ${#experiments[@]} 个实验的训练和评估"
    echo "3. 所有实验结果已保存在对应的目录中"
    echo "4. 建议对比不同k值下的性能指标来选择最优配置"
    echo ""
    echo "文件位置说明:"
    echo "- 训练结果: ./output/"
    echo "- 评估结果: ./eval_logs/"
    echo "- 综合报告: $comprehensive_report"
    echo ""
    echo "========================================"
    echo "报告生成完毕"
    echo "========================================"
    
} > "$comprehensive_report"

echo "✓ 综合总结报告已生成: $comprehensive_report"
echo ""

# ========================================
# 实验总结
# ========================================
echo "========================================"
echo "所有实验完成！"
echo "========================================"

echo ""
echo "实验总结："
echo "- CLoRA k值实验: ${#k_values[@]} 个"
echo "- 基准LoRA实验: 1 个"
echo "- PCA梯度实验: 1 个"
echo "- 成功训练的模型: ${#experiments[@]} 个"
echo ""

echo "结果文件位置："
echo "- 训练结果: ./output/"
echo "- 评估结果: ./eval_logs/"
echo "- 基础实验报告: $report_file"
echo "- 综合总结报告: $comprehensive_report"
echo ""

echo "查看实验结果的建议命令："
echo "cat $comprehensive_report"
echo "cat $report_file"
echo "ls -la ./output/"
echo "ls -la ./eval_logs/"
echo "ls -la ./experiment_report/"
echo ""

echo "实验脚本执行完毕！"