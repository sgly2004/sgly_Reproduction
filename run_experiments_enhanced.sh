#!/bin/bash

# CLoRA自动化实验脚本（并行训练增强版）
# 用于探索不同k值对CLoRA性能的影响
# 支持并行训练和实时日志输出

echo "========================================"
echo "开始CLoRA自动化实验（并行训练版）"
echo "========================================"

# ========================================
# 配置参数
# ========================================
# 定义k值数组
k_values=(1 4 8 16 32 64 128 256 512)

# 并行训练配置
MAX_PARALLEL_JOBS=2  # 最大并行任务数，根据GPU显存调整
MONITOR_INTERVAL=5   # 监控间隔（秒）
ENABLE_REAL_TIME_LOG=true  # 是否启用实时日志输出

# GPU显存检查阈值（MB）
GPU_MEMORY_THRESHOLD=10000  # 单个任务预计使用的显存

echo "配置参数："
echo "- 最大并行任务数: $MAX_PARALLEL_JOBS"
echo "- 监控间隔: ${MONITOR_INTERVAL}秒"
echo "- 实时日志输出: $ENABLE_REAL_TIME_LOG"
echo "- 将要测试的k值: ${k_values[@]}"
echo ""

# ========================================
# 工具函数
# ========================================

# 创建输出目录
mkdir -p ./output
mkdir -p ./eval_logs
mkdir -p ./parallel_logs

# 用于存储所有模型路径和实验信息的数组
declare -a experiments
declare -a running_jobs
declare -a job_pids

# 获取GPU显存使用情况
get_gpu_memory_usage() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1
    else
        echo "0"
    fi
}

# 检查GPU显存是否足够
check_gpu_memory() {
    local current_usage=$(get_gpu_memory_usage)
    local available_memory=$((24000 - current_usage))  # 假设24GB显存
    
    if [ $available_memory -gt $GPU_MEMORY_THRESHOLD ]; then
        return 0  # 显存足够
    else
        return 1  # 显存不足
    fi
}

# 实时日志输出函数
tail_log_with_prefix() {
    local log_file="$1"
    local prefix="$2"
    
    # 等待日志文件创建
    while [ ! -f "$log_file" ]; do
        sleep 1
    done
    
    # 使用tail -f跟踪日志，并添加前缀
    tail -f "$log_file" | while read line; do
        echo "[$prefix] $line"
    done &
    
    echo $!  # 返回tail进程的PID
}

# 从训练输出中提取模型路径
extract_model_path() {
    local log_file="$1"
    local model_path=$(grep "MODEL_PATH:" "$log_file" | tail -1 | cut -d' ' -f2-)
    echo "$model_path"
}

# 运行训练并记录结果（后台版本）
run_training_background() {
    local experiment_name="$1"
    local output_dir="$2"
    shift 2
    local train_cmd=("$@")
    
    echo "----------------------------------------"
    echo "准备后台运行实验: $experiment_name"
    echo "----------------------------------------"
    
    mkdir -p "$output_dir"
    local log_file="${output_dir}/training_log.txt"
    local status_file="${output_dir}/training_status.txt"
    
    echo "输出目录: $output_dir"
    echo "训练命令: ${train_cmd[@]}"
    echo "日志文件: $log_file"
    echo ""
    
    # 创建状态文件
    echo "RUNNING" > "$status_file"
    echo "START_TIME: $(date)" >> "$status_file"
    echo "EXPERIMENT: $experiment_name" >> "$status_file"
    echo "OUTPUT_DIR: $output_dir" >> "$status_file"
    
    # 后台运行训练命令
    (
        echo "开始训练: $experiment_name"
        echo "命令: ${train_cmd[@]}"
        echo "时间: $(date)"
        echo "========================================="
        
        # 运行训练
        if "${train_cmd[@]}"; then
            echo "SUCCESS" > "$status_file"
            echo "END_TIME: $(date)" >> "$status_file"
            
            # 提取模型路径
            model_path=$(extract_model_path "$log_file")
            if [ -n "$model_path" ] && [ -d "$model_path" ]; then
                echo "MODEL_PATH: $model_path" >> "$status_file"
            fi
            
            echo "========================================="
            echo "训练完成: $experiment_name"
            echo "时间: $(date)"
        else
            echo "FAILED" > "$status_file"
            echo "END_TIME: $(date)" >> "$status_file"
            echo "========================================="
            echo "训练失败: $experiment_name"
            echo "时间: $(date)"
        fi
    ) > "$log_file" 2>&1 &
    
    local job_pid=$!
    
    # 如果启用实时日志输出，启动日志跟踪
    if [ "$ENABLE_REAL_TIME_LOG" = true ]; then
        local tail_pid=$(tail_log_with_prefix "$log_file" "$experiment_name")
        echo "TAIL_PID: $tail_pid" >> "$status_file"
    fi
    
    echo "JOB_PID: $job_pid" >> "$status_file"
    echo "✓ 后台任务已启动: $experiment_name (PID: $job_pid)"
    echo ""
    
    return $job_pid
}

# 等待任务完成
wait_for_job_completion() {
    local job_pid="$1"
    local experiment_name="$2"
    local status_file="$3"
    
    # 等待进程结束
    wait $job_pid
    local exit_code=$?
    
    # 停止日志跟踪
    if [ "$ENABLE_REAL_TIME_LOG" = true ]; then
        local tail_pid=$(grep "TAIL_PID:" "$status_file" | cut -d' ' -f2)
        if [ -n "$tail_pid" ]; then
            kill $tail_pid 2>/dev/null
        fi
    fi
    
    # 检查训练状态
    local status=$(grep "^SUCCESS\|^FAILED" "$status_file" | tail -1)
    if [ "$status" = "SUCCESS" ]; then
        local model_path=$(grep "MODEL_PATH:" "$status_file" | cut -d' ' -f2-)
        if [ -n "$model_path" ]; then
            experiments+=("$experiment_name|$model_path")
            echo "✓ $experiment_name 训练成功完成"
            echo "  模型路径: $model_path"
        else
            echo "✓ $experiment_name 训练完成，但无法提取模型路径"
        fi
    else
        echo "✗ $experiment_name 训练失败 (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 监控并行任务
monitor_parallel_jobs() {
    echo "========================================"
    echo "并行任务监控"
    echo "========================================"
    
    while [ ${#running_jobs[@]} -gt 0 ]; do
        echo "当前运行的任务数: ${#running_jobs[@]}"
        echo "GPU显存使用: $(get_gpu_memory_usage) MB"
        echo "时间: $(date)"
        echo ""
        
        # 检查已完成的任务
        local completed_jobs=()
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name status_file <<< "$job_info"
            
            if ! kill -0 $job_pid 2>/dev/null; then
                # 任务已完成
                wait_for_job_completion $job_pid "$experiment_name" "$status_file"
                completed_jobs+=($i)
            fi
        done
        
        # 移除已完成的任务
        for i in "${completed_jobs[@]}"; do
            unset running_jobs[$i]
        done
        
        # 重新排列数组
        running_jobs=("${running_jobs[@]}")
        
        if [ ${#running_jobs[@]} -gt 0 ]; then
            sleep $MONITOR_INTERVAL
        fi
    done
    
    echo "所有并行任务已完成"
    echo ""
}

# 提交训练任务到队列
submit_training_job() {
    local experiment_name="$1"
    local output_dir="$2"
    shift 2
    local train_cmd=("$@")
    
    # 等待有空闲槽位
    while [ ${#running_jobs[@]} -ge $MAX_PARALLEL_JOBS ]; do
        echo "等待空闲槽位... 当前运行任务数: ${#running_jobs[@]}"
        sleep 2
        
        # 检查并移除完成的任务
        local completed_jobs=()
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name_check status_file <<< "$job_info"
            
            if ! kill -0 $job_pid 2>/dev/null; then
                wait_for_job_completion $job_pid "$experiment_name_check" "$status_file"
                completed_jobs+=($i)
            fi
        done
        
        for i in "${completed_jobs[@]}"; do
            unset running_jobs[$i]
        done
        running_jobs=("${running_jobs[@]}")
    done
    
    # 检查GPU显存
    if ! check_gpu_memory; then
        echo "GPU显存不足，等待释放..."
        sleep 10
        return 1
    fi
    
    # 启动训练任务
    run_training_background "$experiment_name" "$output_dir" "${train_cmd[@]}"
    local job_pid=$?
    
    # 添加到运行队列
    local status_file="${output_dir}/training_status.txt"
    running_jobs+=("$job_pid|$experiment_name|$status_file")
    
    echo "任务已提交: $experiment_name (PID: $job_pid)"
    echo "当前运行任务数: ${#running_jobs[@]}"
    echo ""
    
    return 0
}

# ========================================
# 第一部分：并行CLoRA实验
# ========================================
echo "第一部分：开始并行CLoRA k值实验..."
echo ""

for k in "${k_values[@]}"; do
    experiment_name="CLoRA_k${k}"
    output_dir="./output/clora_k_${k}"
    
    # 提交训练任务
    while ! submit_training_job "$experiment_name" "$output_dir" \
        python train_clora.py \
        --use_clora \
        --clora_k "$k" \
        --lambda_param 0.1 \
        --num_epochs 20 \
        --batch_size 32 \
        --learning_rate 2e-4 \
        --output_dir "$output_dir"; do
        echo "等待重试提交任务: $experiment_name"
        sleep 5
    done
    
    echo "已提交CLoRA k=${k}实验"
done

# 等待所有CLoRA实验完成
echo "等待所有CLoRA实验完成..."
monitor_parallel_jobs

echo "所有CLoRA k值实验完成！"
echo ""

# ========================================
# 第二部分：基准实验（标准LoRA）
# ========================================
echo "第二部分：开始基准实验..."
echo ""

experiment_name="Baseline_LoRA"
output_dir="./output/baseline_lora"

submit_training_job "$experiment_name" "$output_dir" \
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

submit_training_job "$experiment_name" "$output_dir" \
    python train_clora.py \
    --use_pca_grad \
    --pca_components 2 \
    --no_clora \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --output_dir "$output_dir"

# 等待所有剩余实验完成
echo "等待所有剩余实验完成..."
monitor_parallel_jobs

# ========================================
# 第四部分：并行模型评估
# ========================================
echo "第四部分：开始并行模型评估..."
echo ""

# 清空运行队列用于评估
running_jobs=()

# 评估也可以并行运行
MAX_EVAL_PARALLEL=3  # 评估并行数可以比训练高一些

eval_counter=1
for experiment_info in "${experiments[@]}"; do
    IFS='|' read -r experiment_name model_path <<< "$experiment_info"
    
    if [ -d "$model_path" ]; then
        # 等待评估槽位
        while [ ${#running_jobs[@]} -ge $MAX_EVAL_PARALLEL ]; do
            sleep 2
            
            # 检查完成的评估
            local completed_jobs=()
            for i in "${!running_jobs[@]}"; do
                local job_info="${running_jobs[$i]}"
                IFS='|' read -r job_pid exp_name status_file <<< "$job_info"
                
                if ! kill -0 $job_pid 2>/dev/null; then
                    wait $job_pid
                    if [ $? -eq 0 ]; then
                        echo "✓ 评估完成: $exp_name"
                    else
                        echo "✗ 评估失败: $exp_name"
                    fi
                    completed_jobs+=($i)
                fi
            done
            
            for i in "${completed_jobs[@]}"; do
                unset running_jobs[$i]
            done
            running_jobs=("${running_jobs[@]}")
        done
        
        echo "----------------------------------------"
        echo "开始评估 ${eval_counter}/${#experiments[@]}: $experiment_name"
        echo "----------------------------------------"
        
        eval_output_dir="./eval_logs/eval_${experiment_name}"
        mkdir -p "$eval_output_dir"
        
        echo "模型路径: $model_path"
        echo "评估输出目录: $eval_output_dir"
        
        # 后台运行评估
        (
            python evaluate_clora.py \
                --model-path "$model_path" \
                --output-dir "$eval_output_dir" \
                > "${eval_output_dir}/evaluation_log.txt" 2>&1
        ) &
        
        local eval_pid=$!
        local eval_status_file="${eval_output_dir}/eval_status.txt"
        echo "RUNNING" > "$eval_status_file"
        running_jobs+=("$eval_pid|$experiment_name|$eval_status_file")
        
        echo "评估任务已启动: $experiment_name (PID: $eval_pid)"
        echo ""
    else
        echo "模型路径不存在: $model_path (实验: $experiment_name)"
    fi
    
    ((eval_counter++))
done

# 等待所有评估完成
echo "等待所有评估完成..."
while [ ${#running_jobs[@]} -gt 0 ]; do
    sleep 2
    
    local completed_jobs=()
    for i in "${!running_jobs[@]}"; do
        local job_info="${running_jobs[$i]}"
        IFS='|' read -r job_pid exp_name status_file <<< "$job_info"
        
        if ! kill -0 $job_pid 2>/dev/null; then
            wait $job_pid
            if [ $? -eq 0 ]; then
                echo "✓ 评估完成: $exp_name"
            else
                echo "✗ 评估失败: $exp_name"
            fi
            completed_jobs+=($i)
        fi
    done
    
    for i in "${completed_jobs[@]}"; do
        unset running_jobs[$i]
    done
    running_jobs=("${running_jobs[@]}")
done

echo "所有评估完成！"
echo ""

# ========================================
# 生成实验报告
# ========================================
echo "========================================"
echo "生成实验报告..."
echo "========================================"

report_file="./output/experiment_report.txt"
{
    echo "CLoRA自动化实验报告（并行版）"
    echo "生成时间: $(date)"
    echo ""
    echo "实验配置："
    echo "- k值范围: ${k_values[@]}"
    echo "- 最大并行任务数: $MAX_PARALLEL_JOBS"
    echo "- 实时日志输出: $ENABLE_REAL_TIME_LOG"
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
    echo "并行日志位置: ./parallel_logs/"
} > "$report_file"

echo "实验报告已生成: $report_file"
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
echo "- 最大并行任务数: $MAX_PARALLEL_JOBS"
echo ""

echo "结果文件位置："
echo "- 训练结果: ./output/"
echo "- 评估结果: ./eval_logs/"
echo "- 实验报告: $report_file"
echo "- 并行日志: ./parallel_logs/"
echo ""

echo "查看实验结果的建议命令："
echo "cat $report_file"
echo "ls -la ./output/"
echo "ls -la ./eval_logs/"
echo ""

echo "并行实验脚本执行完毕！"