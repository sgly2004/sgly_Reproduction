#!/bin/bash

# CLoRA自动化实验脚本（日志输出优化版）
# 解决日志输出问题，确保所有echo都能看到

# ========================================
# 全局日志配置
# ========================================
# 创建主日志文件
MAIN_LOG="./parallel_logs/main_experiment.log"
mkdir -p ./parallel_logs

# 日志函数 - 同时输出到终端和文件
log_info() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$MAIN_LOG"
}

log_debug() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [DEBUG] $message" | tee -a "$MAIN_LOG"
}

log_error() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [ERROR] $message" | tee -a "$MAIN_LOG" >&2
}

# 初始化日志
log_info "========================================"
log_info "开始CLoRA自动化实验（日志优化版）"
log_info "========================================"

# ========================================
# 配置参数
# ========================================
k_values=(1 4 8 16 32 64 128 256 512)
MAX_PARALLEL_JOBS=3
MONITOR_INTERVAL=5
ENABLE_REAL_TIME_LOG=true
GPU_MEMORY_THRESHOLD=4000  # 降低阈值到4GB

log_info "配置参数："
log_info "- 最大并行任务数: $MAX_PARALLEL_JOBS"
log_info "- 监控间隔: ${MONITOR_INTERVAL}秒"
log_info "- 实时日志输出: $ENABLE_REAL_TIME_LOG"
log_info "- GPU显存阈值: ${GPU_MEMORY_THRESHOLD}MB"
log_info "- 将要测试的k值: ${k_values[@]}"
log_info "- 主日志文件: $MAIN_LOG"

# ========================================
# 工具函数
# ========================================
mkdir -p ./output
mkdir -p ./eval_logs

declare -a experiments
declare -a running_jobs

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
    local available_memory=$((24000 - current_usage))
    
    log_debug "GPU显存检查: 当前使用=${current_usage}MB, 可用=${available_memory}MB, 阈值=${GPU_MEMORY_THRESHOLD}MB"
    
    if [ $available_memory -gt $GPU_MEMORY_THRESHOLD ]; then
        log_debug "GPU显存检查: 通过"
        return 0
    else
        log_debug "GPU显存检查: 不足"
        return 1
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
    
    # 使用tail -f跟踪日志，并添加前缀，同时写入主日志
    tail -f "$log_file" | while read line; do
        log_info "[$prefix] $line"
    done &
    
    echo $!
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
    
    log_info "----------------------------------------"
    log_info "准备后台运行实验: $experiment_name"
    log_info "----------------------------------------"
    
    mkdir -p "$output_dir"
    local log_file="${output_dir}/training_log.txt"
    local status_file="${output_dir}/training_status.txt"
    
    log_info "输出目录: $output_dir"
    log_info "训练命令: ${train_cmd[@]}"
    log_info "日志文件: $log_file"
    
    # 创建状态文件
    echo "RUNNING" > "$status_file"
    echo "START_TIME: $(date)" >> "$status_file"
    echo "EXPERIMENT: $experiment_name" >> "$status_file"
    echo "OUTPUT_DIR: $output_dir" >> "$status_file"
    
    # 后台运行训练命令
    (
        log_info "开始训练: $experiment_name"
        log_info "命令: ${train_cmd[@]}"
        log_info "时间: $(date)"
        echo "========================================="
        
        # 运行训练
        if "${train_cmd[@]}"; then
            echo "SUCCESS" > "$status_file"
            echo "END_TIME: $(date)" >> "$status_file"
            
            # 提取模型路径
            model_path=$(extract_model_path "$log_file")
            if [ -n "$model_path" ] && [ -d "$model_path" ]; then
                echo "MODEL_PATH: $model_path" >> "$status_file"
                log_info "模型已保存到: $model_path"
            fi
            
            echo "========================================="
            log_info "训练完成: $experiment_name"
            log_info "时间: $(date)"
        else
            echo "FAILED" > "$status_file"
            echo "END_TIME: $(date)" >> "$status_file"
            echo "========================================="
            log_error "训练失败: $experiment_name"
            log_info "时间: $(date)"
        fi
    ) > "$log_file" 2>&1 &
    
    local job_pid=$!
    
    # 如果启用实时日志输出，启动日志跟踪
    if [ "$ENABLE_REAL_TIME_LOG" = true ]; then
        local tail_pid=$(tail_log_with_prefix "$log_file" "$experiment_name")
        echo "TAIL_PID: $tail_pid" >> "$status_file"
    fi
    
    echo "JOB_PID: $job_pid" >> "$status_file"
    log_info "✓ 后台任务已启动: $experiment_name (PID: $job_pid)"
    
    return $job_pid
}

# 等待任务完成
wait_for_job_completion() {
    local job_pid="$1"
    local experiment_name="$2"
    local status_file="$3"
    
    log_debug "等待任务完成: $experiment_name (PID: $job_pid)"
    
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
            log_info "✓ $experiment_name 训练成功完成"
            log_info "  模型路径: $model_path"
        else
            log_info "✓ $experiment_name 训练完成，但无法提取模型路径"
        fi
    else
        log_error "✗ $experiment_name 训练失败 (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 监控并行任务 - 重点修复这个函数的日志输出
monitor_parallel_jobs() {
    log_info "========================================"
    log_info "开始监控并行任务"
    log_info "========================================"
    
    local monitor_count=0
    
    while [ ${#running_jobs[@]} -gt 0 ]; do
        ((monitor_count++))
        local current_memory=$(get_gpu_memory_usage)
        
        log_info "=== 监控周期 $monitor_count ==="
        log_info "当前运行的任务数: ${#running_jobs[@]}"
        log_info "GPU显存使用: ${current_memory} MB"
        log_info "时间: $(date)"
        
        # 显示当前运行的任务详情
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name status_file <<< "$job_info"
            log_info "  - 任务 $((i+1)): $experiment_name (PID: $job_pid)"
        done
        
        # 检查已完成的任务
        local completed_jobs=()
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name status_file <<< "$job_info"
            
            if ! kill -0 $job_pid 2>/dev/null; then
                log_info "检测到任务完成: $experiment_name (PID: $job_pid)"
                wait_for_job_completion $job_pid "$experiment_name" "$status_file"
                completed_jobs+=($i)
            fi
        done
        
        # 移除已完成的任务
        if [ ${#completed_jobs[@]} -gt 0 ]; then
            log_info "移除 ${#completed_jobs[@]} 个已完成的任务"
            for i in "${completed_jobs[@]}"; do
                unset running_jobs[$i]
            done
            running_jobs=("${running_jobs[@]}")
        fi
        
        if [ ${#running_jobs[@]} -gt 0 ]; then
            log_info "等待 ${MONITOR_INTERVAL} 秒后继续监控..."
            sleep $MONITOR_INTERVAL
        fi
    done
    
    log_info "所有并行任务已完成"
}

# 提交训练任务到队列
submit_training_job() {
    local experiment_name="$1"
    local output_dir="$2"
    shift 2
    local train_cmd=("$@")
    
    log_info "准备提交训练任务: $experiment_name"
    
    # 等待有空闲槽位
    local wait_count=0
    while [ ${#running_jobs[@]} -ge $MAX_PARALLEL_JOBS ]; do
        ((wait_count++))
        log_info "等待空闲槽位... 当前运行任务数: ${#running_jobs[@]} (等待次数: $wait_count)"
        sleep 2
        
        # 检查并移除完成的任务
        local completed_jobs=()
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name_check status_file <<< "$job_info"
            
            if ! kill -0 $job_pid 2>/dev/null; then
                log_info "检测到任务完成: $experiment_name_check"
                wait_for_job_completion $job_pid "$experiment_name_check" "$status_file"
                completed_jobs+=($i)
            fi
        done
        
        if [ ${#completed_jobs[@]} -gt 0 ]; then
            for i in "${completed_jobs[@]}"; do
                unset running_jobs[$i]
            done
            running_jobs=("${running_jobs[@]}")
            log_info "已清理 ${#completed_jobs[@]} 个完成的任务"
        fi
    done
    
    # 检查GPU显存
    if ! check_gpu_memory; then
        log_info "GPU显存不足，等待释放..."
        sleep 10
        return 1
    fi
    
    # 启动训练任务
    run_training_background "$experiment_name" "$output_dir" "${train_cmd[@]}"
    local job_pid=$?
    
    # 添加到运行队列
    local status_file="${output_dir}/training_status.txt"
    running_jobs+=("$job_pid|$experiment_name|$status_file")
    
    log_info "✓ 任务已提交: $experiment_name (PID: $job_pid)"
    log_info "当前运行任务数: ${#running_jobs[@]}"
    
    return 0
}

# ========================================
# 第一部分：并行CLoRA实验
# ========================================
log_info "第一部分：开始并行CLoRA k值实验..."

for k in "${k_values[@]}"; do
    experiment_name="CLoRA_k${k}"
    output_dir="./output/clora_k_${k}"
    
    log_info "准备提交实验: $experiment_name"
    
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
        log_info "等待重试提交任务: $experiment_name"
        sleep 5
    done
    
    log_info "✓ 已提交CLoRA k=${k}实验"
done

# 等待所有CLoRA实验完成
log_info "等待所有CLoRA实验完成..."
monitor_parallel_jobs

log_info "所有CLoRA k值实验完成！"

# ========================================
# 第二部分：基准实验（标准LoRA）
# ========================================
log_info "第二部分：开始基准实验..."

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
log_info "第三部分：开始PCA梯度实验..."

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
log_info "等待所有剩余实验完成..."
monitor_parallel_jobs

# ========================================
# 实验总结
# ========================================
log_info "========================================"
log_info "所有实验完成！"
log_info "========================================"

log_info "实验总结："
log_info "- CLoRA k值实验: ${#k_values[@]} 个"
log_info "- 成功训练的模型: ${#experiments[@]} 个"
log_info "- 最大并行任务数: $MAX_PARALLEL_JOBS"

log_info "结果文件位置："
log_info "- 训练结果: ./output/"
log_info "- 评估结果: ./eval_logs/"
log_info "- 主日志文件: $MAIN_LOG"

log_info "查看日志的命令："
log_info "tail -f $MAIN_LOG              # 实时查看主日志"
log_info "grep 'ERROR' $MAIN_LOG         # 查看错误信息"
log_info "grep 'GPU显存' $MAIN_LOG       # 查看显存使用情况"
log_info "grep '✓.*训练成功' $MAIN_LOG   # 查看成功的训练"

log_info "日志优化版脚本执行完毕！"