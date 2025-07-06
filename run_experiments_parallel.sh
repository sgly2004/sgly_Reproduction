#!/bin/bash

# CLoRA并行实验脚本 - 优化版
# 解决日志实时显示问题，支持智能并行训练

echo "========================================"
echo "开始CLoRA并行实验"
echo "========================================"

# ========================================
# 配置参数
# ========================================
k_values=(4 8 16 32 64 128 256 512)
MAX_PARALLEL_JOBS=3  # 默认并行数，可以通过命令行参数调整
MONITOR_INTERVAL=10  # 监控间隔(秒)
ENABLE_REAL_TIME_LOG=true  # 是否显示实时日志
GPU_MEMORY_THRESHOLD=6000  # GPU显存阈值(MB)

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --gpu-threshold)
            GPU_MEMORY_THRESHOLD="$2"
            shift 2
            ;;
        --no-realtime-log)
            ENABLE_REAL_TIME_LOG=false
            shift
            ;;
        --monitor-interval)
            MONITOR_INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "配置参数："
echo "- 最大并行任务数: $MAX_PARALLEL_JOBS"
echo "- GPU显存阈值: ${GPU_MEMORY_THRESHOLD}MB"
echo "- 实时日志输出: $ENABLE_REAL_TIME_LOG"
echo "- 监控间隔: ${MONITOR_INTERVAL}秒"
echo "- 将要测试的k值: ${k_values[@]}"
echo ""

# ========================================
# 工具函数
# ========================================
mkdir -p ./output
mkdir -p ./eval_logs
mkdir -p ./parallel_logs

# 用于存储所有模型路径和实验信息的数组
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
    # 假设总显存24GB，也可以通过nvidia-smi查询
    local available_memory=$((24000 - current_usage))
    
    echo "[GPU检查] 当前使用: ${current_usage}MB, 可用: ${available_memory}MB, 阈值: ${GPU_MEMORY_THRESHOLD}MB"
    
    if [ $available_memory -gt $GPU_MEMORY_THRESHOLD ]; then
        return 0  # 显存足够
    else
        return 1  # 显存不足
    fi
}

# 实时日志输出函数（带颜色和前缀）
tail_log_with_prefix() {
    local log_file="$1"
    local prefix="$2"
    local color_code="$3"
    
    # 等待日志文件创建
    while [ ! -f "$log_file" ]; do
        sleep 1
    done
    
    # 使用tail -f跟踪日志，并添加颜色前缀
    tail -f "$log_file" | while read line; do
        echo -e "\\033[${color_code}m[$prefix]\\033[0m $line"
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
                echo "模型已保存到: $model_path"
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
        # 使用不同颜色区分不同实验
        local colors=("32" "33" "34" "35" "36" "91" "92" "93" "94" "95")
        local color_index=$((${#running_jobs[@]} % ${#colors[@]}))
        local color_code="${colors[$color_index]}"
        
        local tail_pid=$(tail_log_with_prefix "$log_file" "$experiment_name" "$color_code")
        echo "TAIL_PID: $tail_pid" >> "$status_file"
        echo "启动实时日志跟踪: $experiment_name (Tail PID: $tail_pid)"
    fi
    
    echo "JOB_PID: $job_pid" >> "$status_file"
    echo "✓ 后台任务已启动: $experiment_name (PID: $job_pid)"
    
    return $job_pid
}

# 等待任务完成
wait_for_job_completion() {
    local job_pid="$1"
    local experiment_name="$2"
    local status_file="$3"
    
    echo "等待任务完成: $experiment_name (PID: $job_pid)"
    
    # 等待进程结束
    wait $job_pid
    local exit_code=$?
    
    # 停止日志跟踪
    if [ "$ENABLE_REAL_TIME_LOG" = true ]; then
        local tail_pid=$(grep "TAIL_PID:" "$status_file" | cut -d' ' -f2)
        if [ -n "$tail_pid" ]; then
            kill $tail_pid 2>/dev/null
            echo "停止实时日志跟踪: $experiment_name (Tail PID: $tail_pid)"
        fi
    fi
    
    # 检查训练状态
    local status=$(grep "^SUCCESS\\|^FAILED" "$status_file" | tail -1)
    if [ "$status" = "SUCCESS" ]; then
        local model_path=$(grep "MODEL_PATH:" "$status_file" | cut -d' ' -f2-)
        if [ -n "$model_path" ]; then
            experiments+=("$experiment_name|$model_path")
            echo " $experiment_name 训练成功完成"
            echo "   模型路径: $model_path"
        else
            echo " $experiment_name 训练完成，但无法提取模型路径"
        fi
    else
        echo " $experiment_name 训练失败 (退出码: $exit_code)"
    fi
    
    return $exit_code
}

# 监控并行任务
monitor_parallel_jobs() {
    echo "========================================"
    echo "开始监控并行任务"
    echo "========================================"
    
    local monitor_count=0
    
    while [ ${#running_jobs[@]} -gt 0 ]; do
        ((monitor_count++))
        local current_memory=$(get_gpu_memory_usage)
        
        echo ""
        echo "=== 监控周期 $monitor_count ==="
        echo "当前运行的任务数: ${#running_jobs[@]}"
        echo "GPU显存使用: ${current_memory} MB"
        echo "时间: $(date)"
        
        # 显示当前运行的任务详情
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name status_file <<< "$job_info"
            echo "   任务 $((i+1)): $experiment_name (PID: $job_pid)"
        done
        
        # 检查已完成的任务
        local completed_jobs=()
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name status_file <<< "$job_info"
            
            if ! kill -0 $job_pid 2>/dev/null; then
                echo " 检测到任务完成: $experiment_name (PID: $job_pid)"
                wait_for_job_completion $job_pid "$experiment_name" "$status_file"
                completed_jobs+=($i)
            fi
        done
        
        # 移除已完成的任务
        if [ ${#completed_jobs[@]} -gt 0 ]; then
            echo " 移除 ${#completed_jobs[@]} 个已完成的任务"
            for i in "${completed_jobs[@]}"; do
                unset running_jobs[$i]
            done
            running_jobs=("${running_jobs[@]}")
        fi
        
        if [ ${#running_jobs[@]} -gt 0 ]; then
            echo " 等待 ${MONITOR_INTERVAL} 秒后继续监控..."
            sleep $MONITOR_INTERVAL
        fi
    done
    
    echo " 所有并行任务已完成"
}

# 提交训练任务到队列
submit_training_job() {
    local experiment_name="$1"
    local output_dir="$2"
    shift 2
    local train_cmd=("$@")
    
    echo " 准备提交训练任务: $experiment_name"
    
    # 等待有空闲槽位
    local wait_count=0
    while [ ${#running_jobs[@]} -ge $MAX_PARALLEL_JOBS ]; do
        ((wait_count++))
        echo " 等待空闲槽位... 当前运行任务数: ${#running_jobs[@]} (等待次数: $wait_count)"
        sleep 2
        
        # 检查并移除完成的任务
        local completed_jobs=()
        for i in "${!running_jobs[@]}"; do
            local job_info="${running_jobs[$i]}"
            IFS='|' read -r job_pid experiment_name_check status_file <<< "$job_info"
            
            if ! kill -0 $job_pid 2>/dev/null; then
                echo " 检测到任务完成: $experiment_name_check"
                wait_for_job_completion $job_pid "$experiment_name_check" "$status_file"
                completed_jobs+=($i)
            fi
        done
        
        if [ ${#completed_jobs[@]} -gt 0 ]; then
            for i in "${completed_jobs[@]}"; do
                unset running_jobs[$i]
            done
            running_jobs=("${running_jobs[@]}")
            echo "🧹 已清理 ${#completed_jobs[@]} 个完成的任务"
        fi
    done
    
    # 检查GPU显存
    if ! check_gpu_memory; then
        echo " GPU显存不足，等待释放..."
        sleep 10
        return 1
    fi
    
    # 启动训练任务
    run_training_background "$experiment_name" "$output_dir" "${train_cmd[@]}"
    local job_pid=$?
    
    # 添加到运行队列
    local status_file="${output_dir}/training_status.txt"
    running_jobs+=("$job_pid|$experiment_name|$status_file")
    
    echo " 任务已提交: $experiment_name (PID: $job_pid)"
    echo " 当前运行任务数: ${#running_jobs[@]}"
    
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
    
    echo " 准备提交实验: $experiment_name"
    
    # 提交训练任务（带重试机制）
    while ! submit_training_job "$experiment_name" "$output_dir" \
        python train_clora.py \
        --use_clora \
        --clora_k "$k" \
        --lambda_param 0.1 \
        --num_epochs 20 \
        --batch_size 32 \
        --learning_rate 2e-4 \
        --output_dir "$output_dir"; do
        echo " 等待重试提交任务: $experiment_name"
        sleep 5
    done
    
    echo " 已提交CLoRA k=${k}实验"
    echo ""
done

# 等待所有CLoRA实验完成
echo " 等待所有CLoRA实验完成..."
monitor_parallel_jobs

echo " 所有CLoRA k值实验完成！"
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
echo " 等待所有剩余实验完成..."
monitor_parallel_jobs

# ========================================
# 第四部分：模型评估
# ========================================
echo "第四部分：开始模型评估..."
echo ""

eval_counter=1
for experiment_info in "${experiments[@]}"; do
    IFS='|' read -r experiment_name model_path <<< "$experiment_info"
    
    if [ -d "$model_path" ]; then
        echo "----------------------------------------"
        echo " 评估模型 ${eval_counter}/${#experiments[@]}: $experiment_name"
        echo "----------------------------------------"
        
        eval_output_dir="./eval_logs/eval_${experiment_name}"
        mkdir -p "$eval_output_dir"
        
        echo "模型路径: $model_path"
        echo "评估输出目录: $eval_output_dir"
        
        # 评估时也显示实时输出
        if [ "$ENABLE_REAL_TIME_LOG" = true ]; then
            echo "开始评估，实时输出如下："
            python evaluate_clora.py \
                --model-path "$model_path" \
                --output-dir "$eval_output_dir" | \
                while read line; do
                    echo "  [EVAL-$experiment_name] $line"
                done
        else
            python evaluate_clora.py \
                --model-path "$model_path" \
                --output-dir "$eval_output_dir" \
                > "${eval_output_dir}/evaluation_log.txt" 2>&1
        fi
        
        if [ $? -eq 0 ]; then
            echo " 模型评估完成: $experiment_name"
        else
            echo " 模型评估失败: $experiment_name"
            echo "   查看日志: ${eval_output_dir}/evaluation_log.txt"
        fi
        
        echo ""
    else
        echo "  模型路径不存在: $model_path (实验: $experiment_name)"
    fi
    
    ((eval_counter++))
done

# ========================================
# 生成综合总结报告
# ========================================
echo "========================================"
echo " 生成综合总结报告..."
echo "========================================"

mkdir -p ./experiment_report

timestamp=$(date +"%Y%m%d_%H%M%S")
comprehensive_report="./experiment_report/comprehensive_report_${timestamp}.txt"

{
    echo "========================================"
    echo "CLoRA并行实验综合报告"
    echo "========================================"
    echo "生成时间: $(date)"
    echo "脚本版本: Parallel Version"
    echo "并行配置: 最大${MAX_PARALLEL_JOBS}个任务同时运行"
    echo ""
    
    echo "实验配置概览："
    echo "- k值范围: ${k_values[@]}"
    echo "- 总实验数量: $((${#k_values[@]} + 2))"
    echo "- 成功完成的实验: ${#experiments[@]}"
    echo "- 最大并行任务数: $MAX_PARALLEL_JOBS"
    echo "- GPU显存阈值: ${GPU_MEMORY_THRESHOLD}MB"
    echo ""
    
    echo "成功完成的实验："
    for experiment_info in "${experiments[@]}"; do
        IFS='|' read -r experiment_name model_path <<< "$experiment_info"
        echo "- $experiment_name: $model_path"
    done
    echo ""
    
} > "$comprehensive_report"

echo " 综合总结报告已生成: $comprehensive_report"
echo ""

# ========================================
# 实验总结
# ========================================
echo "========================================"
echo " 所有实验完成！"
echo "========================================"

echo ""
echo " 实验总结："
echo "- CLoRA k值实验: ${#k_values[@]} 个"
echo "- 基准LoRA实验: 1 个"
echo "- PCA梯度实验: 1 个"
echo "- 成功训练的模型: ${#experiments[@]} 个"
echo "- 最大并行任务数: $MAX_PARALLEL_JOBS"
echo ""

echo " 结果文件位置："
echo "- 训练结果: ./output/"
echo "- 评估结果: ./eval_logs/"
echo "- 综合报告: $comprehensive_report"
echo "- 并行日志: ./parallel_logs/"
echo ""

echo " 并行实验脚本执行完毕！"