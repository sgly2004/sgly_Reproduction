"""
通用日志系统模块
支持同时输出到控制台和文件的日志记录
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional, List


class DualOutputLogger:
    """双输出日志器 - 同时输出到控制台和文件"""
    
    def __init__(self, 
                 name: str = "CLoRA",
                 log_dir: str = "./logs",
                 log_level: int = logging.INFO,
                 console_output: bool = True):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_dir: 日志文件目录
            log_level: 日志级别
            console_output: 是否输出到控制台
        """
        self.name = name
        self.log_dir = log_dir
        self.console_output = console_output
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # 设置日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 记录初始化信息
        self.logger.info(f"日志系统初始化完成")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info(f"当前时间: {datetime.now().isoformat()}")
    
    def log_command_info(self, command_args: List[str], script_name: str):
        """记录命令信息"""
        self.logger.info("=" * 80)
        self.logger.info(f"脚本启动: {script_name}")
        self.logger.info(f"执行命令: {' '.join(command_args)}")
        self.logger.info(f"工作目录: {os.getcwd()}")
        self.logger.info(f"Python版本: {sys.version}")
        self.logger.info("=" * 80)
    
    def info(self, message: str):
        """记录信息级别日志"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """记录调试级别日志"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """记录警告级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录错误级别日志"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """记录严重错误级别日志"""
        self.logger.critical(message)
    
    def log_section(self, title: str, level: str = "="):
        """记录章节标题"""
        separator = level * 60
        self.logger.info(f"\n{separator}")
        self.logger.info(f"{title}")
        self.logger.info(separator)
    
    def log_config(self, config_dict: dict, title: str = "配置信息"):
        """记录配置信息"""
        self.log_section(title)
        for key, value in config_dict.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics_dict: dict, title: str = "指标信息"):
        """记录指标信息"""
        self.log_section(title, "-")
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_step(self, step: int, description: str):
        """记录训练/评估步骤"""
        self.logger.info(f"\n步骤 {step}: {description}")
    
    def log_results(self, results_dict: dict, title: str = "结果"):
        """记录结果信息"""
        self.log_section(title)
        for key, value in results_dict.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """记录异常信息"""
        self.logger.error(f"异常发生 {context}: {str(exception)}")
        self.logger.exception("详细异常信息:")
    
    def get_log_file_path(self) -> str:
        """获取日志文件路径"""
        return self.log_file
    
    def close(self):
        """关闭日志器"""
        self.logger.info("日志系统关闭")
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def setup_logger(name: str, 
                 log_dir: str = "./logs", 
                 console_output: bool = True,
                 command_args: Optional[List[str]] = None,
                 script_name: str = "") -> DualOutputLogger:
    """
    设置日志器的便捷函数
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        console_output: 是否输出到控制台
        command_args: 命令行参数
        script_name: 脚本名称
    
    Returns:
        DualOutputLogger: 配置好的日志器
    """
    logger = DualOutputLogger(
        name=name,
        log_dir=log_dir,
        console_output=console_output
    )
    
    # 记录命令信息
    if command_args and script_name:
        logger.log_command_info(command_args, script_name)
    
    return logger


# 重定向print函数的类
class LoggerPrintRedirect:
    """重定向print输出到日志器"""
    
    def __init__(self, logger: DualOutputLogger, original_stdout=None):
        self.logger = logger
        self.original_stdout = original_stdout or sys.stdout
    
    def write(self, text):
        """重写write方法"""
        text = text.strip()
        if text:  # 忽略空行
            self.logger.info(text)
        # 同时输出到原始stdout（如果需要）
        if hasattr(self, 'also_print') and self.also_print:
            self.original_stdout.write(text + '\n')
    
    def flush(self):
        """flush方法"""
        pass


def redirect_print_to_logger(logger: DualOutputLogger, also_print: bool = False):
    """
    将print输出重定向到日志器
    
    Args:
        logger: 日志器实例
        also_print: 是否同时保持原始print输出
    
    Returns:
        原始的stdout，用于恢复
    """
    original_stdout = sys.stdout
    redirect = LoggerPrintRedirect(logger, original_stdout)
    redirect.also_print = also_print
    sys.stdout = redirect
    return original_stdout


def restore_print(original_stdout):
    """恢复原始的print输出"""
    sys.stdout = original_stdout