"""
模型设置模块 - 加载GPT2模型并集成LoRA
"""

from transformers import AutoModelForSequenceClassification, GPT2Tokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_base_model(model_name="gpt2", num_labels=2):
    """
    加载基础LLM模型
    
    Args:
        model_name: 模型名称
        num_labels: 分类标签数量
    
    Returns:
        基础模型
    """
    print(f"正在加载基础模型: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        pad_token_id=50256  # GPT2的eos_token_id
    )
    return model


def create_lora_config():
    """
    创建LoRA配置
    
    Returns:
        LoraConfig对象
    """
    print("正在创建LoRA配置...")
    lora_config = LoraConfig(
        r=8,                           # LoRA rank
        lora_alpha=16,                 # LoRA scaling parameter
        target_modules=["c_attn", "c_proj"],  # 目标模块
        lora_dropout=0.1,              # LoRA dropout
        bias="none",                   # bias设置
        task_type=TaskType.SEQ_CLS,    # 任务类型：序列分类
    )
    return lora_config


def setup_lora_model(model_name="gpt2", num_labels=2):
    """
    设置集成LoRA的模型
    
    Args:
        model_name: 基础模型名称
        num_labels: 分类标签数量
    
    Returns:
        集成LoRA的PeftModel
    """
    # 1. 加载基础模型
    base_model = load_base_model(model_name, num_labels)
    
    # 2. 创建LoRA配置
    lora_config = create_lora_config()
    
    # 3. 集成LoRA
    print("正在集成LoRA到基础模型...")
    lora_model = get_peft_model(base_model, lora_config)
    
    return lora_model


def count_parameters(model):
    """
    计算模型参数数量
    
    Args:
        model: 模型对象
    
    Returns:
        tuple: (可训练参数数量, 总参数数量)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return trainable_params, total_params


def print_model_info(model):
    """
    打印模型信息
    
    Args:
        model: 模型对象
    """
    trainable_params, total_params = count_parameters(model)
    
    print("\n" + "=" * 50)
    print("模型参数信息")
    print("=" * 50)
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数占比: {100 * trainable_params / total_params:.2f}%")
    
    # 如果是PEFT模型，打印LoRA相关信息
    if hasattr(model, 'print_trainable_parameters'):
        print("\nLoRA模型训练参数详情:")
        model.print_trainable_parameters()


def main():
    """测试函数"""
    print("=" * 50)
    print("LoRA模型设置测试")
    print("=" * 50)
    
    try:
        # 设置LoRA模型
        lora_model = setup_lora_model()
        
        # 打印模型信息
        print_model_info(lora_model)
        
        # 验证模型结构
        print("\n" + "=" * 30)
        print("模型结构验证")
        print("=" * 30)
        print(f"模型类型: {type(lora_model)}")
        print(f"基础模型类型: {type(lora_model.base_model)}")
        
        # 检查LoRA模块
        print("\nLoRA目标模块:")
        for name, module in lora_model.named_modules():
            if 'lora' in name.lower():
                print(f"  {name}: {type(module)}")
        
        print("\nLoRA模型设置完成!")
        
        return lora_model
        
    except Exception as e:
        print(f"错误: {e}")
        return None


if __name__ == "__main__":
    main()