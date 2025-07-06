"""
数据处理模块 - 处理多个数据集用于CLoRA微调
支持缓存机制以提高加载效率
"""

from datasets import load_dataset
from transformers import GPT2Tokenizer
import random
import os
from cache_manager import get_cache_manager


def load_tokenizer():
    """加载GPT2分词器（支持缓存）"""
    cache_manager = get_cache_manager()
    
    # 尝试从缓存加载
    tokenizer = cache_manager.load_tokenizer(GPT2Tokenizer, "gpt2")
    
    if tokenizer is None:
        print("正在加载GPT2分词器...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # 为GPT2添加pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 缓存分词器
        cache_manager.cache_tokenizer(tokenizer, "gpt2")
    
    return tokenizer


def preprocess_commonsense_qa(examples, tokenizer, max_length=512):
    """
    预处理CommonsenseQA数据集
    
    Args:
        examples: 数据批次
        tokenizer: 分词器
        max_length: 最大序列长度
    
    Returns:
        处理后的数据批次
    """
    inputs = []
    labels = []
    
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        choices = examples["choices"][i]["text"]
        answer_key = examples["answerKey"][i]
        
        # 将选择题转换为二分类：正确答案为1，错误答案为0
        # 对于每个问题，我们随机选择一个选项作为输入
        correct_idx = ord(answer_key) - ord('A')
        
        # 生成正样本（正确答案）
        correct_choice = choices[correct_idx]
        input_text = f"Question: {question}\nAnswer: {correct_choice}"
        inputs.append(input_text)
        labels.append(1)
        
        # 生成负样本（错误答案）
        wrong_choices = [choices[j] for j in range(len(choices)) if j != correct_idx]
        if wrong_choices:
            wrong_choice = random.choice(wrong_choices)
            input_text = f"Question: {question}\nAnswer: {wrong_choice}"
            inputs.append(input_text)
            labels.append(0)
    
    # 使用分词器编码
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors=None
    )
    
    model_inputs["labels"] = labels
    return model_inputs


def preprocess_boolq(examples, tokenizer, max_length=512):
    """
    预处理BoolQ数据集
    """
    inputs = []
    for question, passage in zip(examples["question"], examples["passage"]):
        input_text = f"Question: {question}\nPassage: {passage}\nAnswer:"
        inputs.append(input_text)
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors=None
    )
    
    labels = [1 if answer else 0 for answer in examples["answer"]]
    model_inputs["labels"] = labels
    return model_inputs


def preprocess_piqa(examples, tokenizer, max_length=512):
    """
    预处理PIQA数据集
    """
    inputs = []
    labels = []
    
    for i in range(len(examples["goal"])):
        goal = examples["goal"][i]
        sol1 = examples["sol1"][i]
        sol2 = examples["sol2"][i]
        label = examples["label"][i]
        
        # 生成正样本
        correct_sol = sol1 if label == 0 else sol2
        input_text = f"Goal: {goal}\nSolution: {correct_sol}"
        inputs.append(input_text)
        labels.append(1)
        
        # 生成负样本
        wrong_sol = sol2 if label == 0 else sol1
        input_text = f"Goal: {goal}\nSolution: {wrong_sol}"
        inputs.append(input_text)
        labels.append(0)
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors=None
    )
    
    model_inputs["labels"] = labels
    return model_inputs


def preprocess_winogrande(examples, tokenizer, max_length=512):
    """
    预处理Winogrande数据集
    """
    inputs = []
    labels = []
    
    for i in range(len(examples["sentence"])):
        sentence = examples["sentence"][i]
        option1 = examples["option1"][i]
        option2 = examples["option2"][i]
        answer = examples["answer"][i]
        
        if answer in ['1', '2']:
            # 生成正样本
            correct_option = option1 if answer == '1' else option2
            completed_sentence = sentence.replace('_', correct_option)
            inputs.append(completed_sentence)
            labels.append(1)
            
            # 生成负样本
            wrong_option = option2 if answer == '1' else option1
            wrong_sentence = sentence.replace('_', wrong_option)
            inputs.append(wrong_sentence)
            labels.append(0)
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors=None
    )
    
    model_inputs["labels"] = labels
    return model_inputs


def preprocess_arc(examples, tokenizer, max_length=512):
    """
    预处理AI2 ARC数据集
    
    Args:
        examples: 数据批次
        tokenizer: 分词器
        max_length: 最大序列长度
    
    Returns:
        处理后的数据批次
    """
    inputs = []
    labels = []
    
    for i in range(len(examples["question"])):
        question = examples["question"][i]
        choices = examples["choices"][i]
        answer_key = examples["answerKey"][i]
        
        # 获取选项文本和标签
        choice_texts = choices["text"]
        choice_labels = choices["label"]
        
        # 找到正确答案的索引
        try:
            correct_idx = choice_labels.index(answer_key)
        except ValueError:
            # 如果找不到对应的标签，跳过这个样本
            continue
        
        # 生成正样本（正确答案）
        correct_choice = choice_texts[correct_idx]
        input_text = f"Question: {question}\nAnswer: {correct_choice}"
        inputs.append(input_text)
        labels.append(1)
        
        # 生成负样本（错误答案）
        wrong_choices = [choice_texts[j] for j in range(len(choice_texts)) if j != correct_idx]
        if wrong_choices:
            wrong_choice = random.choice(wrong_choices)
            input_text = f"Question: {question}\nAnswer: {wrong_choice}"
            inputs.append(input_text)
            labels.append(0)
    
    # 使用分词器编码
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors=None
    )
    
    model_inputs["labels"] = labels
    return model_inputs


def preprocess_hellaswag(examples, tokenizer, max_length=512):
    """
    预处理HellaSwag数据集
    
    Args:
        examples: 数据批次
        tokenizer: 分词器
        max_length: 最大序列长度
    
    Returns:
        处理后的数据批次
    """
    inputs = []
    labels = []
    
    for i in range(len(examples["ctx"])):
        context = examples["ctx"][i]
        endings = examples["endings"][i]
        label = int(examples["label"][i])
        
        # 生成正样本
        correct_ending = endings[label]
        input_text = f"Context: {context}\nEnding: {correct_ending}"
        inputs.append(input_text)
        labels.append(1)
        
        # 生成负样本
        wrong_endings = [endings[j] for j in range(len(endings)) if j != label]
        if wrong_endings:
            wrong_ending = random.choice(wrong_endings)
            input_text = f"Context: {context}\nEnding: {wrong_ending}"
            inputs.append(input_text)
            labels.append(0)
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors=None
    )
    
    model_inputs["labels"] = labels
    return model_inputs


def get_train_data(max_samples=None):
    """
    获取训练数据（使用CommonsenseQA）支持缓存
    
    Args:
        max_samples: 最大样本数量（用于快速测试）
    
    Returns:
        tuple: (train_dataset, tokenizer)
    """
    cache_manager = get_cache_manager()
    
    # 尝试从缓存加载
    cache_key_params = {"split": "train", "max_samples": max_samples}
    processed_dataset = cache_manager.load_dataset("commonsense_qa_train", **cache_key_params)
    
    if processed_dataset is None:
        print("正在加载CommonsenseQA训练数据集...")
        
        # 加载CommonsenseQA数据集
        dataset = load_dataset("commonsense_qa")
        train_data = dataset["train"]
        
        # 如果指定了最大样本数，则进行采样
        if max_samples and len(train_data) > max_samples:
            indices = random.sample(range(len(train_data)), max_samples)
            train_data = train_data.select(indices)
        
        # 加载分词器
        tokenizer = load_tokenizer()
        
        # 预处理数据
        print("正在预处理训练数据...")
        processed_dataset = train_data.map(
            lambda examples: preprocess_commonsense_qa(examples, tokenizer),
            batched=True,
            remove_columns=train_data.column_names,
            desc="预处理训练数据"
        )
        
        # 缓存处理后的数据集
        cache_manager.cache_dataset(processed_dataset, "commonsense_qa_train", **cache_key_params)
    else:
        tokenizer = load_tokenizer()
    
    return processed_dataset, tokenizer


def get_in_domain_eval_dataset(tokenizer=None, max_samples=1000):
    """
    获取域内评估数据集（CommonsenseQA验证集）
    
    Args:
        tokenizer: 分词器（如果为None则重新加载）
        max_samples: 最大样本数
    
    Returns:
        Dataset: CommonsenseQA验证数据集
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    
    print("正在加载CommonsenseQA验证数据集...")
    
    # 加载CommonsenseQA验证集
    dataset = load_dataset("commonsense_qa")
    validation_data = dataset["validation"]
    
    # 如果指定了最大样本数，则进行采样
    if max_samples and len(validation_data) > max_samples:
        indices = random.sample(range(len(validation_data)), max_samples)
        validation_data = validation_data.select(indices)
    
    # 预处理数据
    print("正在预处理CommonsenseQA验证数据...")
    processed_dataset = validation_data.map(
        lambda examples: preprocess_commonsense_qa(examples, tokenizer),
        batched=True,
        remove_columns=validation_data.column_names,
        desc="预处理CommonsenseQA验证数据"
    )
    
    print(f"CommonsenseQA验证集: {len(processed_dataset)} 样本")
    return processed_dataset


def get_eval_datasets(tokenizer=None, max_samples_per_dataset=1000):
    """
    获取域外评估数据集（支持缓存）
    
    Args:
        tokenizer: 分词器（如果为None则重新加载）
        max_samples_per_dataset: 每个数据集的最大样本数
    
    Returns:
        dict: 包含所有域外评估数据集的字典
    """
    if tokenizer is None:
        tokenizer = load_tokenizer()
    
    cache_manager = get_cache_manager()
    eval_datasets = {}
    
    # 域外评估数据集
    print("正在加载域外评估数据集...")
    
    # 数据集配置：(名称, 加载参数, 预处理函数, 信任远程代码)
    dataset_configs = [
        ("boolq", {"path": "boolq", "split": "validation"}, preprocess_boolq, False),
        ("arc", {"path": "ai2_arc", "name": "ARC-Challenge", "split": "validation"}, preprocess_arc, False),
        ("hellaswag", {"path": "hellaswag", "split": "validation"}, preprocess_hellaswag, True),
        # 其他数据集可以在这里添加
        # ("piqa", {"path": "piqa", "split": "validation"}, preprocess_piqa, True),
        # ("winogrande", {"path": "winogrande", "name": "winogrande_xl", "split": "validation"}, preprocess_winogrande, True),
    ]
    
    for dataset_name, load_params, preprocess_func, trust_remote_code in dataset_configs:
        try:
            print(f"  - 加载{dataset_name.upper()}...")
            
            # 尝试从缓存加载
            cache_key_params = {**load_params, "max_samples": max_samples_per_dataset}
            processed_data = cache_manager.load_dataset(f"{dataset_name}_eval", **cache_key_params)
            
            if processed_data is None:
                # 从数据集库加载
                load_kwargs = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
                
                if "name" in load_params:
                    raw_data = load_dataset(load_params["path"], load_params["name"], 
                                          split=load_params["split"], **load_kwargs)
                else:
                    raw_data = load_dataset(load_params["path"], 
                                          split=load_params["split"], **load_kwargs)
                
                # 处理不同的数据格式
                if isinstance(raw_data, list):
                    # 如果返回的是列表，转换为正确的格式
                    from datasets import Dataset
                    if len(raw_data) > 0 and isinstance(raw_data[0], dict):
                        # 字典列表格式
                        keys = raw_data[0].keys()
                        data_dict = {key: [item[key] for item in raw_data] for key in keys}
                        raw_data = Dataset.from_dict(data_dict)
                
                # 采样
                if len(raw_data) > max_samples_per_dataset:
                    indices = random.sample(range(len(raw_data)), max_samples_per_dataset)
                    raw_data = raw_data.select(indices)
                
                # 预处理
                processed_data = raw_data.map(
                    lambda examples: preprocess_func(examples, tokenizer),
                    batched=True,
                    remove_columns=raw_data.column_names,
                    desc=f"预处理{dataset_name.upper()}"
                )
                
                # 缓存处理后的数据
                cache_manager.cache_dataset(processed_data, f"{dataset_name}_eval", **cache_key_params)
            
            eval_datasets[dataset_name] = processed_data
            
        except Exception as e:
            print(f"    加载{dataset_name.upper()}失败: {e}")
            # 可选：打印详细错误信息用于调试
            import traceback
            print(f"    详细错误: {traceback.format_exc()}")
    
    print(f"\n成功加载 {len(eval_datasets)} 个评估数据集")
    for name, dataset in eval_datasets.items():
        print(f"  {name}: {len(dataset)} 样本")
    
    return eval_datasets


# 向后兼容的函数
def process_dataset():
    """
    向后兼容的数据处理函数（用于旧代码）
    
    Returns:
        tuple: (train_dataset, validation_dataset, tokenizer)
    """
    print("警告: process_dataset()已弃用，请使用get_train_data()和get_eval_datasets()")
    
    # 获取训练数据
    train_dataset, tokenizer = get_train_data(max_samples=5000)  # 限制样本数量用于快速测试
    
    # 获取一个验证数据集（BoolQ）
    eval_datasets = get_eval_datasets(tokenizer, max_samples_per_dataset=500)
    validation_dataset = eval_datasets.get("boolq", None)
    
    if validation_dataset is None:
        raise ValueError("无法加载验证数据集")
    
    return train_dataset, validation_dataset, tokenizer


def main():
    """测试函数"""
    print("=" * 60)
    print("多数据集处理测试")
    print("=" * 60)
    
    # 测试训练数据加载
    print("\n1. 测试训练数据加载 (CommonsenseQA)")
    print("-" * 40)
    train_dataset, tokenizer = get_train_data(max_samples=100)
    print(f"训练集样本数量: {len(train_dataset)}")
    
    # 打印训练样本示例
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"训练样本示例:")
        print(f"  Input IDs 长度: {len(sample['input_ids'])}")
        print(f"  Labels: {sample['labels']}")
        decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"  解码文本: '{decoded_text[:150]}...'")
    
    # 测试评估数据加载
    print("\n2. 测试评估数据加载")
    print("-" * 40)
    eval_datasets = get_eval_datasets(tokenizer, max_samples_per_dataset=50)
    
    # 打印每个评估数据集的信息
    for name, dataset in eval_datasets.items():
        print(f"\n{name.upper()} 数据集:")
        print(f"  样本数量: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
            print(f"  样本示例: '{decoded_text[:100]}...'")
            print(f"  标签: {sample['labels']}")
    
    print("\n" + "=" * 60)
    print("数据处理测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()