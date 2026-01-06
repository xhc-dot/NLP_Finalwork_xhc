import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import random
import time
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(42)

class FraudBERTDataset(Dataset):
    """诈骗对话数据集"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTFraudClassifier(nn.Module):
    """BERT诈骗分类器"""
    def __init__(self, n_classes=2, dropout_prob=0.3):
        super(BERTFraudClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

def train_model_optimized(model, train_loader, val_loader, config):
    """优化的训练函数 - 保存最佳模型"""
    print("\n" + "="*60)
    print("开始训练模型...")
    print("="*60)
    
    device = torch.device('cpu')
    model = model.to(device)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10%的warmup
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-"*40)
        
        # 训练阶段
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        # 使用tqdm进度条
        train_iterator = tqdm(train_loader, desc="训练", unit="batch", leave=False)
        
        start_time = time.time()
        
        for batch in train_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # 统计
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
            
            # 更新进度条
            train_iterator.set_postfix({'loss': loss.item()})
        
        # 验证阶段
        val_accuracy = evaluate_model_quick(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        
        epoch_time = time.time() - start_time
        epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        print(f"训练损失: {epoch_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f} ({val_accuracy:.2%})")
        print(f"时间: {epoch_time:.1f}秒")
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model_temp.pth')
            print(f"✅ 保存新的最佳模型 (准确率: {val_accuracy:.2%})")
    
    # 加载最佳模型
    print("\n加载训练过程中的最佳模型...")
    model.load_state_dict(torch.load('best_model_temp.pth', map_location=device))
    
    # 将临时模型保存为最终模型
    final_model_path = 'bert_fraud_model_final.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"✅ 最终模型已保存至: {final_model_path}")
    
    # 清理临时文件
    if os.path.exists('best_model_temp.pth'):
        os.remove('best_model_temp.pth')
        print("临时模型文件已清理")
    
    return model, train_losses, val_accuracies, final_model_path

def evaluate_model_quick(model, data_loader, device):
    """快速评估函数"""
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            correct_predictions += torch.sum(preds == labels)
            total_samples += labels.size(0)
    
    return (correct_predictions.double() / total_samples).item()

def main_optimized():
    """优化的主函数 - 包含模型保存"""
    print("="*70)
    print("BERT诈骗分类模型训练与测试（包含模型保存）")
    print("="*70)
    
    # 配置参数
    config = {
        'max_length': 128,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'dropout_rate': 0.3,
        'epochs': 2,
    }
    
    # 文件路径
    train_path = 'D:\\自然语言处理\\实验一\\train_data.csv'
    test_path = 'D:\\自然语言处理\\实验一\\test_data.csv'
    
    try:
        start_time = time.time()
        
        # 1. 加载数据
        print("\n[1/6] 加载数据...")
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # 清理数据
        train_data = train_data[train_data['specific_dialogue_content'].notna() & 
                               (train_data['specific_dialogue_content'] != '')]
        train_data = train_data[train_data['is_fraud'].notna()]
        
        test_data = test_data[test_data['specific_dialogue_content'].notna() & 
                             (test_data['specific_dialogue_content'] != '')]
        test_data = test_data[test_data['is_fraud'].notna()]
        
        train_texts = train_data['specific_dialogue_content'].astype(str).tolist()
        train_labels = train_data['is_fraud'].astype(int).tolist()
        
        test_texts = test_data['specific_dialogue_content'].astype(str).tolist()
        test_labels = test_data['is_fraud'].astype(int).tolist()
        
        print(f"训练集: {len(train_texts)} 条")
        print(f"测试集: {len(test_texts)} 条")
        
        # 2. 创建验证集
        print("\n[2/6] 划分训练集/验证集...")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
        
        # 3. 初始化tokenizer
        print("\n[3/6] 初始化BERT...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 4. 创建数据加载器
        print("\n[4/6] 创建数据加载器...")
        train_dataset = FraudBERTDataset(train_texts, train_labels, tokenizer, config['max_length'])
        val_dataset = FraudBERTDataset(val_texts, val_labels, tokenizer, config['max_length'])
        test_dataset = FraudBERTDataset(test_texts, test_labels, tokenizer, config['max_length'])
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        
        print(f"训练批次: {len(train_loader)}")
        print(f"验证批次: {len(val_loader)}")
        print(f"测试批次: {len(test_loader)}")
        
        # 5. 初始化模型
        print("\n[5/6] 初始化模型...")
        model = BERTFraudClassifier(n_classes=2, dropout_prob=config['dropout_rate'])
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 6. 训练模型
        model, train_losses, val_accuracies, model_path = train_model_optimized(
            model, train_loader, val_loader, config
        )
        
        # 7. 最终测试
        print("\n[7/6] 最终测试...")
        device = torch.device('cpu')
        
        predictions, true_labels, probabilities = [], [], []
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        
        print("\n" + "="*60)
        print("测试结果汇总")
        print("="*60)
        print(f"最终测试准确率: {accuracy:.4f} ({accuracy:.2%})")
        print("\n分类报告:")
        print(classification_report(true_labels, predictions, target_names=['非诈骗', '诈骗']))
        
        # 计算混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        print("混淆矩阵:")
        print(f"True Negative (非诈骗->非诈骗): {cm[0, 0]}")
        print(f"False Positive (非诈骗->诈骗): {cm[0, 1]}")
        print(f"False Negative (诈骗->非诈骗): {cm[1, 0]}")
        print(f"True Positive (诈骗->诈骗): {cm[1, 1]}")
        
        # 保存预测结果
        results_df = pd.DataFrame({
            'text': test_texts,
            'true_label': true_labels,
            'predicted_label': predictions,
            'prob_non_fraud': [p[0] for p in probabilities],
            'prob_fraud': [p[1] for p in probabilities],
            'is_correct': [1 if true == pred else 0 for true, pred in zip(true_labels, predictions)]
        })
        
        results_df.to_csv('bert_results_fast.csv', index=False, encoding='utf-8-sig')
        print(f"\n✅ 预测结果已保存至: bert_results_fast.csv")
        
        # 8. 保存完整模型和tokenizer
        print("\n[8/6] 保存模型和tokenizer...")
        
        # 保存完整模型（包含结构）
        full_model_path = 'bert_fraud_full_model.pth'
        torch.save(model, full_model_path)
        print(f"✅ 完整模型已保存至: {full_model_path}")
        
        # 保存tokenizer
        tokenizer_save_dir = './bert_tokenizer/'
        tokenizer.save_pretrained(tokenizer_save_dir)
        print(f"✅ Tokenizer已保存至: {tokenizer_save_dir}")
        
        # 保存模型配置信息
        config_info = {
            'model_path': model_path,
            'full_model_path': full_model_path,
            'tokenizer_dir': tokenizer_save_dir,
            'accuracy': accuracy,
            'test_samples': len(test_texts),
            'config': config,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }
        
        import json
        with open('model_config.json', 'w', encoding='utf-8') as f:
            json.dump(config_info, f, ensure_ascii=False, indent=2)
        print(f"✅ 模型配置已保存至: model_config.json")
        
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print(f"总时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        print(f"模型文件: {model_path}")
        print(f"测试准确率: {accuracy:.4f} ({accuracy:.2%})")
        print("训练完成!")
        print("="*60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def load_and_test_only(model_path='bert_fraud_model_final.pth', test_file=None):
    """只加载已有模型进行测试，不重新训练"""
    print("="*70)
    print("加载已有模型进行测试")
    print("="*70)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        print("请先运行训练代码或检查文件路径")
        return
    
    # 使用默认测试文件或指定文件
    if test_file is None:
        test_file = 'D:\\自然语言处理\\实验一\\test_data.csv'
    
    try:
        # 1. 加载tokenizer
        print("\n1. 加载tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 2. 加载模型
        print("\n2. 加载模型...")
        model = BERTFraudClassifier(n_classes=2, dropout_prob=0.3)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ 模型加载成功: {model_path}")
        
        # 3. 加载测试数据
        print("\n3. 加载测试数据...")
        test_data = pd.read_csv(test_file)
        
        # 清理数据
        test_data = test_data[test_data['specific_dialogue_content'].notna() & 
                             (test_data['specific_dialogue_content'] != '')]
        test_data = test_data[test_data['is_fraud'].notna()]
        
        test_texts = test_data['specific_dialogue_content'].astype(str).tolist()
        test_labels = test_data['is_fraud'].astype(int).tolist()
        
        print(f"测试集大小: {len(test_texts)}")
        
        # 4. 创建数据加载器
        print("\n4. 创建数据加载器...")
        test_dataset = FraudBERTDataset(test_texts, test_labels, tokenizer, max_len=128)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 5. 测试
        print("\n5. 进行测试...")
        device = torch.device('cpu')
        model.to(device)
        
        predictions, true_labels, probabilities = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # 6. 评估
        accuracy = accuracy_score(true_labels, predictions)
        print(f"\n" + "="*60)
        print(f"测试准确率: {accuracy:.4f} ({accuracy:.2%})")
        print("\n分类报告:")
        print(classification_report(true_labels, predictions, target_names=['非诈骗', '诈骗']))
        
        # 7. 保存结果
        results_df = pd.DataFrame({
            'text': test_texts,
            'true_label': true_labels,
            'predicted_label': predictions,
            'prob_non_fraud': [p[0] for p in probabilities],
            'prob_fraud': [p[1] for p in probabilities],
            'is_correct': [1 if true == pred else 0 for true, pred in zip(true_labels, predictions)]
        })
        
        output_file = 'test_results_only.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ 测试结果已保存至: {output_file}")
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

def fast_test_solution():
    """超快速测试方案 - 先验证代码能运行"""
    print("="*70)
    print("快速测试方案（先验证代码能运行）")
    print("="*70)
    
    config = {
        'max_length': 64,   # 减少序列长度
        'batch_size': 32,   # 增大batch_size
        'learning_rate': 2e-5,
        'epochs': 1,        # 只训练1轮
    }
    
    # 只加载少量数据
    train_data = pd.read_csv('D:\\自然语言处理\\实验一\\train_data.csv')
    
    # ========== 修复核心：增加数据清洗 ==========
    # 1. 过滤空值行（关键修复）
    train_data = train_data[
        train_data['specific_dialogue_content'].notna() &  # 对话内容非空
        (train_data['specific_dialogue_content'] != '') &  # 对话内容不是空字符串
        train_data['is_fraud'].notna()  # 标签非空
    ]
    
    # 2. 只取500条数据（如果数据不足500条则取全部）
    train_data = train_data.sample(min(500, len(train_data)), random_state=42)  
    
    # 3. 确保标签是整数类型（双重保险）
    train_data['is_fraud'] = train_data['is_fraud'].astype(float).astype(int)
    
    train_texts = train_data['specific_dialogue_content'].astype(str).tolist()
    train_labels = train_data['is_fraud'].tolist()
    
    print(f"使用 {len(train_texts)} 条有效数据进行快速测试...")
    
    # 快速训练和测试
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    dataset = FraudBERTDataset(train_texts, train_labels, tokenizer, config['max_length'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    model = BERTFraudClassifier(n_classes=2)
    
    # 简单训练1个epoch
    device = torch.device('cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    model.train()
    for batch in tqdm(loader, desc="快速训练"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 保存快速测试的模型
    torch.save(model.state_dict(), 'bert_model_quick_test.pth')
    print("快速测试完成！")
    print("模型已保存为: bert_model_quick_test.pth")
    print("如果这个能运行，再使用完整数据训练。")

if __name__ == "__main__":
    print("="*70)
    print("BERT诈骗分类模型")
    print("="*70)
    print("请选择运行模式：")
    print("1. 快速测试（先验证代码能运行）")
    print("2. 完整训练（训练并保存模型）")
    print("3. 只测试（加载已有模型测试）")
    print("4. 测试对抗性样本")
    
    choice = input("请输入选择 (1/2/3/4): ").strip()
    
    if choice == "1":
        fast_test_solution()
    elif choice == "2":
        main_optimized()  # 完整训练
    elif choice == "3":
        # 只测试现有模型
        model_file = input("请输入模型文件路径 (直接回车使用默认): ").strip()
        if not model_file:
            model_file = 'bert_fraud_model_final.pth'
        load_and_test_only(model_path=model_file)
    elif choice == "4":
        print("请运行独立的对抗性样本测试脚本")
        print("创建 BERT_test_adversarial.py 文件进行测试")
    else:
        print("默认运行完整训练")
        main_optimized()