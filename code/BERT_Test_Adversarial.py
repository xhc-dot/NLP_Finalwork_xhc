# BERT_test_adversarial.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 配置文件路径 - 请根据需要修改这些路径
# ============================================

# 模型文件路径（选择您训练好的模型）
MODEL_PATH = 'D:\\自然语言处理\\期末大作业\\代码\\bert_fraud_model_final.pth'  # 可以改为您的模型路径

# 对抗性样本文件路径（您需要测试的文件）
ADVERSARIAL_FILE = 'D:\\自然语言处理\\期末大作业\\promptattack_C1.csv'  # 改为您的对抗性样本文件路径

# 输出结果文件路径
OUTPUT_FILE = 'D:\\自然语言处理\\期末大作业\\bert_adversarial_results.csv'  # 测试结果保存路径

# 批量测试时的结果汇总文件
BATCH_SUMMARY_FILE = 'D:\\自然语言处理\\期末大作业\\bert_adversarial_batch_summary.csv'

# ============================================
# 模型和数据处理代码
# ============================================

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

def load_model(model_path, device='cpu'):
    """加载训练好的BERT模型"""
    print(f"正在加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        # 尝试查找其他可能的模型文件
        possible_paths = [
            'bert_fraud_model_final.pth',
            'bert_fraud_full_model.pth',
            'bert_model_quick_test.pth',
            './models/bert_fraud_model_final.pth',
            './saved_models/bert_fraud_model_final.pth'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"使用找到的模型文件: {model_path}")
                break
        else:
            raise FileNotFoundError("找不到任何模型文件！请先训练模型。")
    
    # 初始化模型
    model = BERTFraudClassifier(n_classes=2, dropout_prob=0.3)
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ 模型权重加载成功")
    except:
        # 如果是完整模型文件（包含结构）
        try:
            model = torch.load(model_path, map_location=device)
            print("✅ 完整模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    model.to(device)
    model.eval()
    return model

def convert_labels(labels):
    """将'诈骗'/'非诈骗'转换为0/1"""
    label_map = {'非诈骗': 0, '诈骗': 1}
    numeric_labels = []
    
    for label in labels:
        # 转换为字符串并去除空格
        label_str = str(label).strip()
        
        if label_str in label_map:
            numeric_labels.append(label_map[label_str])
        elif label_str == '0':
            numeric_labels.append(0)
        elif label_str == '1':
            numeric_labels.append(1)
        elif label_str.lower() in ['true', 't', '是', '欺诈']:
            numeric_labels.append(1)
        elif label_str.lower() in ['false', 'f', '否', '正常']:
            numeric_labels.append(0)
        else:
            # 尝试自动识别
            if '诈骗' in label_str or '欺诈' in label_str:
                numeric_labels.append(1)
            elif '非' in label_str or '正常' in label_str or '普通' in label_str:
                numeric_labels.append(0)
            else:
                # 默认设置为非诈骗
                print(f"警告: 无法识别的标签 '{label_str}'，默认为非诈骗")
                numeric_labels.append(0)
    
    return numeric_labels

def convert_generation_success(values):
    """将generation_success转换为布尔值"""
    boolean_values = []
    
    for value in values:
        if isinstance(value, bool):
            boolean_values.append(value)
        elif isinstance(value, str):
            value_str = str(value).strip().lower()
            boolean_values.append(value_str in ['true', '1', 't', '是', '成功', 'yes', 'y'])
        elif isinstance(value, (int, float)):
            boolean_values.append(bool(value))
        else:
            boolean_values.append(False)
    
    return boolean_values

def predict_texts(texts, model, tokenizer, device, batch_size=16):
    """批量预测文本"""
    # 创建数据集
    dummy_labels = [0] * len(texts)  # 创建虚拟标签
    dataset = FraudBERTDataset(texts, dummy_labels, tokenizer, max_len=128)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 进行预测
    predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="预测进度", unit="batch", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return predictions, all_probabilities

def test_single_adversarial_file(adversarial_file, model_path, output_file):
    """
    测试单个对抗性样本文件，计算ASR
    
    Args:
        adversarial_file: 对抗性样本文件路径
        model_path: 模型文件路径
        output_file: 结果保存路径
    """
    print("="*70)
    print(f"测试文件: {adversarial_file}")
    print("="*70)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载模型和tokenizer
        print("\n[1/3] 加载模型和tokenizer...")
        model = load_model(model_path, device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 2. 加载对抗性样本数据
        print(f"\n[2/3] 加载对抗性样本数据: {adversarial_file}")
        adversarial_data = pd.read_csv(adversarial_file)
        
        # 检查必要的列
        required_columns = ['original_dialogue', 'adversarial_dialogue', 'true_label', 'generation_success']
        missing_columns = [col for col in required_columns if col not in adversarial_data.columns]
        
        if missing_columns:
            print(f"警告: CSV文件中缺少列: {missing_columns}")
            print(f"文件中的列: {list(adversarial_data.columns)}")
            
            # 尝试寻找可能的列名
            possible_columns = {
                'original_dialogue': ['original', '原始对话', '原始文本', 'source'],
                'adversarial_dialogue': ['adversarial', '对抗对话', '对抗文本', 'dialogue', 'text', 'content'],
                'true_label': ['label', '真实标签', 'target', '分类', 'is_fraud'],
                'generation_success': ['success', '生成成功', '成功', 'generated']
            }
            
            for req_col, possible_names in possible_columns.items():
                for possible_name in possible_names:
                    if possible_name in adversarial_data.columns:
                        adversarial_data = adversarial_data.rename(columns={possible_name: req_col})
                        print(f"已将列 '{possible_name}' 重命名为 '{req_col}'")
                        break
        
        # 再次检查必要的列
        missing_columns = [col for col in required_columns if col not in adversarial_data.columns]
        if missing_columns:
            raise ValueError(f"CSV文件中缺少必要的列: {missing_columns}")
        
        # 数据清洗
        original_count = len(adversarial_data)
        adversarial_data = adversarial_data.dropna(subset=required_columns)
        adversarial_data = adversarial_data[adversarial_data['original_dialogue'].astype(str) != '']
        adversarial_data = adversarial_data[adversarial_data['adversarial_dialogue'].astype(str) != '']
        cleaned_count = len(adversarial_data)
        
        if cleaned_count < original_count:
            print(f"数据清洗: {original_count} -> {cleaned_count} (移除了{original_count-cleaned_count}条无效数据)")
        
        # 获取文本和标签
        original_texts = adversarial_data['original_dialogue'].astype(str).tolist()
        adversarial_texts = adversarial_data['adversarial_dialogue'].astype(str).tolist()
        original_labels = adversarial_data['true_label'].astype(str).tolist()
        generation_success = adversarial_data['generation_success'].tolist()
        
        # 将标签转换为数值
        true_labels = convert_labels(original_labels)
        
        # 将generation_success转换为布尔值
        generation_success_bool = convert_generation_success(generation_success)
        
        print(f"有效样本数量: {len(original_texts)}")
        print(f"标签分布 - 非诈骗: {true_labels.count(0)}, 诈骗: {true_labels.count(1)}")
        print(f"生成成功样本数: {sum(generation_success_bool)} (占比: {sum(generation_success_bool)/len(generation_success_bool):.2%})")
        
        # 3. 进行预测
        print("\n[3/3] 进行预测和计算ASR...")
        
        # 3.1 预测原始对话
        print("  1. 预测原始对话...")
        original_predictions, original_probabilities = predict_texts(original_texts, model, tokenizer, device)
        
        # 3.2 预测对抗性对话
        print("  2. 预测对抗性对话...")
        adversarial_predictions, adversarial_probabilities = predict_texts(adversarial_texts, model, tokenizer, device)
        
        # 4. 计算ASR（攻击成功率）
        print("\n" + "="*70)
        print("ASR (攻击成功率) 计算")
        print("="*70)
        
        # 转换numpy数组以便计算
        true_labels_np = np.array(true_labels)
        original_predictions_np = np.array(original_predictions)
        adversarial_predictions_np = np.array(adversarial_predictions)
        generation_success_np = np.array(generation_success_bool)
        
        # 计算原始预测正确率
        original_correct_mask = (original_predictions_np == true_labels_np)
        original_accuracy = np.mean(original_correct_mask)
        original_correct_count = np.sum(original_correct_mask)  # 保存原始预测正确样本数
        print(f"原始对话预测准确率: {original_accuracy:.4f} ({original_accuracy:.2%})")
        print(f"原始预测正确样本数: {original_correct_count}")
        
        # 计算对抗性预测准确率
        adversarial_correct_mask = (adversarial_predictions_np == true_labels_np)
        adversarial_accuracy = np.mean(adversarial_correct_mask)
        print(f"\n对抗性对话预测准确率: {adversarial_accuracy:.4f} ({adversarial_accuracy:.2%})")
        print(f"对抗性预测正确样本数: {np.sum(adversarial_correct_mask)}")
        
        # 计算ASR
        # ASR = (原始正确 ∩ 生成成功 ∩ 对抗错误) / (原始正确)
        
        # 计算分子：原始正确且生成成功且对抗错误
        attack_success_mask = (
            original_correct_mask & 
            generation_success_np & 
            (adversarial_predictions_np != true_labels_np)
        )
        attack_success_count = np.sum(attack_success_mask)
        
        print(f"\nASR计算公式:")
        print(f"  分子（攻击成功）: 原始正确 ∩ 生成成功 ∩ 对抗错误 = {attack_success_count}")
        print(f"  分母（原始预测正确）: 原始预测正确的样本数 = {original_correct_count}")
        
        if original_correct_count > 0:
            asr = attack_success_count / original_correct_count
            print(f"\n攻击成功率 (ASR): {asr:.4f} ({asr:.2%})")
            print(f"攻击成功比例: {attack_success_count}/{original_correct_count}")
        else:
            asr = np.nan
            print(f"\n❌ 没有原始预测正确的样本，无法计算ASR")
        
        # 按类别计算ASR
        print(f"\n按类别统计:")
        for label_value, label_name in [(0, '非诈骗'), (1, '诈骗')]:
            label_mask = (true_labels_np == label_value)
            
            if np.sum(label_mask) > 0:
                # 该类别的原始预测正确样本数
                label_original_correct = np.sum(original_correct_mask[label_mask])
                
                # 该类别的攻击成功数
                label_attack_success = np.sum(attack_success_mask[label_mask])
                
                print(f"\n{label_name}类:")
                print(f"  样本数: {np.sum(label_mask)}")
                print(f"  原始预测正确数: {label_original_correct}")
                print(f"  攻击成功数: {label_attack_success}")
                
                if label_original_correct > 0:
                    label_asr = label_attack_success / label_original_correct
                    print(f"  {label_name}类ASR: {label_asr:.4f} ({label_asr:.2%})")
                else:
                    print(f"  无原始预测正确样本，无法计算{label_name}类ASR")
        
        # 5. 详细性能评估
        print("\n" + "="*70)
        print("详细性能评估")
        print("="*70)
        
        # 分类报告
        print("\n对抗性样本分类报告:")
        print("-"*60)
        print(classification_report(true_labels_np, adversarial_predictions_np, 
                                   target_names=['非诈骗', '诈骗'], digits=4))
        
        # 混淆矩阵
        print("\n对抗性样本混淆矩阵:")
        print("-"*60)
        cm = confusion_matrix(true_labels_np, adversarial_predictions_np)
        print(f"True Negative (非诈骗->非诈骗): {cm[0, 0]}")
        print(f"False Positive (非诈骗->诈骗): {cm[0, 1]}")
        print(f"False Negative (诈骗->非诈骗): {cm[1, 0]}")
        print(f"True Positive (诈骗->诈骗): {cm[1, 1]}")
        
        # 计算特异度和敏感度
        if cm[0, 0] + cm[0, 1] > 0:
            tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # 特异度
            print(f"特异度 (非诈骗类正确率): {tnr:.4f} ({tnr:.2%})")
        
        if cm[1, 1] + cm[1, 0] > 0:
            tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # 敏感度/召回率
            print(f"敏感度/召回率 (诈骗类正确率): {tpr:.4f} ({tpr:.2%})")
        
        # 6. 保存结果
        results_df = pd.DataFrame({
            'original_dialogue': original_texts,
            'adversarial_dialogue': adversarial_texts,
            'true_label_original': original_labels,
            'true_label_numeric': true_labels,
            'true_label_text': ['非诈骗' if t == 0 else '诈骗' for t in true_labels],
            'generation_success_original': generation_success,
            'generation_success_bool': generation_success_bool,
            'original_prediction': original_predictions,
            'original_prediction_text': ['非诈骗' if p == 0 else '诈骗' for p in original_predictions],
            'original_prob_non_fraud': [p[0] for p in original_probabilities],
            'original_prob_fraud': [p[1] for p in original_probabilities],
            'original_is_correct': original_correct_mask.astype(int),
            'adversarial_prediction': adversarial_predictions,
            'adversarial_prediction_text': ['非诈骗' if p == 0 else '诈骗' for p in adversarial_predictions],
            'adversarial_prob_non_fraud': [p[0] for p in adversarial_probabilities],
            'adversarial_prob_fraud': [p[1] for p in adversarial_probabilities],
            'adversarial_is_correct': adversarial_correct_mask.astype(int),
            'attack_success': attack_success_mask.astype(int)
        })
        
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 详细结果已保存至: {output_file}")
        
        # 7. 显示结果示例
        print("\n" + "="*70)
        print("攻击成功示例（前3条）")
        print("="*70)
        
        attack_success_df = results_df[results_df['attack_success'] == 1]
        if len(attack_success_df) > 0:
            for i in range(min(3, len(attack_success_df))):
                row = attack_success_df.iloc[i]
                print(f"\n示例 {i+1}:")
                print(f"  真实标签: {row['true_label_text']}")
                print(f"  原始预测: {row['original_prediction_text']} (正确)")
                print(f"  对抗预测: {row['adversarial_prediction_text']} (错误)")
                print(f"  生成成功: {row['generation_success_original']}")
                print(f"  原始对话预览: {row['original_dialogue'][:80]}...")
                print(f"  对抗对话预览: {row['adversarial_dialogue'][:80]}...")
        else:
            print("无攻击成功样本")
        
        # 8. 显示攻击失败示例
        print("\n" + "="*70)
        print("攻击失败示例（防御成功，前3条）")
        print("="*70)
        
        attack_failed_df = results_df[
            (results_df['original_is_correct'] == 1) & 
            (results_df['generation_success_bool'] == True) & 
            (results_df['adversarial_is_correct'] == 1)
        ]
        
        if len(attack_failed_df) > 0:
            for i in range(min(3, len(attack_failed_df))):
                row = attack_failed_df.iloc[i]
                print(f"\n示例 {i+1}:")
                print(f"  真实标签: {row['true_label_text']}")
                print(f"  原始预测: {row['original_prediction_text']} (正确)")
                print(f"  对抗预测: {row['adversarial_prediction_text']} (正确)")
                print(f"  生成成功: {row['generation_success_original']}")
                print(f"  诈骗概率变化: {row['original_prob_fraud']:.4f} -> {row['adversarial_prob_fraud']:.4f}")
        else:
            print("无防御成功样本（或样本不足）")
        
        return {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'asr': asr,
            'attack_success_count': attack_success_count,
            'original_correct_count': original_correct_count,
            'sample_count': len(results_df)
        }
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_test_adversarial_files(adversarial_files, model_path, summary_file):
    """
    批量测试多个对抗性样本文件
    
    Args:
        adversarial_files: 对抗性样本文件路径列表
        model_path: 模型文件路径
        summary_file: 汇总结果保存路径
    """
    print("="*70)
    print("批量测试对抗性样本文件")
    print("="*70)
    
    if not adversarial_files:
        print("没有需要测试的文件")
        return
    
    print(f"找到 {len(adversarial_files)} 个对抗性样本文件:")
    for i, file in enumerate(adversarial_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # 逐个测试
    results = []
    for file_path in adversarial_files:
        print(f"\n{'='*70}")
        print(f"测试文件: {os.path.basename(file_path)}")
        print('='*70)
        
        # 为每个文件生成独立的输出文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        individual_output_file = f"bert_{base_name}_results.csv"
        
        result_dict = test_single_adversarial_file(
            file_path, 
            model_path, 
            individual_output_file
        )
        
        if result_dict is not None:
            results.append({
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'original_accuracy': result_dict['original_accuracy'],
                'original_accuracy_percent': f"{result_dict['original_accuracy']:.2%}",
                'adversarial_accuracy': result_dict['adversarial_accuracy'],
                'adversarial_accuracy_percent': f"{result_dict['adversarial_accuracy']:.2%}",
                'asr': result_dict['asr'],
                'asr_percent': f"{result_dict['asr']:.2%}" if not np.isnan(result_dict['asr']) else 'N/A',
                'attack_success_count': result_dict['attack_success_count'],
                'original_correct_count': result_dict['original_correct_count'],
                'sample_count': result_dict['sample_count'],
                'results_file': individual_output_file
            })
        else:
            results.append({
                'file_name': os.path.basename(file_path),
                'file_path': file_path,
                'error': '测试失败'
            })
    
    # 汇总结果
    if results:
        print("\n" + "="*70)
        print("批量测试结果汇总")
        print("="*70)
        
        successful_results = [r for r in results if 'error' not in r]
        
        if successful_results:
            print("\n各文件结果:")
            for result in successful_results:
                print(f"\n{result['file_name']}:")
                print(f"  样本数: {result['sample_count']}")
                print(f"  原始准确率: {result['original_accuracy_percent']}")
                print(f"  对抗准确率: {result['adversarial_accuracy_percent']}")
                print(f"  ASR: {result['asr_percent']}")
                print(f"  攻击成功数/原始正确数: {result['attack_success_count']}/{result['original_correct_count']}")
            
            # 计算平均指标
            valid_asr_results = [r for r in successful_results if not np.isnan(r['asr'])]
            if valid_asr_results:
                avg_original_accuracy = np.mean([r['original_accuracy'] for r in valid_asr_results])
                avg_adversarial_accuracy = np.mean([r['adversarial_accuracy'] for r in valid_asr_results])
                avg_asr = np.mean([r['asr'] for r in valid_asr_results])
                
                print(f"\n平均指标 (基于{len(valid_asr_results)}个文件):")
                print(f"  平均原始准确率: {avg_original_accuracy:.4f} ({avg_original_accuracy:.2%})")
                print(f"  平均对抗准确率: {avg_adversarial_accuracy:.4f} ({avg_adversarial_accuracy:.2%})")
                print(f"  平均ASR: {avg_asr:.4f} ({avg_asr:.2%})")
                print(f"  平均防御成功率: {1-avg_asr:.4f} ({(1-avg_asr):.2%})")
            
            # 保存汇总结果
            summary_df = pd.DataFrame(successful_results)
            summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ 汇总结果已保存至: {summary_file}")
        else:
            print("没有成功测试的文件")

def main():
    """主函数 - 使用配置文件中的路径"""
    print("BERT诈骗分类模型对抗性样本测试 (包含ASR计算)")
    print("="*70)
    
    # 显示当前配置
    print("当前配置:")
    print(f"  模型文件: {MODEL_PATH}")
    print(f"  对抗性样本文件: {ADVERSARIAL_FILE}")
    print(f"  输出结果文件: {OUTPUT_FILE}")
    print(f"  批量汇总文件: {BATCH_SUMMARY_FILE}")
    
    print("\n请选择测试模式:")
    print("1. 测试单个对抗性样本文件")
    print("2. 批量测试多个对抗性样本文件")
    print("3. 测试当前目录下所有对抗性样本文件")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 测试单个文件（使用配置文件中的路径）
        print(f"\n开始测试单个文件: {ADVERSARIAL_FILE}")
        if os.path.exists(ADVERSARIAL_FILE):
            result_dict = test_single_adversarial_file(ADVERSARIAL_FILE, MODEL_PATH, OUTPUT_FILE)
            
            # 输出详细的ASR计算报告
            if result_dict is not None:
                print(f"\n{'='*70}")
                print("ASR计算详细报告")
                print("="*70)
                print(f"ASR计算公式: ASR = 攻击成功样本数 / 原始预测正确样本数")
                print(f"攻击成功样本数: {result_dict['attack_success_count']}")
                print(f"原始预测正确样本数: {result_dict['original_correct_count']}")
                print(f"ASR = {result_dict['attack_success_count']} / {result_dict['original_correct_count']} = {result_dict['asr']:.4f} ({result_dict['asr']:.2%})")
                
                if result_dict['original_correct_count'] > 0:
                    defense_success_rate = 1 - result_dict['asr']
                    print(f"防御成功率: {defense_success_rate:.4f} ({defense_success_rate:.2%})")
        else:
            print(f"❌ 文件不存在: {ADVERSARIAL_FILE}")
            print("请修改 ADVERSARIAL_FILE 变量为正确的文件路径")
    
    elif choice == "2":
        # 批量测试多个指定文件
        print("\n批量测试模式")
        print("请输入要测试的文件路径（用逗号分隔），或直接回车使用配置文件中的文件:")
        files_input = input("文件路径: ").strip()
        
        if files_input:
            # 用户输入了文件路径
            file_paths = [f.strip() for f in files_input.split(',')]
            # 检查文件是否存在
            valid_files = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    valid_files.append(file_path)
                else:
                    print(f"❌ 文件不存在: {file_path}")
            
            if valid_files:
                batch_test_adversarial_files(valid_files, MODEL_PATH, BATCH_SUMMARY_FILE)
            else:
                print("没有有效的文件路径")
        else:
            # 使用配置文件中的文件
            if os.path.exists(ADVERSARIAL_FILE):
                batch_test_adversarial_files([ADVERSARIAL_FILE], MODEL_PATH, BATCH_SUMMARY_FILE)
            else:
                print(f"❌ 配置文件中的文件不存在: {ADVERSARIAL_FILE}")
    
    elif choice == "3":
        # 测试当前目录下所有对抗性样本文件
        print("\n扫描当前目录下的对抗性样本文件...")
        
        # 查找所有CSV文件
        all_csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        
        if not all_csv_files:
            print("当前目录下没有CSV文件")
            return
        
        # 自动识别对抗性样本文件（根据文件名关键词）
        adversarial_keywords = ['adversarial', 'attack', 'prompt', '对抗', '攻击']
        adversarial_files = []
        
        for csv_file in all_csv_files:
            for keyword in adversarial_keywords:
                if keyword.lower() in csv_file.lower():
                    adversarial_files.append(csv_file)
                    break
        
        if not adversarial_files:
            print("没有找到对抗性样本文件（文件名不包含对抗性关键词）")
            print("当前目录下的CSV文件:")
            for csv_file in all_csv_files:
                print(f"  - {csv_file}")
            
            # 让用户选择
            use_all = input("\n是否使用所有CSV文件进行测试？(y/n): ").strip().lower()
            if use_all == 'y':
                adversarial_files = all_csv_files
        
        if adversarial_files:
            print(f"\n找到 {len(adversarial_files)} 个对抗性样本文件:")
            for file in adversarial_files:
                print(f"  - {file}")
            
            confirm = input("\n是否开始测试？(y/n): ").strip().lower()
            if confirm == 'y':
                batch_test_adversarial_files(adversarial_files, MODEL_PATH, BATCH_SUMMARY_FILE)
        else:
            print("没有文件需要测试")
    
    else:
        print("无效的选择，默认测试单个文件")
        if os.path.exists(ADVERSARIAL_FILE):
            result_dict = test_single_adversarial_file(ADVERSARIAL_FILE, MODEL_PATH, OUTPUT_FILE)
            if result_dict is not None:
                print(f"\nASR计算结果: {result_dict['asr']:.4f} ({result_dict['asr']:.2%})")
        else:
            print(f"❌ 文件不存在: {ADVERSARIAL_FILE}")

# 提供一个简单的测试函数，可以直接调用
def quick_test(file_path=None, model_path=None):
    """
    快速测试函数
    
    Args:
        file_path: 对抗性样本文件路径（可选）
        model_path: 模型文件路径（可选）
    """
    # 使用参数或默认值
    test_file = file_path if file_path else ADVERSARIAL_FILE
    test_model = model_path if model_path else MODEL_PATH
    output_file = f"bert_quick_test_{os.path.basename(test_file)}"
    
    print(f"快速测试: {test_file}")
    print(f"使用模型: {test_model}")
    
    if os.path.exists(test_file):
        result_dict = test_single_adversarial_file(test_file, test_model, output_file)
        if result_dict is not None:
            print(f"\nASR计算结果: {result_dict['asr']:.4f} ({result_dict['asr']:.2%})")
            print(f"攻击成功数/原始正确数: {result_dict['attack_success_count']}/{result_dict['original_correct_count']}")
        return result_dict
    else:
        print(f"❌ 文件不存在: {test_file}")
        return None

if __name__ == "__main__":
    # 如果需要直接运行测试，取消下面的注释并修改文件路径
    # quick_test('your_adversarial_file.csv', 'your_model.pth')
    
    # 否则运行交互式主程序
    main()