import pandas as pd
import jieba
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

def load_model_pipeline(model_path, vectorizer_path):
    """
    加载完整的模型流水线
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ 模型和向量化器加载完成！")
    return model, vectorizer

def preprocess_and_predict(texts, model, vectorizer):
    """
    预处理文本并进行预测
    """
    # 预处理文本
    texts_tokens = [' '.join(jieba.cut(str(text))) for text in texts]
    
    # 向量化
    texts_vectorized = vectorizer.transform(texts_tokens)
    
    # 预测
    predictions = model.predict(texts_vectorized)
    probabilities = model.predict_proba(texts_vectorized)
    
    return predictions, probabilities

def predict_adversarial_dialogues(data_file, model, vectorizer):
    """
    预测对抗性样本的分类结果，并计算ASR
    """
    # 读取数据
    data = pd.read_csv(data_file)
    
    # 检查必要的列是否存在
    required_columns = ['original_dialogue', 'adversarial_dialogue', 'true_label', 'generation_success']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"❌ 数据文件中缺少以下必要列: {missing_columns}")
        print(f"可用列: {list(data.columns)}")
        return None
    
    # 打印原始数据信息
    print(f"原始数据行数: {len(data)}")
    print(f"true_label列数据类型: {data['true_label'].dtype}")
    print(f"true_label列唯一值: {data['true_label'].unique()}")
    print(f"true_label列缺失值数量: {data['true_label'].isna().sum()}")
    print(f"generation_success列统计:")
    print(data['generation_success'].value_counts(dropna=False))
    
    # 处理缺失值和空字符串
    print("\n清理数据...")
    data_clean = data.copy()
    
    # 删除 original_dialogue 为空的行
    data_clean = data_clean[data_clean['original_dialogue'].notna() & (data_clean['original_dialogue'] != '')]
    
    # 删除 adversarial_dialogue 为空的行
    data_clean = data_clean[data_clean['adversarial_dialogue'].notna() & (data_clean['adversarial_dialogue'] != '')]
    
    # 删除 true_label 为空的行
    data_clean = data_clean[data_clean['true_label'].notna()]
    
    print(f"清理后数据行数: {len(data_clean)}")
    
    if len(data_clean) == 0:
        print("❌ 没有有效的数据用于预测")
        return None
    
    # 提取文本和真实标签
    X_original = data_clean['original_dialogue']
    X_adversarial = data_clean['adversarial_dialogue']
    y_true = data_clean['true_label']
    generation_success = data_clean['generation_success']
    
    # 将标签转换为整数（1表示诈骗，0表示非诈骗）
    def convert_to_int(value):
        if isinstance(value, bool):
            return 1 if value else 0
        elif isinstance(value, str):
            value_str = value.strip().lower()
            if value_str in ['true', '1', 't', '是', '诈骗', '欺诈']:
                return 1
            elif value_str in ['false', '0', 'f', '否', '非诈骗', '正常']:
                return 0
        elif isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return int(value)
        return None
    
    # 应用转换函数
    y_true_int_list = []
    valid_indices = []
    
    for idx, value in enumerate(y_true):
        converted = convert_to_int(value)
        if converted is not None:
            y_true_int_list.append(converted)
            valid_indices.append(idx)
    
    # 只保留有效的数据
    X_original_valid = X_original.iloc[valid_indices]
    X_adversarial_valid = X_adversarial.iloc[valid_indices]
    y_true_int = np.array(y_true_int_list)
    generation_success_valid = generation_success.iloc[valid_indices]
    
    print(f"\n有效样本数: {len(X_original_valid)}")
    print(f"诈骗样本数 (标签1): {np.sum(y_true_int)}")
    print(f"非诈骗样本数 (标签0): {len(y_true_int) - np.sum(y_true_int)}")
    
    if len(y_true_int) == 0:
        print("❌ 没有有效的标签数据用于预测")
        return None
    
    # 1. 首先对原始对话进行预测
    print("\n" + "=" * 60)
    print("步骤1: 对原始对话进行预测")
    print("=" * 60)
    
    original_predictions, original_probabilities = preprocess_and_predict(X_original_valid, model, vectorizer)
    
    # 计算原始预测的准确率
    original_accuracy = accuracy_score(y_true_int, original_predictions)
    print(f"原始对话预测准确率: {original_accuracy:.4f} ({original_accuracy:.2%})")
    
    # 2. 识别原始预测正确的样本
    original_correct_mask = (original_predictions == y_true_int)
    original_correct_count = np.sum(original_correct_mask)
    print(f"原始预测正确的样本数: {original_correct_count} (占比: {original_correct_count/len(y_true_int):.2%})")
    
    # 3. 对对抗性样本进行预测
    print("\n" + "=" * 60)
    print("步骤2: 对对抗性样本进行预测")
    print("=" * 60)
    
    adversarial_predictions, adversarial_probabilities = preprocess_and_predict(X_adversarial_valid, model, vectorizer)
    
    # 计算对抗性样本的预测准确率
    adversarial_accuracy = accuracy_score(y_true_int, adversarial_predictions)
    print(f"对抗性样本预测准确率: {adversarial_accuracy:.4f} ({adversarial_accuracy:.2%})")
    
    # 4. 计算ASR（攻击成功率）- 修改为：攻击成功样本数/原始预测正确样本数
    print("\n" + "=" * 60)
    print("步骤3: 计算攻击成功率 (ASR)")
    print("=" * 60)
    
    # 将generation_success转换为布尔值
    def convert_to_bool(value):
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            value_str = value.strip().lower()
            return value_str in ['true', '1', 't', '是', '成功', 'yes', 'y']
        elif isinstance(value, (int, float)):
            return bool(value)
        return False
    
    generation_success_bool = np.array([convert_to_bool(val) for val in generation_success_valid])
    print(f"成功生成的对抗样本数: {np.sum(generation_success_bool)} (占比: {np.sum(generation_success_bool)/len(generation_success_bool):.2%})")
    
    # 计算攻击成功的样本数：原始正确 AND 生成成功 AND 对抗错误
    attack_success_mask = (
        original_correct_mask & 
        generation_success_bool & 
        (adversarial_predictions != y_true_int)
    )
    
    attack_success_count = np.sum(attack_success_mask)
    
    print(f"\nASR计算公式:")
    print(f"分子（攻击成功）: 原始正确 ∩ 生成成功 ∩ 对抗错误 = {attack_success_count}")
    print(f"分母（原始预测正确）: 原始预测正确的样本数 = {original_correct_count}")
    
    if original_correct_count > 0:
        asr = attack_success_count / original_correct_count
        print(f"攻击成功率 (ASR): {asr:.4f} ({asr:.2%})")
        print(f"攻击成功比例: {attack_success_count}/{original_correct_count}")
    else:
        asr = np.nan
        print(f"❌ 没有原始预测正确的样本，无法计算ASR")
    
    # 计算各类别的ASR
    print(f"\n按类别统计:")
    for label_value, label_name in [(0, '非诈骗'), (1, '诈骗')]:
        # 筛选该类别的样本
        label_mask = (y_true_int == label_value)
        
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
    
    # 5. 计算其他相关指标
    print("\n" + "=" * 60)
    print("其他相关指标")
    print("=" * 60)
    
    # 对抗性样本的整体准确率
    adversarial_correct_mask = (adversarial_predictions == y_true_int)
    print(f"对抗性样本正确预测数: {np.sum(adversarial_correct_mask)}")
    
    # 生成成功对抗样本的准确率
    if np.sum(generation_success_bool) > 0:
        adversarial_accuracy_success = accuracy_score(
            y_true_int[generation_success_bool], 
            adversarial_predictions[generation_success_bool]
        )
        print(f"生成成功对抗样本的准确率: {adversarial_accuracy_success:.4f} ({adversarial_accuracy_success:.2%})")
    
    # 生成失败对抗样本的准确率（如果有）
    generation_failed_bool = ~generation_success_bool
    if np.sum(generation_failed_bool) > 0:
        adversarial_accuracy_failed = accuracy_score(
            y_true_int[generation_failed_bool], 
            adversarial_predictions[generation_failed_bool]
        )
        print(f"生成失败对抗样本的准确率: {adversarial_accuracy_failed:.4f} ({adversarial_accuracy_failed:.2%})")
    
    # 原始预测正确且生成成功的样本统计
    original_correct_gen_success_mask = original_correct_mask & generation_success_bool
    original_correct_gen_success_count = np.sum(original_correct_gen_success_mask)
    print(f"原始预测正确且生成成功的样本数: {original_correct_gen_success_count}")
    
    # 6. 完整的性能评估
    print("\n" + "=" * 60)
    print("对抗性样本完整分类结果评估")
    print("=" * 60)
    
    # 计算对抗性样本的准确率
    accuracy = accuracy_score(y_true_int, adversarial_predictions)
    
    # 诈骗类样本（标签1）的准确率
    fraud_mask = y_true_int == 1
    if np.sum(fraud_mask) > 0:
        accuracy_fraud = accuracy_score(y_true_int[fraud_mask], adversarial_predictions[fraud_mask])
    else:
        accuracy_fraud = np.nan
    
    # 非诈骗类样本（标签0）的准确率
    nonfraud_mask = y_true_int == 0
    if np.sum(nonfraud_mask) > 0:
        accuracy_nonfraud = accuracy_score(y_true_int[nonfraud_mask], adversarial_predictions[nonfraud_mask])
    else:
        accuracy_nonfraud = np.nan
    
    print(f"对抗性样本总准确率: {accuracy:.4f} ({accuracy:.2%})")
    print(f"对抗性样本诈骗类准确率: {accuracy_fraud:.4f} ({accuracy_fraud:.2%})")
    print(f"对抗性样本非诈骗类准确率: {accuracy_nonfraud:.4f} ({accuracy_nonfraud:.2%})")
    
    # 详细的分类报告
    print("\n对抗性样本分类报告:")
    print("-" * 60)
    print(classification_report(y_true_int, adversarial_predictions, target_names=['非诈骗', '诈骗']))
    
    # 混淆矩阵
    print("对抗性样本混淆矩阵:")
    print("-" * 60)
    cm = confusion_matrix(y_true_int, adversarial_predictions)
    print(f"True Negative (非诈骗->非诈骗): {cm[0, 0]}")
    print(f"False Positive (非诈骗->诈骗): {cm[0, 1]}")
    print(f"False Negative (诈骗->非诈骗): {cm[1, 0]}")
    print(f"True Positive (诈骗->诈骗): {cm[1, 1]}")
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'original_dialogue': X_original_valid.reset_index(drop=True),
        'adversarial_dialogue': X_adversarial_valid.reset_index(drop=True),
        'true_label': y_true_int,
        'generation_success': generation_success_valid.reset_index(drop=True),
        'generation_success_bool': generation_success_bool,
        'original_prediction': original_predictions,
        'original_prob_non_fraud': original_probabilities[:, 0],
        'original_prob_fraud': original_probabilities[:, 1],
        'original_is_correct': original_correct_mask.astype(int),
        'adversarial_prediction': adversarial_predictions,
        'adversarial_prob_non_fraud': adversarial_probabilities[:, 0],
        'adversarial_prob_fraud': adversarial_probabilities[:, 1],
        'adversarial_is_correct': adversarial_correct_mask.astype(int),
        'attack_success': attack_success_mask.astype(int)
    })
    
    # 显示攻击成功的样本
    attack_success_df = result_df[result_df['attack_success'] == 1]
    if len(attack_success_df) > 0:
        print(f"\n攻击成功的样本 (前5个):")
        for idx, row in attack_success_df.head(5).iterrows():
            print(f"\n样本 {idx}:")
            print(f"真实标签: {'诈骗' if row['true_label'] == 1 else '非诈骗'}")
            print(f"原始预测: {'诈骗' if row['original_prediction'] == 1 else '非诈骗'} (正确)")
            print(f"对抗预测: {'诈骗' if row['adversarial_prediction'] == 1 else '非诈骗'} (错误)")
            print(f"生成成功: {row['generation_success']}")
            print(f"原始对话: {str(row['original_dialogue'])[:80]}...")
            print(f"对抗对话: {str(row['adversarial_dialogue'])[:80]}...")
    
    return result_df, accuracy, accuracy_fraud, accuracy_nonfraud, original_accuracy, asr, original_correct_count, attack_success_count

def main():
    # 模型文件路径
    model_path = 'D:\\自然语言处理\\实验一\\svm_fraud_model.pkl'
    vectorizer_path = 'D:\\自然语言处理\\实验一\\tfidf_vectorizer.pkl'
    
    # 数据文件路径 - 更新为你的文件路径
    data_file = 'D:\\自然语言处理\\期末大作业\\promptattack_C1.csv'
    
    try:
        # 加载模型
        print("正在加载模型...")
        model, vectorizer = load_model_pipeline(model_path, vectorizer_path)
        
        # 预测对抗性样本
        print(f"\n正在处理文件: {data_file}")
        results = predict_adversarial_dialogues(data_file, model, vectorizer)
        
        if results is not None:
            result_df, accuracy, accuracy_fraud, accuracy_nonfraud, original_accuracy, asr, original_correct_count, attack_success_count = results
            
            # 保存完整的预测结果
            output_path = 'adversarial_dialogue_predictions_with_asr_original_correct.csv'
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n✅ 预测结果已保存至: {output_path}")
            
            # 输出总结报告
            print("\n" + "=" * 60)
            print("对抗性攻击分析总结报告")
            print("=" * 60)
            print(f"测试文件: {data_file}")
            print(f"有效样本数: {len(result_df)}")
            print(f"原始对话预测准确率: {original_accuracy:.4f} ({original_accuracy:.2%})")
            print(f"对抗性样本总准确率: {accuracy:.4f} ({accuracy:.2%})")
            
            # 计算新的ASR：攻击成功样本数/原始预测正确样本数
            if original_correct_count > 0:
                new_asr = attack_success_count / original_correct_count
                
                print(f"\n【攻击成功率 (ASR) - 新计算方式】")
                print(f"攻击成功样本数: {attack_success_count}")
                print(f"原始预测正确样本数: {original_correct_count}")
                print(f"ASR = 攻击成功样本数 / 原始预测正确样本数")
                print(f"ASR = {attack_success_count} / {original_correct_count} = {new_asr:.4f} ({new_asr:.2%})")
                
                # 计算防御成功率
                defense_success_rate = 1 - new_asr
                print(f"防御成功率: {defense_success_rate:.4f} ({defense_success_rate:.2%})")
            else:
                print(f"\n⚠️ 无法计算ASR：原始预测正确样本数为0")
                new_asr = np.nan
            
            print(f"\n【详细统计】")
            print(f"诈骗类准确率: {accuracy_fraud:.4f} ({accuracy_fraud:.2%})")
            print(f"非诈骗类准确率: {accuracy_nonfraud:.4f} ({accuracy_nonfraud:.2%})")
            print(f"原始预测正确样本数: {result_df['original_is_correct'].sum()}")
            print(f"对抗性预测正确样本数: {result_df['adversarial_is_correct'].sum()}")
            print(f"成功生成的对抗样本数: {result_df['generation_success_bool'].sum()}")
            print(f"攻击成功样本数: {result_df['attack_success'].sum()}")
            
            # 按类别统计ASR
            print(f"\n【按类别统计ASR】")
            fraud_df = result_df[result_df['true_label'] == 1]
            non_fraud_df = result_df[result_df['true_label'] == 0]
            
            if len(fraud_df) > 0:
                fraud_original_correct = fraud_df['original_is_correct'].sum()
                fraud_attack_success = fraud_df['attack_success'].sum()
                if fraud_original_correct > 0:
                    fraud_asr = fraud_attack_success / fraud_original_correct
                    print(f"诈骗类ASR: {fraud_asr:.4f} ({fraud_asr:.2%}) [{fraud_attack_success}/{fraud_original_correct}]")
                else:
                    print(f"诈骗类ASR: 无法计算（原始预测正确样本数为0）")
            
            if len(non_fraud_df) > 0:
                non_fraud_original_correct = non_fraud_df['original_is_correct'].sum()
                non_fraud_attack_success = non_fraud_df['attack_success'].sum()
                if non_fraud_original_correct > 0:
                    non_fraud_asr = non_fraud_attack_success / non_fraud_original_correct
                    print(f"非诈骗类ASR: {non_fraud_asr:.4f} ({non_fraud_asr:.2%}) [{non_fraud_attack_success}/{non_fraud_original_correct}]")
                else:
                    print(f"非诈骗类ASR: 无法计算（原始预测正确样本数为0）")
            
            # 计算其他统计指标
            total_samples = len(result_df)
            generation_success_count = result_df['generation_success_bool'].sum()
            original_correct_gen_success = result_df[(result_df['original_is_correct'] == 1) & (result_df['generation_success_bool'] == True)].shape[0]
            
            print(f"\n【其他统计指标】")
            print(f"总样本数: {total_samples}")
            print(f"生成成功样本数: {generation_success_count} ({generation_success_count/total_samples:.2%})")
            print(f"原始正确且生成成功样本数: {original_correct_gen_success}")
            print(f"原始正确且生成成功样本中攻击成功的比例: {attack_success_count}/{original_correct_gen_success} = {attack_success_count/original_correct_gen_success if original_correct_gen_success > 0 else 0:.2%}")
            
            # 保存统计摘要
            summary = {
                'total_samples': total_samples,
                'original_correct_count': original_correct_count,
                'attack_success_count': attack_success_count,
                'asr': new_asr,
                'original_accuracy': original_accuracy,
                'adversarial_accuracy': accuracy,
                'defense_success_rate': defense_success_rate if not np.isnan(new_asr) else np.nan,
                'generation_success_count': generation_success_count,
                'adversarial_correct_count': result_df['adversarial_is_correct'].sum(),
                'original_correct_gen_success_count': original_correct_gen_success
            }
            
            summary_df = pd.DataFrame([summary])
            summary_path = 'asr_summary_original_correct.csv'
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f"\n✅ ASR统计摘要已保存至: {summary_path}")
            
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print("请确保模型文件和数据文件存在于指定路径")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()