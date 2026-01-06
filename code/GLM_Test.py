import pandas as pd
import time
import requests
from typing import List, Dict
import re
import os

# =========================
# 1. 基础配置
# =========================

API_KEY = "a835fc322e094cb98848b40310e16202.GTTeL2vxWikgn8Wd"
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4-air"

INPUT_CSV = "D:\\自然语言处理\\期末大作业\\测试集结果.csv"
OUTPUT_CSV = "D:\\自然语言处理\\期末大作业\\GLM_evaluation_results.csv"

# API错误重试配置
MAX_API_RETRIES = 3  # 最大重试次数
API_RETRY_DELAY = 5  # 重试延迟（秒）
NETWORK_TIMEOUT = 30  # 网络超时时间（秒）

# 任务配置
TASK_DESCRIPTION = "请对以下对话进行诈骗检测，判断是否为诈骗对话。"
TRUE_LABEL_COLUMN = "is_fraud"  # 真实标签列名

# =========================
# 2. 标签标准化函数
# =========================

def normalize_label(label):
    """统一标签格式：返回True(诈骗)或False(非诈骗)"""
    if pd.isna(label):
        return None
    
    # 如果是布尔值
    if isinstance(label, bool):
        return label
    
    # 如果是字符串
    if isinstance(label, str):
        label_lower = label.lower().strip()
        # 识别为诈骗的情况
        if label_lower in ['true', '是', 'yes', '1', '诈骗', '欺诈', '真', '正确']:
            return True
        # 识别为非诈骗的情况  
        elif label_lower in ['false', '否', 'no', '0', '非诈骗', '正常', '非欺诈', '假', '错误']:
            return False
    
    # 如果是数字
    if isinstance(label, (int, float)):
        return bool(label)
    
    return None

def label_to_chinese(label):
    """将标签转换为中文"""
    normalized = normalize_label(label)
    if normalized is None:
        return "unknown"
    return "诈骗" if normalized == True else "非诈骗"

# =========================
# 3. LLM 调用（带重试机制）
# =========================

def chat_completion_with_retry(messages, temperature=0.1, max_tokens=500, max_retries=MAX_API_RETRIES):
    """调用GLM-4 API，带重试机制"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=NETWORK_TIMEOUT)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = API_RETRY_DELAY * (attempt + 1)
                print(f"    API调用失败 ({attempt+1}/{max_retries}): {e}")
                print(f"    {wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"    API调用失败，已达最大重试次数: {e}")
                raise RuntimeError(f"API调用失败: {e}")
        except Exception as e:
            print(f"    API调用异常: {e}")
            if attempt < max_retries - 1:
                wait_time = API_RETRY_DELAY * (attempt + 1)
                print(f"    {wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"API调用异常: {e}")
    
    return None

# =========================
# 4. LLM诈骗检测预测
# =========================

def detect_fraud(dialogue: str) -> str:
    """
    调用LLM对对话进行诈骗检测预测
    返回："诈骗" 或 "非诈骗"
    """
    prompt = f"""{TASK_DESCRIPTION}

对话：
{dialogue}

请判断这个对话是否为诈骗对话，并严格选择以下选项：
A. 诈骗
B. 非诈骗

请只输出字母"A"或"B"，不要输出任何其他文字、标点或解释。"""

    try:
        response = chat_completion_with_retry([
            {"role": "system", "content": "你是诈骗检测助手。严格按照要求只输出'A'或'B'，不输出其他任何内容。"},
            {"role": "user", "content": prompt}
        ], temperature=0.1, max_tokens=5)
        
        if not response:
            print("    API返回空")
            return "error"
        
        # 清理和解析响应
        response_clean = response.strip().upper()
        
        # 直接匹配A或B
        if response_clean == "A":
            return "诈骗"
        elif response_clean == "B":
            return "非诈骗"
        
        # 尝试提取第一个A或B
        match = re.search(r'[AB]', response_clean)
        if match:
            return "诈骗" if match.group() == "A" else "非诈骗"
        
        # 如果还是无法解析，打印原始响应用于调试
        print(f"    无法解析预测结果，原响应: '{response}'")
        return "unknown"
            
    except Exception as e:
        print(f"    API调用失败: {e}")
        return "error"

# =========================
# 5. 处理单个对话
# =========================

def process_single_dialogue(dialogue: str, true_label: bool, sample_id: int) -> Dict:
    """
    处理单个对话：调用LLM预测并与真实标签比较
    """
    # 转换为中文标签
    true_label_chinese = "诈骗" if true_label == True else "非诈骗"
    
    print(f"  处理样本 {sample_id}...")
    print(f"  真实标签: {true_label_chinese}")
    
    # 调用LLM进行预测
    start_time = time.time()
    prediction = detect_fraud(dialogue)
    processing_time = time.time() - start_time
    
    # 判断是否正确
    is_correct = (prediction == true_label_chinese)
    
    result = {
        "sample_id": sample_id,
        "original_dialogue": dialogue[:200] + "..." if len(dialogue) > 200 else dialogue,  # 只保存前200字符
        "true_label": true_label_chinese,
        "prediction": prediction,
        "is_correct": is_correct,
        "processing_time_seconds": round(processing_time, 2),
        "dialogue_length": len(dialogue)
    }
    
    print(f"  模型预测: {prediction}")
    print(f"  预测是否正确: {'✓' if is_correct else '✗'}")
    print(f"  处理时间: {processing_time:.2f}秒")
    
    return result

# =========================
# 6. 批量处理主程序
# =========================

def main():
    """主函数：批量测试模型性能"""
    print("=" * 80)
    print("GLM-4诈骗检测模型性能评估")
    print("=" * 80)
    
    # 读取数据
    try:
        df = pd.read_csv(INPUT_CSV, encoding='gb18030')
        print(f"读取CSV成功，使用gb18030编码")
    except Exception as e:
        print(f"读取CSV(gb18030)失败: {e}")
        try:
            df = pd.read_csv(INPUT_CSV, encoding='utf-8')
            print(f"读取CSV成功，使用utf-8编码")
        except Exception as e2:
            print(f"读取CSV(utf-8)失败: {e2}")
            try:
                df = pd.read_csv(INPUT_CSV, encoding='gbk')
                print(f"读取CSV成功，使用gbk编码")
            except Exception as e3:
                print(f"读取CSV(gbk)失败: {e3}")
                return
    
    print(f"读取到 {len(df)} 条数据")
    
    # 检查必要列
    if "specific_dialogue_content" not in df.columns:
        print("错误：CSV中缺少 'specific_dialogue_content' 列")
        print(f"可用列: {df.columns.tolist()}")
        return
    
    if TRUE_LABEL_COLUMN not in df.columns:
        print(f"错误：CSV中缺少 '{TRUE_LABEL_COLUMN}' 列")
        print(f"可用列: {df.columns.tolist()}")
        return
    
    # 标准化标签处理
    print(f"\n正在标准化标签列 '{TRUE_LABEL_COLUMN}'...")
    df['label_normalized'] = df[TRUE_LABEL_COLUMN].apply(normalize_label)
    
    # 过滤掉无法识别的标签
    original_count = len(df)
    valid_df = df[df['label_normalized'].notna()]
    filtered_count = len(valid_df)
    invalid_count = original_count - filtered_count
    
    if invalid_count > 0:
        print(f"过滤掉 {invalid_count} 条无法识别的标签，剩余 {filtered_count} 条有效数据")
        
        # 显示一些无效标签的示例
        invalid_samples = df[df['label_normalized'].isna()].head(3)
        print("无效标签示例:")
        for i, row in invalid_samples.iterrows():
            print(f"  行{i}: '{row[TRUE_LABEL_COLUMN]}' -> 无法识别")
    
    df = valid_df.copy()
    
    # 显示标准化后的标签分布
    fraud_count = df['label_normalized'].sum()  # True的数量
    non_fraud_count = len(df) - fraud_count
    print(f"\n标准化后标签分布:")
    print(f"  诈骗(True): {fraud_count}条 ({fraud_count/len(df)*100:.1f}%)")
    print(f"  非诈骗(False): {non_fraud_count}条 ({non_fraud_count/len(df)*100:.1f}%)")
    
    # 检查是否有已处理的结果
    processed_count = 0
    all_results = []
    
    if os.path.exists(OUTPUT_CSV):
        try:
            processed_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')
            processed_count = len(processed_df)
            all_results = processed_df.to_dict('records')
            print(f"\n检测到已有进度：已处理 {processed_count} 条对话")
            print(f"将从第 {processed_count + 1} 条开始继续处理")
        except Exception as e:
            print(f"读取已有结果文件失败: {e}")
            print("将重新开始处理所有数据")
            processed_count = 0
            all_results = []
    else:
        print("\n未检测到已有结果文件，将从第一条开始处理")
    
    # 批量处理
    print(f"\n开始测试模型性能...")
    print("=" * 80)
    
    try:
        for idx in range(processed_count, len(df)):
            row = df.iloc[idx]
            dialogue = str(row["specific_dialogue_content"]).strip()
            
            # 使用标准化后的标签
            true_label = row['label_normalized']
            
            if not dialogue or len(dialogue) < 5:
                print(f"[{idx+1}/{len(df)}] 跳过：对话内容过短")
                continue
            
            print(f"\n[{idx+1}/{len(df)}]")
            
            try:
                # 处理单个对话
                result = process_single_dialogue(dialogue, true_label, idx)
                all_results.append(result)
                
                # 每10条保存一次
                if len(all_results) % 10 == 0:
                    save_df = pd.DataFrame(all_results)
                    save_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                    print(f"  已保存{len(all_results)}条结果到文件")
                
            except RuntimeError as e:
                # API调用失败，终止程序
                print(f"\n⚠️ API调用失败，终止程序: {e}")
                print("正在保存已处理的结果...")
                break
            except Exception as e:
                print(f"[ERROR] 样本 {idx} 失败: {e}")
                import traceback
                traceback.print_exc()
                
                # 记录失败
                error_result = {
                    "sample_id": idx,
                    "original_dialogue": dialogue[:200] + "..." if len(dialogue) > 200 else dialogue,
                    "true_label": label_to_chinese(true_label),
                    "prediction": "error",
                    "is_correct": False,
                    "processing_time_seconds": 0,
                    "dialogue_length": len(dialogue),
                    "error": str(e)
                }
                all_results.append(error_result)
                continue
            
            # 短暂延迟避免API限制
            time.sleep(0.5)
        
        # 处理完成或中断后保存
        print(f"\n{'='*80}")
        print("处理完成或中断，正在保存最终结果...")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ 用户中断程序")
        print("正在保存已处理的结果...")
    except Exception as e:
        print(f"\n\n⚠️ 程序发生异常: {e}")
        print("正在保存已处理的结果...")
        import traceback
        traceback.print_exc()
    
    # 保存结果
    if all_results:
        try:
            save_df = pd.DataFrame(all_results)
            save_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"结果已保存到: {OUTPUT_CSV}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    # 最终汇总分析
    if all_results:
        try:
            final_df = pd.DataFrame(all_results)
            
            print(f"\n{'='*80}")
            print("测试完成！最终统计")
            print(f"{'='*80}")
            
            # 1. 总体统计
            total_samples = len(final_df)
            correct_predictions = final_df[final_df['is_correct'] == True]
            incorrect_predictions = final_df[final_df['is_correct'] == False]
            error_predictions = final_df[final_df['prediction'] == 'error']
            unknown_predictions = final_df[final_df['prediction'] == 'unknown']
            
            total_correct = len(correct_predictions)
            total_incorrect = len(incorrect_predictions)
            total_error = len(error_predictions)
            total_unknown = len(unknown_predictions)
            
            # 计算正确率（排除error和unknown）
            valid_predictions = final_df[~final_df['prediction'].isin(['error', 'unknown'])]
            total_valid = len(valid_predictions)
            valid_correct = len(valid_predictions[valid_predictions['is_correct'] == True])
            
            accuracy = valid_correct / total_valid * 100 if total_valid > 0 else 0
            
            print(f"【总体统计】:")
            print(f"  总处理样本数: {total_samples}")
            print(f"  有效预测数: {total_valid} ({total_valid/total_samples*100:.1f}%)")
            print(f"  正确预测数: {valid_correct}")
            print(f"  模型正确率: {accuracy:.2f}% ({valid_correct}/{total_valid})")
            
            if total_error > 0:
                print(f"  API调用失败数: {total_error} ({total_error/total_samples*100:.1f}%)")
            
            if total_unknown > 0:
                print(f"  无法解析的响应数: {total_unknown} ({total_unknown/total_samples*100:.1f}%)")
            
            # 2. 按标签统计
            print(f"\n【按标签统计】:")
            
            # 诈骗样本
            fraud_samples = valid_predictions[valid_predictions['true_label'] == '诈骗']
            if len(fraud_samples) > 0:
                fraud_correct = len(fraud_samples[fraud_samples['is_correct'] == True])
                fraud_accuracy = fraud_correct / len(fraud_samples) * 100 if len(fraud_samples) > 0 else 0
                print(f"  诈骗样本:")
                print(f"    总数: {len(fraud_samples)}")
                print(f"    正确识别: {fraud_correct} ({fraud_accuracy:.1f}%)")
            
            # 非诈骗样本
            non_fraud_samples = valid_predictions[valid_predictions['true_label'] == '非诈骗']
            if len(non_fraud_samples) > 0:
                non_fraud_correct = len(non_fraud_samples[non_fraud_samples['is_correct'] == True])
                non_fraud_accuracy = non_fraud_correct / len(non_fraud_samples) * 100 if len(non_fraud_samples) > 0 else 0
                print(f"  非诈骗样本:")
                print(f"    总数: {len(non_fraud_samples)}")
                print(f"    正确识别: {non_fraud_correct} ({non_fraud_accuracy:.1f}%)")
            
            # 3. 性能统计
            print(f"\n【性能统计】:")
            avg_time = final_df['processing_time_seconds'].mean()
            max_time = final_df['processing_time_seconds'].max()
            min_time = final_df['processing_time_seconds'].min()
            
            print(f"  平均处理时间: {avg_time:.2f}秒")
            print(f"  最短处理时间: {min_time:.2f}秒")
            print(f"  最长处理时间: {max_time:.2f}秒")
            
            # 4. 错误分析
            if total_incorrect > 0:
                print(f"\n【错误分析】:")
                
                # 误报（非诈骗被识别为诈骗）
                false_positives = final_df[(final_df['true_label'] == '非诈骗') & (final_df['prediction'] == '诈骗')]
                if len(false_positives) > 0:
                    print(f"  误报(False Positive): {len(false_positives)}条")
                
                # 漏报（诈骗被识别为非诈骗）
                false_negatives = final_df[(final_df['true_label'] == '诈骗') & (final_df['prediction'] == '非诈骗')]
                if len(false_negatives) > 0:
                    print(f"  漏报(False Negative): {len(false_negatives)}条")
            
            # 5. 保存详细报告
            report_file = OUTPUT_CSV.replace('.csv', '_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("GLM-4诈骗检测模型性能评估报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"测试时间: {pd.Timestamp.now()}\n")
                f.write(f"总样本数: {total_samples}\n")
                f.write(f"有效预测数: {total_valid}\n")
                f.write(f"正确预测数: {valid_correct}\n")
                f.write(f"模型正确率: {accuracy:.2f}%\n\n")
                
                f.write("按标签统计:\n")
                f.write(f"  诈骗样本: {len(fraud_samples)}条, 正确率: {fraud_accuracy:.1f}%\n")
                f.write(f"  非诈骗样本: {len(non_fraud_samples)}条, 正确率: {non_fraud_accuracy:.1f}%\n\n")
                
                f.write("性能统计:\n")
                f.write(f"  平均处理时间: {avg_time:.2f}秒\n")
                f.write(f"  最短处理时间: {min_time:.2f}秒\n")
                f.write(f"  最长处理时间: {max_time:.2f}秒\n")
            
            print(f"\n✅ 详细报告已保存到: {report_file}")
            print(f"✅ 完整结果已保存到: {OUTPUT_CSV}")
            
            # 显示前几个样本的预测情况
            print(f"\n{'='*80}")
            print("前10个样本的预测结果:")
            print(f"{'='*80}")
            
            for i, (_, row) in enumerate(final_df.head(10).iterrows()):
                status = "✓" if row['is_correct'] else "✗"
                print(f"{i+1:3d}. ID:{row['sample_id']:4d} | 真实:{row['true_label']:3s} | 预测:{row['prediction']:4s} | {status}")
            
        except Exception as e:
            print(f"生成统计信息失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ 没有生成任何结果")

# =========================
# 7. 主程序入口
# =========================

if __name__ == "__main__":
    main()