import pandas as pd
import time
import requests
from typing import Dict, List, Tuple, Optional
import re
import os
import sys
from bert_score import BERTScorer

# =========================
# 1. 基础配置
# =========================

API_KEY = "a835fc322e094cb98848b40310e16202.GTTeL2vxWikgn8Wd"
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME = "glm-4-air"

INPUT_CSV = "D:\\自然语言处理\\期末大作业\\测试集结果.csv"
OUTPUT_CSV = "D:\\自然语言处理\\期末大作业\\promptattack_A1.csv"
SLEEP_TIME = 1.5
BATCH_SIZE = 10  # 每多少条保存一次

# API错误重试配置
MAX_API_RETRIES = 3  # 最大重试次数
API_RETRY_DELAY = 5  # 重试延迟（秒）
NETWORK_TIMEOUT = 30  # 网络超时时间（秒）

# BERTScore配置
BERTSCORE_MODEL = "bert-base-chinese"  # 中文模型
USE_CUDA = False  # 如果没有GPU，设为False

# 任务配置
TASK_DESCRIPTION = "请对以下对话进行诈骗检测，判断是否为诈骗对话。"
LABEL_SET = ["诈骗", "非诈骗"]
TRUE_LABEL_COLUMN = "is_fraud"  # 真实标签列名

# =========================
# 2. 对话扰动指令（优化版）
# =========================

DIALOGUE_PERTURBATIONS = {
    "C1": "在对话中选择7-9个词，让它们出现轻微拼写错误（如错别字），但不影响理解。",
    
    "W1": "替换对话中7-9个关键词为同义词或更正式的表达方式。",
    
    "S1": "将句子改写成更官方、更规范的表达方式。",  
    # 特殊选项
    "ALL": "尝试所有扰动类型",
    "RANDOM": "随机选择一个扰动类型"
}

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
        if label_lower in ['true', '是', 'yes', '1', '诈骗', '欺诈', '真', '正确', 'true']:
            return True
        # 识别为非诈骗的情况  
        elif label_lower in ['false', '否', 'no', '0', '非诈骗', '正常', '非欺诈', '假', '错误', 'false']:
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
# 3. 初始化BERTScorer
# =========================

def initialize_bertscorer():
    """初始化BERTScorer"""
    print("初始化BERTScorer...")
    try:
        # 使用中文BERT模型
        scorer = BERTScorer(
            model_type=BERTSCORE_MODEL,
            lang="zh",  # 指定中文
            num_layers=8,  # 使用中间层，平衡速度和精度
            rescale_with_baseline=True,  # 使用基线调整
            device='cuda' if USE_CUDA else 'cpu'
        )
        print(f"✓ BERTScorer初始化成功，使用模型: {BERTSCORE_MODEL}")
        print(f"  设备: {'GPU' if USE_CUDA else 'CPU'}")
        return scorer
    except Exception as e:
        print(f"✗ BERTScorer初始化失败: {e}")
        print("请检查是否安装了必要的包：pip install bert-score torch transformers")
        return None

# =========================
# 4. 用户交互：选择扰动类型
# =========================

def select_perturbation_mode() -> List[str]:
    """
    让用户选择扰动模式
    返回: 选择的扰动ID列表
    """
    print("\n" + "="*60)
    print("请选择扰动类型:")
    print("="*60)
    
    # 显示所有选项
    options = list(DIALOGUE_PERTURBATIONS.keys())
    for i, key in enumerate(options, 1):
        if key in ["ALL", "RANDOM"]:
            print(f"{i:2d}. {key:8s} - {DIALOGUE_PERTURBATIONS[key]}")
        else:
            print(f"{i:2d}. {key:8s} - {DIALOGUE_PERTURBATIONS[key][:50]}...")
    
    print("\n输入说明:")
    print("  - 输入单个数字选择对应扰动类型")
    print("  - 输入多个数字（用逗号分隔）选择多个扰动类型")
    print("  - 输入 'all' 选择所有扰动类型")
    print("  - 输入 'random' 随机选择")
    print("  - 按Enter使用默认值 (ALL)")
    
    while True:
        user_input = input("\n请选择（输入数字或关键字）: ").strip()
        
        if user_input == "":
            print("使用默认值: ALL (所有扰动类型)")
            return list(DIALOGUE_PERTURBATIONS.keys())[:-2]  # 排除ALL和RANDOM
        
        if user_input.lower() == 'all':
            print("选择: ALL (所有扰动类型)")
            return list(DIALOGUE_PERTURBATIONS.keys())[:-2]  # 排除ALL和RANDOM
        
        if user_input.lower() == 'random':
            print("选择: RANDOM (随机选择)")
            import random
            perturbations = list(DIALOGUE_PERTURBATIONS.keys())[:-2]  # 排除ALL和RANDOM
            selected = [random.choice(perturbations)]
            print(f"随机选择: {selected[0]}")
            return selected
        
        # 检查是否是多个数字
        if ',' in user_input:
            try:
                indices = [int(x.strip()) for x in user_input.split(',')]
                selected_keys = []
                for idx in indices:
                    if 1 <= idx <= len(options):
                        selected_keys.append(options[idx-1])
                
                # 过滤掉ALL和RANDOM
                filtered_keys = [k for k in selected_keys if k not in ["ALL", "RANDOM"]]
                
                if filtered_keys:
                    print(f"已选择: {', '.join(filtered_keys)}")
                    return filtered_keys
                else:
                    print("无效的选择，请重新输入")
            except ValueError:
                print("请输入有效的数字，如: 1,3,5")
        
        # 检查是否是单个数字
        try:
            idx = int(user_input)
            if 1 <= idx <= len(options):
                selected_key = options[idx-1]
                
                if selected_key in ["ALL", "RANDOM"]:
                    if selected_key == "ALL":
                        print("选择: ALL (所有扰动类型)")
                        return list(DIALOGUE_PERTURBATIONS.keys())[:-2]
                    else:
                        print("选择: RANDOM (随机选择)")
                        import random
                        perturbations = list(DIALOGUE_PERTURBATIONS.keys())[:-2]
                        selected = [random.choice(perturbations)]
                        print(f"随机选择: {selected[0]}")
                        return selected
                else:
                    print(f"已选择: {selected_key}")
                    return [selected_key]
            else:
                print(f"请输入1-{len(options)}之间的数字")
        except ValueError:
            print("请输入有效的数字或关键字")

# =========================
# 5. 修复对话解析工具（支持\n分隔）
# =========================

def extract_dialogue_parts(dialogue_text: str) -> List[Dict]:
    """解析对话文本，提取左右发言 - 支持多种分隔符"""
    parts = []
    if not dialogue_text or not isinstance(dialogue_text, str):
        return parts
    
    # 标准化分隔符：先统一为\n
    if '<br>' in dialogue_text:
        lines = dialogue_text.replace('<br>', '\n').split('\n')
    else:
        lines = dialogue_text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # 更灵活的匹配模式
        line_lower = line.lower()
        if 'left:' in line_lower or line_lower.startswith('left'):
            # 提取left部分
            if 'left:' in line:
                colon_index = line.find(':')
                if colon_index != -1:
                    content = line[colon_index + 1:].strip()
                else:
                    content = line
            else:
                content = line[4:].strip() if line.lower().startswith('left') else line
            parts.append({
                'speaker': 'left',
                'content': content,
                'line_num': i
            })
            
        elif 'right:' in line_lower or line_lower.startswith('right'):
            # 提取right部分
            if 'right:' in line:
                colon_index = line.find(':')
                if colon_index != -1:
                    content = line[colon_index + 1:].strip()
                else:
                    content = line
            else:
                content = line[5:].strip() if line.lower().startswith('right') else line
            parts.append({
                'speaker': 'right',
                'content': content,
                'line_num': i
            })
        else:
            # 如果这行没有明确标签，检查前一行是什么
            if parts:
                last_speaker = parts[-1]['speaker']
                parts.append({
                    'speaker': last_speaker,
                    'content': line,
                    'line_num': i,
                    'continued': True
                })
            else:
                # 如果没有之前的发言，假设是left
                parts.append({
                    'speaker': 'left',
                    'content': line,
                    'line_num': i,
                    'assumed': True
                })
    
    return parts

def reconstruct_dialogue(parts: List[Dict], separator: str = '\n') -> str:
    """将对话部分重新组合为文本 - 支持指定分隔符"""
    lines = []
    for part in parts:
        if part.get('continued', False):
            lines.append(part['content'])
        else:
            lines.append(f"{part['speaker']}: {part['content']}")
    return separator.join(lines)

# =========================
# 6. BERTScore保真度计算
# =========================

def calculate_bertscore(original: str, adversarial: str, scorer) -> Dict:
    """使用BERTScore计算相似度"""
    if original == adversarial:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'fidelity': 1.0
        }
    
    try:
        # 清理文本（保留标签和完整内容）
        def clean_for_bert(text):
            # 保留原始格式，BERT模型可以处理
            text = re.sub(r'<br>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        orig_clean = clean_for_bert(original)
        adv_clean = clean_for_bert(adversarial)
        
        # 计算BERTScore
        P, R, F1 = scorer.score([adv_clean], [orig_clean])
        
        # 转换为Python float
        precision = P.item() if hasattr(P, 'item') else float(P)
        recall = R.item() if hasattr(R, 'item') else float(R)
        f1_score = F1.item() if hasattr(F1, 'item') else float(F1)
        
        # 使用F1作为保真度分数
        fidelity = f1_score
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1_score, 4),
            'fidelity': round(fidelity, 4)
        }
        
    except Exception as e:
        print(f"    BERTScore计算失败: {e}")
        # 回退到简单相似度
        return calculate_simple_similarity(original, adversarial)

def calculate_simple_similarity(original: str, adversarial: str) -> Dict:
    """备用：简单相似度计算（当BERTScore失败时使用）"""
    if original == adversarial:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'fidelity': 1.0,
            'method': 'simple'
        }
    
    # 简单计算字符级别的重叠
    orig_chars = set(original)
    adv_chars = set(adversarial)
    
    intersection = len(orig_chars & adv_chars)
    union = len(orig_chars | adv_chars)
    
    jaccard = intersection / union if union > 0 else 0
    
    return {
        'precision': round(jaccard, 4),
        'recall': round(jaccard, 4),
        'f1': round(jaccard, 4),
        'fidelity': round(jaccard, 4),
        'method': 'simple'
    }

# =========================
# 7. 保真度过滤器（使用BERTScore）
# =========================

def fidelity_filter(original: str, adversarial: str, scorer, 
                   min_fidelity: float = 0.65,
                   max_fidelity: float = 0.93) -> Tuple[bool, Dict]:
    """保真度过滤器：使用BERTScore确保对抗样本语义基本不变"""
    if original == adversarial:
        return False, {'fidelity': 1.0, 'method': 'identical'}
    
    # 计算BERTScore
    bertscore_result = calculate_bertscore(original, adversarial, scorer)
    fidelity = bertscore_result['fidelity']
    
    result = {
        'fidelity': fidelity,
        'precision': bertscore_result['precision'],
        'recall': bertscore_result['recall'],
        'f1': bertscore_result['f1'],
        'method': 'bertscore' if 'method' not in bertscore_result else bertscore_result['method']
    }
    
    if fidelity < min_fidelity:
        print(f"    保真度过滤：相似度过低({fidelity:.4f} < {min_fidelity})")
        return False, result
    elif fidelity > max_fidelity:
        print(f"    保真度过滤：相似度过高({fidelity:.4f} > {max_fidelity})，修改太小")
        return False, result
    
    return True, result

# =========================
# 8. LLM 调用（带重试机制）
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
# 9. 构建三段式攻击 Prompt（OI + AO + AG）
# =========================

def build_attack_prompt_three_stages(
    original_dialogue: str,
    perturbation_id: str,
    true_label: bool
) -> str:
    """构建三段式攻击 Prompt"""
    
    # 转换True/False为中文标签
    true_label_chinese = "诈骗" if true_label == True else "非诈骗"
    target_label_chinese = "非诈骗" if true_label == True else "诈骗"
    
    # 处理对话显示（关键：不要在f-string内部处理转义）
    dialogue_display = original_dialogue.replace('<br>', '\n')
    
    # 使用 + 号连接字符串而不是复杂的f-string
    oi_section = f"""# 原始输入 (Original Input)
原始对话：
{dialogue_display}

真实标签：{true_label_chinese}
标签集合：诈骗, 非诈骗
任务描述：{TASK_DESCRIPTION}"""
    
    ao_section = f"""# 攻击目标 (Attack Objective)
你的任务是生成一个修改后的对话，满足以下条件：
1. 保持原对话的语义不变；
2. 修改后的对话应使LLM（你自己）做出错误的分类预测；
3. 原始标签是"{true_label_chinese}"，你需要让LLM预测为"{target_label_chinese}"；
4. 修改后的对话应该看起来自然，不引起怀疑。"""
     # 根据标签动态生成攻击指导
    if true_label == True:  # 诈骗
        svm_attack_requirements = """【特别针对SVM的攻击要求】：
1. 必须替换以下诈骗关键词：
   - '投资' → '理财'或'资产配置'
   - '链接' → '网址'或'页面'
   - '贷款' → '借款'或'融资'
   - '下载' → '获取'或'安装'
   - '点击' → '访问'或'打开'
   
2. 尽量添加以下非诈骗词语：
   - 积分、预约、外卖、价格、小区、物业
   
目标是欺骗分类器"""
    else:  # 非诈骗
        svm_attack_requirements = """【特别针对SVM的攻击要求】：
1. 必须替换以下非诈骗关键词：
   - '积分' → '投资'或'理财'
   - '预约' → '链接'或'网址'
   - '外卖' → '贷款'或'融资'
   - '价格' → '下载'或'获取'
   - '小区' → '点击'或'访问'
   - '物业' → '资产配置'或'页面'
   
2. 尽量添加以下诈骗词语：
   - 投资、链接、贷款、下载、点击
   
目标是欺骗分类器"""   
    ag_section = f"""# 攻击指导 (Attack Guidance)
请按照以下指导修改对话：
{DIALOGUE_PERTURBATIONS[perturbation_id]}
{svm_attack_requirements}

重要格式要求：
1. 保持"left:"和"right:"标签不变；
2. 每行格式为"标签: 内容"，用换行分隔；
3. 只输出修改后的完整对话，不要添加额外说明；
4. 保持对话的行数和顺序不变。

请输出修改后的对话："""
    
    return f"{oi_section}\n\n{ao_section}\n\n{ag_section}"

# =========================
# 10. 清理LLM输出
# =========================

def clean_llm_output(text: str) -> str:
    """清理LLM输出的文本"""
    if not text:
        return ""
    
    # 移除常见的开头提示词
    patterns = [
        r'^修改后的对话[:：]?\s*',
        r'^新的对话[:：]?\s*',
        r'^输出[:：]?\s*',
        r'^结果[:：]?\s*',
        r'^对抗对话[:：]?\s*',
        r'^对抗样本[:：]?\s*',
        r'^完整对话[:：]?\s*',
        r'^以下是我的修改[:：]?\s*'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 移除多余的空白字符
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

# =========================
# 11. 验证对话格式
# =========================

def validate_dialogue_format(original: str, adversarial: str) -> bool:
    """验证对抗对话格式是否与原始对话一致"""
    original_parts = extract_dialogue_parts(original)
    adversarial_parts = extract_dialogue_parts(adversarial)
    
    if len(original_parts) != len(adversarial_parts):
        print(f"    格式错误：行数不一致（原始:{len(original_parts)}，对抗:{len(adversarial_parts)}）")
        return False
    
    # 检查每行的说话者标签是否一致
    for i in range(len(original_parts)):
        if original_parts[i]['speaker'] != adversarial_parts[i]['speaker']:
            print(f"    格式错误：第{i+1}行说话者标签改变")
            return False
    
    return True

# =========================
# 12. LLM自评估预测
# =========================

def self_evaluate_prediction(dialogue: str, return_raw: bool = False) -> str:
    """
    修复版：调用LLM自身对生成的对话进行诈骗检测预测
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
        
        if return_raw:
            return response
        
        if not response:
            print("    自评估失败：API返回空")
            return "error"
        
        # 清理和解析响应
        response_clean = response.strip().upper()
        
        # 直接匹配A或B
        if response_clean == "A":
            return "诈骗"
        elif response_clean == "B":
            return "非诈骗"
        
        # 尝试提取第一个A或B
        import re
        match = re.search(r'[AB]', response_clean)
        if match:
            return "诈骗" if match.group() == "A" else "非诈骗"
        
        # 如果还是无法解析，打印原始响应用于调试
        print(f"    无法解析预测结果，原响应: '{response}'")
        return "unknown"
            
    except Exception as e:
        print(f"    自评估失败: {e}")
        return "error"

# =========================
# 13. 生成对抗样本（带攻击循环和自评估）
# =========================

def generate_adversarial_with_attack_loop(
    dialogue: str,
    perturbation_id: str,
    scorer,
    true_label: bool,
    original_prediction: str,
    max_attempts: int = 3  
) -> Tuple[str, str, Dict, bool, bool, str]:
    """
    生成对抗样本，带攻击循环
    返回: (对抗对话, 扰动类型, 保真度结果, 生成成功, 攻击成功, 预测结果)
    """
    
    true_label_chinese = "诈骗" if true_label == True else "非诈骗"
    target_label_chinese = "非诈骗" if true_label == True else "诈骗"
    
    best_adv_dialogue = dialogue  # 初始化最佳为原始对话
    best_fidelity_result = {'fidelity': 1.0, 'method': 'identical'}
    best_prediction = original_prediction
    best_perturbation = perturbation_id
    best_gen_success = False  # 标记是否生成成功
    best_attack_success = False  # 标记是否攻击成功
    
    for attempt in range(max_attempts):
        print(f"    攻击尝试 {attempt + 1}/{max_attempts}")
        
        # 构建三段式Prompt
        prompt = build_attack_prompt_three_stages(dialogue, perturbation_id, true_label)
        
        # 调用LLM生成对抗样本
        try:
            adv_dialogue = chat_completion_with_retry([
                {"role": "system", "content": "你是一个对抗样本生成助手，需要修改对话以欺骗诈骗检测系统。"},
                {"role": "user", "content": prompt}
            ])
        except RuntimeError as e:
            print(f"    LLM调用失败: {e}")
            continue
        
        # 清理输出
        adv_dialogue = clean_llm_output(adv_dialogue)
        
        # 检查是否有实际修改
        if adv_dialogue == dialogue:
            print("    内容未修改，跳过")
            continue
        
        # 检查格式
        if not validate_dialogue_format(dialogue, adv_dialogue):
            print("    格式验证失败")
            continue
        
        # 保真度检查（BERTScore）
        passed, fidelity_result = fidelity_filter(dialogue, adv_dialogue, scorer, 0.55, 0.95)
        if not passed:
            print(f"    保真度不足: {fidelity_result['fidelity']:.4f}")
            continue
        
        # 调用LLM自身进行预测（自评估）
        prediction = self_evaluate_prediction(adv_dialogue)
        
        # 标记生成成功
        current_gen_success = True
        current_attack_success = (prediction == target_label_chinese and original_prediction == true_label_chinese)
        
        # 更新最佳结果（即使攻击失败也要保存生成的对抗样本）
        if not best_gen_success:
            # 第一次生成成功
            best_adv_dialogue = adv_dialogue
            best_fidelity_result = fidelity_result
            best_prediction = prediction
            best_perturbation = perturbation_id
            best_gen_success = current_gen_success
            best_attack_success = current_attack_success
        elif fidelity_result['fidelity'] > best_fidelity_result['fidelity']:
            # 保真度更高的结果
            best_adv_dialogue = adv_dialogue
            best_fidelity_result = fidelity_result
            best_prediction = prediction
            best_perturbation = perturbation_id
            best_gen_success = current_gen_success
            best_attack_success = current_attack_success
        elif fidelity_result['fidelity'] == best_fidelity_result['fidelity'] and current_attack_success and not best_attack_success:
            # 保真度相同但这次攻击成功了
            best_adv_dialogue = adv_dialogue
            best_fidelity_result = fidelity_result
            best_prediction = prediction
            best_perturbation = perturbation_id
            best_gen_success = current_gen_success
            best_attack_success = current_attack_success
        
        # 检查是否欺骗成功
        if current_attack_success:
            print(f"    ✓ 欺骗成功！预测标签：{prediction}（原预测：{original_prediction}，真实：{true_label_chinese}）")
            return adv_dialogue, perturbation_id, fidelity_result, True, True, prediction
        
        print(f"    ✗ 欺骗失败，预测为：{prediction}（原预测：{original_prediction}，目标：{target_label_chinese}）")
    
    # 返回最好的生成结果（即使攻击失败）
    if best_gen_success:
        print(f"    ⚠️ 生成成功但攻击失败，保存最佳对抗样本")
        return best_adv_dialogue, best_perturbation, best_fidelity_result, True, best_attack_success, best_prediction
    else:
        print(f"    ❌ 所有尝试均失败")
        return dialogue, "none", {'fidelity': 1.0, 'method': 'failed'}, False, False, original_prediction

# =========================
# 辅助函数定义
# =========================

def create_failure_result(dialogue, true_label, reason, parts_count):
    """创建失败结果"""
    true_label_chinese = "诈骗" if true_label == True else "非诈骗"
    return {
        "original_dialogue": dialogue,
        "adversarial_dialogue": dialogue,
        "perturbation_type": "none",
        "fidelity": 0.0,
        "bert_precision": 0.0,
        "bert_recall": 0.0,
        "bert_f1": 0.0,
        "similarity_method": reason,
        "generation_success": False,
        "attack_success": False,
        "true_label": true_label_chinese,
        "original_prediction": "none",
        "adversarial_prediction": "none",
        "original_correct": False,
        "parts_count": parts_count,
        "adv_parts_count": parts_count
    }

def create_skip_result(dialogue, true_label, original_prediction, parts_count):
    """创建跳过结果（原始预测错误）"""
    true_label_chinese = "诈骗" if true_label == True else "非诈骗"
    return {
        "original_dialogue": dialogue,
        "adversarial_dialogue": dialogue,
        "perturbation_type": "skip_original_wrong",
        "fidelity": 1.0,
        "bert_precision": 1.0,
        "bert_recall": 1.0,
        "bert_f1": 1.0,
        "similarity_method": "identical",
        "generation_success": False,
        "attack_success": False,
        "true_label": true_label_chinese,
        "original_prediction": original_prediction,
        "adversarial_prediction": "none",
        "original_correct": False,
        "parts_count": parts_count,
        "adv_parts_count": parts_count,
        "skip_reason": "original_wrong"
    }

def create_complete_failure_result(dialogue, true_label, original_prediction, parts_count):
    """创建完全失败结果（原始预测正确但生成失败）"""
    true_label_chinese = "诈骗" if true_label == True else "非诈骗"
    return {
        "original_dialogue": dialogue,
        "adversarial_dialogue": dialogue,
        "perturbation_type": "none",
        "fidelity": 1.0,
        "bert_precision": 1.0,
        "bert_recall": 1.0,
        "bert_f1": 1.0,
        "similarity_method": "identical",
        "generation_success": False,
        "attack_success": False,
        "true_label": true_label_chinese,
        "original_prediction": original_prediction,
        "adversarial_prediction": original_prediction,
        "original_correct": True,
        "parts_count": parts_count,
        "adv_parts_count": parts_count
    }

# =========================
# 14. 主攻击流程（符合论文要求）- 修改版
# =========================

def generate_for_dialogue_paper_version(
    dialogue: str,
    true_label: bool,  # 现在这个已经是标准化后的布尔值
    selected_perturbations: List[str],
    scorer
) -> Dict:
    """
    为单个对话生成对抗样本 - 符合论文要求：
    1. 先检查LLM对原始样本的预测
    2. 只在原始预测正确的情况下进行攻击
    """
    
    # 首先验证原始对话格式
    parts = extract_dialogue_parts(dialogue)
    if len(parts) < 1:
        print(f"  无法解析对话，跳过")
        return create_failure_result(dialogue, true_label, "parse_error", 0)
    
    print(f"  解析出 {len(parts)} 个对话部分")
    
    # ========== 修改：直接使用标准化后的布尔值标签 ==========
    true_label_chinese = "诈骗" if true_label == True else "非诈骗"
    print(f"  真实标签: {true_label_chinese}")
    
    # ========== 论文关键步骤1：检查原始预测 ==========
    original_prediction = self_evaluate_prediction(dialogue)
    original_correct = (original_prediction == true_label_chinese)
    
    print(f"  原始预测: {original_prediction}")
    print(f"  原始预测是否正确: {'✓' if original_correct else '✗'}")
    
    # 如果原始预测错误，按照论文要求跳过
    if not original_correct:
        print(f"  ⚠️ 跳过：LLM对原始样本的预测已错误")
        return create_skip_result(dialogue, true_label, original_prediction, len(parts))
    
    # ========== 论文关键步骤2：只在原始正确的样本上攻击 ==========
    print(f"  ✓ LLM原始预测正确，开始攻击...")
    
    best_result = None
    attack_success_found = False
    
    # 尝试每个扰动类型
    for perturbation_id in selected_perturbations:
        print(f"  尝试扰动: {perturbation_id}")
        
        try:
            adv_dialogue, used_perturbation, fidelity_result, gen_success, attack_success, adv_prediction = generate_adversarial_with_attack_loop(
                dialogue, perturbation_id, scorer, true_label, original_prediction
            )
        except RuntimeError as e:
            # API调用失败，向上抛出异常
            print(f"  ⚠️ API调用失败，停止处理: {e}")
            raise
        
        if gen_success:
            # 构建结果 - 无论攻击成功与否都保存对抗样本
            result = {
                "original_dialogue": dialogue,
                "adversarial_dialogue": adv_dialogue,  # 这里保存对抗样本，不是原始对话
                "perturbation_type": used_perturbation,
                "fidelity": fidelity_result['fidelity'],
                "bert_precision": fidelity_result['precision'],
                "bert_recall": fidelity_result['recall'],
                "bert_f1": fidelity_result['f1'],
                "similarity_method": fidelity_result.get('method', 'bertscore'),
                "generation_success": gen_success,
                "attack_success": attack_success,
                "true_label": true_label_chinese,
                "original_prediction": original_prediction,
                "adversarial_prediction": adv_prediction,
                "original_correct": True,
                "parts_count": len(parts),
                "adv_parts_count": len(extract_dialogue_parts(adv_dialogue))
            }
            
            # 保存结果
            if attack_success:
                print(f"  ✅ 攻击成功！")
                attack_success_found = True
                return result  # 攻击成功立即返回
            
            # 保存最佳结果（如果没有攻击成功的话）
            if best_result is None:
                best_result = result
            elif fidelity_result['fidelity'] > best_result['fidelity']:
                best_result = result
            
            print(f"  ⚠️ 生成成功但攻击失败，继续尝试其他扰动...")
            
        else:
            print(f"  ✗ 扰动 {perturbation_id} 失败")
    
    # 返回结果
    if attack_success_found:
        # 这行不会执行，因为攻击成功时已经在上面return了
        pass
    elif best_result is not None:
        print(f"  ⚠️ 所有扰动类型均攻击失败，返回最佳生成结果")
        return best_result
    else:
        # 全部失败
        print(f"  ❌ 所有扰动类型均失败")
        return create_complete_failure_result(dialogue, true_label, original_prediction, len(parts))

# =========================
# 15. 批量处理主程序（支持断点续传）- 修改版，使用正确的ASR计算
# =========================

def main():
    """主函数：批量生成对抗样本 - 支持断点续传"""
    print("=" * 80)
    print("对抗样本生成器 (诈骗检测任务 - 符合论文要求的PromptAttack)")
    print("论文要求：只对LLM原始预测正确的样本进行攻击")
    print("=" * 80)
    
    # 初始化BERTScorer
    scorer = initialize_bertscorer()
    if scorer is None:
        print("警告: BERTScorer初始化失败，将使用简单相似度计算")
        # 可以继续运行，但使用备用方案
    
    # 用户选择扰动类型
    selected_perturbations = select_perturbation_mode()
    
    print(f"\n选定的扰动类型: {', '.join(selected_perturbations)}")
    print(f"共 {len(selected_perturbations)} 种扰动")
    print(f"批量大小: {BATCH_SIZE}条/批")
    print(f"API最大重试次数: {MAX_API_RETRIES}")
    print("程序将在API调用失败时自动终止")
    print(f"任务描述: {TASK_DESCRIPTION}")
    
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
    
    # ========== 新增：标准化标签处理 ==========
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
    print(f"标准化后标签分布:")
    print(f"  诈骗(True): {fraud_count}条 ({fraud_count/len(df)*100:.1f}%)")
    print(f"  非诈骗(False): {non_fraud_count}条 ({non_fraud_count/len(df)*100:.1f}%)")
    
    # 显示前几个样本的标签转换情况（用于调试）
    if len(df) > 0:
        print(f"\n前3个样本的标签转换示例:")
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            original = row[TRUE_LABEL_COLUMN]
            normalized = row['label_normalized']
            chinese = label_to_chinese(normalized)
            print(f"  样本{i}: 原始='{original}' → 标准化={normalized} → 中文='{chinese}'")
    # ========== 标签标准化结束 ==========
    
    # 检查是否有已处理的结果（断点续传）
    processed_count = 0
    all_results = []
    
    if os.path.exists(OUTPUT_CSV):
        try:
            processed_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')
            processed_count = len(processed_df)
            all_results = processed_df.to_dict('records')
            print(f"\n检测到已有进度：已处理 {processed_count} 条对话")
            print(f"将从第 {processed_count + 1} 条开始继续处理")
            
            # 显示已有统计
            if len(all_results) > 0:
                # 计算符合论文要求的攻击成功率（使用正确的ASR公式）
                original_correct = [r for r in all_results if r.get('original_correct', False)]
                if original_correct:
                    # 原始正确且生成成功的样本（分母）
                    valid_for_asr = [r for r in original_correct if r.get('generation_success', False)]
                    valid_count = len(valid_for_asr)
                    
                    # 攻击成功的样本（分子）：原始正确∩生成成功∩对抗错误
                    attacks_on_valid = sum(1 for r in valid_for_asr if r.get('attack_success', False))
                    
                    # 正确的ASR计算
                    if valid_count > 0:
                        paper_asr = attacks_on_valid / valid_count * 100
                    else:
                        paper_asr = 0
                    
                    print(f"已有结果统计:")
                    print(f"  原始预测正确的样本数: {len(original_correct)}")
                    print(f"  其中生成成功的样本数: {valid_count}")
                    print(f"  在这些生成成功样本中攻击成功的样本数: {attacks_on_valid}")
                    print(f"  论文攻击成功率(ASR): {paper_asr:.2f}%")
        except Exception as e:
            print(f"读取已有结果文件失败: {e}")
            print("将重新开始处理所有数据")
            processed_count = 0
            all_results = []
    else:
        print("\n未检测到已有结果文件，将从第一条开始处理")
    
    # 批量处理
    current_batch_results = []
    batch_count = 0
    
    print(f"\n开始生成对抗样本...")
    print("=" * 80)
    
    try:
        for idx in range(processed_count, len(df)):
            row = df.iloc[idx]
            dialogue = str(row["specific_dialogue_content"]).strip()
            
            # ========== 修改：使用标准化后的标签 ==========
            true_label = row['label_normalized']  # 使用标准化后的布尔值标签
            
            if not dialogue or len(dialogue) < 5:
                print(f"[{idx+1}/{len(df)}] 跳过：对话内容过短")
                continue
            
            print(f"\n[{idx+1}/{len(df)}] 处理样本 {idx}...")
            
            try:
                # 生成对抗样本（使用论文版本）
                if scorer is not None:
                    result = generate_for_dialogue_paper_version(
                        dialogue, true_label, selected_perturbations, scorer
                    )
                else:
                    # 如果没有BERTScorer，使用简单版本
                    from copy import deepcopy
                    simple_scorer = {'dummy': True}  # 占位符
                    result = generate_for_dialogue_paper_version(
                        dialogue, true_label, selected_perturbations, simple_scorer
                    )
                
                # 添加样本ID和其他信息
                result["sample_id"] = idx
                result["selected_perturbations"] = ', '.join(selected_perturbations)
                result["row_index"] = idx
                
                current_batch_results.append(result)
                all_results.append(result)
                
                # 显示处理结果
                if result["attack_success"]:
                    print("  ✅ 攻击成功！")
                elif result["generation_success"]:
                    print("  ⚠️ 生成成功但攻击失败")
                elif not result["original_correct"]:
                    print("  ⚠️ 跳过：原始预测已错误")
                else:
                    print("  ❌ 生成失败")
                
                # 显示详细信息（仅当有对抗样本时）
                if result["generation_success"]:
                    print(f"    扰动类型: {result['perturbation_type']}")
                    print(f"    BERT-F1: {result['bert_f1']:.4f}")
                    print(f"    原始预测→对抗预测: {result['original_prediction']} → {result['adversarial_prediction']}")
                    
                    # 检查对抗对话是否与原始相同
                    if result["adversarial_dialogue"] == result["original_dialogue"]:
                        print("  ⚠️ 注意：对抗对话与原始对话相同！")
                    else:
                        # 显示修改对比
                        original_brief = result["original_dialogue"][:80] + "..." if len(result["original_dialogue"]) > 80 else result["original_dialogue"]
                        adv_brief = result["adversarial_dialogue"][:80] + "..." if len(result["adversarial_dialogue"]) > 80 else result["adversarial_dialogue"]
                        print(f"  修改前: {original_brief}")
                        print(f"  修改后: {adv_brief}")
                
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
                true_label_chinese = label_to_chinese(true_label)
                error_result = {
                    "sample_id": idx,
                    "original_dialogue": dialogue,
                    "adversarial_dialogue": dialogue,
                    "perturbation_type": "none",
                    "fidelity": 0.0,
                    "bert_precision": 0.0,
                    "bert_recall": 0.0,
                    "bert_f1": 0.0,
                    "similarity_method": "error",
                    "generation_success": False,
                    "attack_success": False,
                    "true_label": true_label_chinese,
                    "original_prediction": "error",
                    "adversarial_prediction": "error",
                    "original_correct": False,
                    "selected_perturbations": ', '.join(selected_perturbations),
                    "row_index": idx,
                    "error": str(e)
                }
                all_results.append(error_result)
                current_batch_results.append(error_result)
                continue
            
            # 每BATCH_SIZE条保存一次
            if len(current_batch_results) >= BATCH_SIZE:
                batch_count += 1
                print(f"\n{'='*80}")
                print(f"完成第 {batch_count} 批 ({BATCH_SIZE}条)，正在保存...")
                
                # 保存所有结果
                save_df = pd.DataFrame(all_results)
                save_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
                
                # 计算本批统计（使用正确的ASR公式）
                batch_original_correct = [r for r in current_batch_results if r.get('original_correct', False)]
                if batch_original_correct:
                    # 原始正确且生成成功的样本（分母）
                    batch_valid_for_asr = [r for r in batch_original_correct if r.get('generation_success', False)]
                    batch_valid_count = len(batch_valid_for_asr)
                    
                    # 攻击成功的样本（分子）：原始正确∩生成成功∩对抗错误
                    batch_attack_success = sum(1 for r in batch_valid_for_asr if r.get('attack_success', False))
                    
                    batch_gen_success = sum(1 for r in batch_original_correct if r.get('generation_success', False))
                    batch_gen_success_but_fail = sum(1 for r in batch_original_correct if r.get('generation_success', False) and not r.get('attack_success', False))
                    batch_correct_count = len(batch_original_correct)
                    
                    # 正确的ASR计算
                    batch_paper_asr = batch_attack_success / batch_valid_count * 100 if batch_valid_count > 0 else 0
                else:
                    batch_attack_success = 0
                    batch_gen_success = 0
                    batch_gen_success_but_fail = 0
                    batch_correct_count = 0
                    batch_valid_count = 0
                    batch_paper_asr = 0
                
                # 计算总体统计（使用正确的ASR公式）
                all_original_correct = [r for r in all_results if r.get('original_correct', False)]
                if all_original_correct:
                    # 原始正确且生成成功的样本（分母）
                    all_valid_for_asr = [r for r in all_original_correct if r.get('generation_success', False)]
                    total_valid_count = len(all_valid_for_asr)
                    
                    # 攻击成功的样本（分子）
                    total_attack_success = sum(1 for r in all_valid_for_asr if r.get('attack_success', False))
                    
                    total_gen_success = sum(1 for r in all_original_correct if r.get('generation_success', False))
                    total_gen_success_but_fail = sum(1 for r in all_original_correct if r.get('generation_success', False) and not r.get('attack_success', False))
                    total_correct_count = len(all_original_correct)
                    
                    # 正确的ASR计算
                    total_paper_asr = total_attack_success / total_valid_count * 100 if total_valid_count > 0 else 0
                    
                    # 计算平均BERTScore（攻击成功的）
                    successful_attacks = [r for r in all_valid_for_asr if r.get('attack_success', False)]
                    gen_success_samples = [r for r in all_valid_for_asr if r.get('generation_success', False)]
                    
                    if successful_attacks:
                        avg_f1_success = sum(r.get('bert_f1', 0) for r in successful_attacks) / len(successful_attacks)
                    else:
                        avg_f1_success = 0
                    
                    if gen_success_samples:
                        avg_f1_gen = sum(r.get('bert_f1', 0) for r in gen_success_samples) / len(gen_success_samples)
                    else:
                        avg_f1_gen = 0
                else:
                    total_attack_success = 0
                    total_gen_success = 0
                    total_gen_success_but_fail = 0
                    total_correct_count = 0
                    total_valid_count = 0
                    total_paper_asr = 0
                    avg_f1_success = 0
                    avg_f1_gen = 0
                
                print(f"本批统计:")
                print(f"  原始预测正确的样本数: {batch_correct_count}")
                print(f"  其中生成成功的样本数: {batch_valid_count}")
                print(f"  在这些生成成功样本中攻击成功的样本数: {batch_attack_success}")
                print(f"  本批攻击成功率(ASR): {batch_paper_asr:.2f}%")
                print(f"累计统计:")
                print(f"  原始预测正确的样本数: {total_correct_count}")
                print(f"  其中生成成功的样本数: {total_valid_count}")
                print(f"  在这些生成成功样本中攻击成功的样本数: {total_attack_success}")
                print(f"  总体攻击成功率(ASR): {total_paper_asr:.2f}%")
                if successful_attacks:
                    print(f"  攻击成功样本的平均BERT-F1: {avg_f1_success:.4f}")
                if gen_success_samples:
                    print(f"  所有生成成功样本的平均BERT-F1: {avg_f1_gen:.4f}")
                print(f"结果已保存到: {OUTPUT_CSV}")
                
                # 重置当前批次
                current_batch_results = []
            
            time.sleep(SLEEP_TIME)
        
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
    
    # 保存最后一批（如果还有）
    if current_batch_results or all_results:
        try:
            save_df = pd.DataFrame(all_results)
            save_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"结果已保存到: {OUTPUT_CSV}")
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    # 最终汇总分析（使用正确的ASR公式）
    if all_results:
        try:
            final_df = pd.DataFrame(all_results)
            
            print(f"\n{'='*80}")
            print("处理完成！最终统计（按论文要求计算，使用正确的ASR公式）")
            print(f"{'='*80}")
            
            # 1. 原始预测正确的样本
            original_correct_df = final_df[final_df['original_correct'] == True]
            total_original_correct = len(original_correct_df)
            
            # 2. 在这些样本中生成成功的样本（分母）
            valid_for_asr_df = original_correct_df[original_correct_df['generation_success'] == True]
            total_valid_for_asr = len(valid_for_asr_df)
            
            # 3. 在这些生成成功样本中攻击成功的样本（分子）
            successful_attacks_df = valid_for_asr_df[valid_for_asr_df['attack_success'] == True]
            total_successful_attacks = len(successful_attacks_df)
            
            # 4. 生成成功的样本数（无论攻击是否成功）
            gen_success_df = original_correct_df[original_correct_df['generation_success'] == True]
            total_gen_success = len(gen_success_df)
            
            # 5. 生成成功但攻击失败的样本数
            gen_success_but_fail_df = original_correct_df[(original_correct_df['generation_success'] == True) & 
                                                         (original_correct_df['attack_success'] == False)]
            total_gen_success_but_fail = len(gen_success_but_fail_df)
            
            # 6. 论文攻击成功率 (ASR) - 使用正确的公式
            if total_valid_for_asr > 0:
                paper_asr = total_successful_attacks / total_valid_for_asr * 100
            else:
                paper_asr = 0
            
            # 7. 生成成功率
            if total_original_correct > 0:
                gen_success_rate = total_gen_success / total_original_correct * 100
            else:
                gen_success_rate = 0
            
            print(f"【论文指标】攻击成功率 (Attack Success Rate, ASR):")
            print(f"  分母: LLM原始预测正确且对抗样本生成成功的样本数 = {total_valid_for_asr}")
            print(f"  分子: 原始正确∩生成成功∩对抗错误 = {total_successful_attacks}")
            print(f"  ASR = {paper_asr:.2f}% ({total_successful_attacks}/{total_valid_for_asr})")
            print(f"  详细说明:")
            print(f"  - 原始预测正确的样本数: {total_original_correct}")
            print(f"  - 其中生成成功的样本数: {total_gen_success}")
            print(f"  - 在这些生成成功的样本中攻击成功的样本数: {total_successful_attacks}")
            
            print(f"\n【生成统计】:")
            print(f"  生成成功的样本数: {total_gen_success} ({gen_success_rate:.1f}%)")
            print(f"  其中攻击失败的样本数: {total_gen_success_but_fail}")
            
            # 7. 总体统计
            total_processed = len(final_df)
            print(f"\n【总体统计】:")
            print(f"  总处理样本: {total_processed}")
            print(f"  原始预测正确的样本: {total_original_correct} ({total_original_correct/total_processed*100:.1f}%)")
            print(f"  原始预测错误的样本: {total_processed - total_original_correct} ({(total_processed-total_original_correct)/total_processed*100:.1f}%)")
            
            # 8. 按标签统计（使用正确的ASR公式）
            print(f"\n【按标签统计】:")
            fraud_df = final_df[final_df['true_label'] == '诈骗']
            non_fraud_df = final_df[final_df['true_label'] == '非诈骗']
            
            if len(fraud_df) > 0:
                fraud_correct = fraud_df[fraud_df['original_correct'] == True]
                if len(fraud_correct) > 0:
                    # 诈骗类：原始正确且生成成功（分母）
                    fraud_valid = fraud_correct[fraud_correct['generation_success'] == True]
                    fraud_valid_count = len(fraud_valid)
                    
                    # 诈骗类：攻击成功（分子）
                    fraud_attacks = fraud_valid[fraud_valid['attack_success'] == True]
                    fraud_attack_count = len(fraud_attacks)
                    
                    fraud_gen_success = fraud_correct[fraud_correct['generation_success'] == True]
                    fraud_gen_count = len(fraud_gen_success)
                    
                    # 正确的ASR计算
                    fraud_asr = fraud_attack_count / fraud_valid_count * 100 if fraud_valid_count > 0 else 0
                    fraud_gen_rate = fraud_gen_count / len(fraud_correct) * 100 if len(fraud_correct) > 0 else 0
                    
                    print(f"  诈骗样本: 总数={len(fraud_df)}, 原始正确={len(fraud_correct)}")
                    print(f"            有效攻击样本={fraud_valid_count}, 攻击成功={fraud_attack_count}")
                    print(f"            ASR={fraud_asr:.2f}%, 生成成功率={fraud_gen_rate:.1f}%")
            
            if len(non_fraud_df) > 0:
                non_fraud_correct = non_fraud_df[non_fraud_df['original_correct'] == True]
                if len(non_fraud_correct) > 0:
                    # 非诈骗类：原始正确且生成成功（分母）
                    non_fraud_valid = non_fraud_correct[non_fraud_correct['generation_success'] == True]
                    non_fraud_valid_count = len(non_fraud_valid)
                    
                    # 非诈骗类：攻击成功（分子）
                    non_fraud_attacks = non_fraud_valid[non_fraud_valid['attack_success'] == True]
                    non_fraud_attack_count = len(non_fraud_attacks)
                    
                    non_fraud_gen_success = non_fraud_correct[non_fraud_correct['generation_success'] == True]
                    non_fraud_gen_count = len(non_fraud_gen_success)
                    
                    # 正确的ASR计算
                    non_fraud_asr = non_fraud_attack_count / non_fraud_valid_count * 100 if non_fraud_valid_count > 0 else 0
                    non_fraud_gen_rate = non_fraud_gen_count / len(non_fraud_correct) * 100 if len(non_fraud_correct) > 0 else 0
                    
                    print(f"  非诈骗样本: 总数={len(non_fraud_df)}, 原始正确={len(non_fraud_correct)}")
                    print(f"              有效攻击样本={non_fraud_valid_count}, 攻击成功={non_fraud_attack_count}")
                    print(f"              ASR={non_fraud_asr:.2f}%, 生成成功率={non_fraud_gen_rate:.1f}%")
            
            # 9. BERTScore统计
            if total_successful_attacks > 0:
                avg_f1_success = successful_attacks_df['bert_f1'].mean()
                avg_precision_success = successful_attacks_df['bert_precision'].mean()
                avg_recall_success = successful_attacks_df['bert_recall'].mean()
                
                print(f"\n【BERTScore统计 (攻击成功样本)】:")
                print(f"  平均Precision: {avg_precision_success:.4f}")
                print(f"  平均Recall:    {avg_recall_success:.4f}")
                print(f"  平均F1:        {avg_f1_success:.4f}")
            
            if total_gen_success > 0:
                avg_f1_gen = gen_success_df['bert_f1'].mean()
                avg_precision_gen = gen_success_df['bert_precision'].mean()
                avg_recall_gen = gen_success_df['bert_recall'].mean()
                
                print(f"\n【BERTScore统计 (所有生成成功样本)】:")
                print(f"  平均Precision: {avg_precision_gen:.4f}")
                print(f"  平均Recall:    {avg_recall_gen:.4f}")
                print(f"  平均F1:        {avg_f1_gen:.4f}")
            
            # 10. 扰动类型分布
            if total_successful_attacks > 0:
                pert_stats = successful_attacks_df['perturbation_type'].value_counts()
                print(f"\n【扰动类型分布 (攻击成功样本)】:")
                for pert, count in pert_stats.items():
                    percentage = count / total_successful_attacks * 100
                    print(f"  {pert}: {count}次 ({percentage:.1f}%)")
            
            # 11. 所有生成成功的扰动类型分布
            if total_gen_success > 0:
                pert_stats_all = gen_success_df['perturbation_type'].value_counts()
                print(f"\n【扰动类型分布 (所有生成成功样本)】:")
                for pert, count in pert_stats_all.items():
                    percentage = count / total_gen_success * 100
                    print(f"  {pert}: {count}次 ({percentage:.1f}%)")
            
            print(f"\n✅ 结果已保存到: {OUTPUT_CSV}")
            
            # 显示成功案例
            show_successful_examples_paper_version(successful_attacks_df)
            
        except Exception as e:
            print(f"生成统计信息失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ 没有生成任何结果")

# =========================
# 16. 成功案例展示函数（论文版本）
# =========================

def show_successful_examples_paper_version(results_df: pd.DataFrame, n: int = 5):
    """展示成功的生成示例（论文版本）"""
    if len(results_df) == 0:
        print("没有成功的攻击示例")
        return
    
    print(f"\n{'='*80}")
    print(f"成功攻击示例 (前{min(n, len(results_df))}个)")
    print(f"{'='*80}")
    
    for i, (idx, row) in enumerate(results_df.head(n).iterrows()):
        print(f"\n示例 {i+1}:")
        print(f"样本ID: {row.get('sample_id', 'N/A')}")
        print(f"扰动类型: {row.get('perturbation_type', 'N/A')}")
        print(f"真实标签: {row.get('true_label', 'N/A')}")
        print(f"原始预测: {row.get('original_prediction', 'N/A')} → 对抗预测: {row.get('adversarial_prediction', 'N/A')}")
        print(f"BERT-F1分数: {row.get('bert_f1', 0):.4f}")
        
        orig = row.get('original_dialogue', '')
        adv = row.get('adversarial_dialogue', '')
        
        print(f"\n【原始对话】:")
        # 显示前200字符
        if len(orig) > 200:
            print(orig[:200] + "...")
        else:
            print(orig)
        
        print(f"\n【对抗对话】:")
        if len(adv) > 200:
            print(adv[:200] + "...")
        else:
            print(adv)
        
        # 显示BERTScore详情
        print(f"\n【BERTScore详情】:")
        print(f"  Precision: {row.get('bert_precision', 0):.4f}")
        print(f"  Recall:    {row.get('bert_recall', 0):.4f}")
        print(f"  F1:        {row.get('bert_f1', 0):.4f}")
        print("-" * 80)

# =========================
# 17. 主程序入口
# =========================

if __name__ == "__main__":
    main()