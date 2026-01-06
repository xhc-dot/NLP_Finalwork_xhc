对抗性数据改写在欺诈对话检测中的应用
# 项目简介
本项目为《自然语言处理》课程大作业，研究对抗性数据改写对欺诈对话检测模型鲁棒性的影响。基于PromptAttack方法生成对抗样本，评估GLM、SVM、BERT三类模型在对抗攻击下的性能变化。

## 环境要求
- Python 3.9+
- 依赖包：见requirements.txt

## 项目结构

```text
NLP_Finalwork_xhc/
├── README.md                  # 项目说明文档
├── requirements.txt           # Python 依赖包列表
├── code/                      # 所有代码文件
│   ├── Prompts_Attack.py      # 生成对抗性样本
│   ├── Prompts_Attack_1.py    # 中间步骤（可忽略）
│   ├── GLM_Test.py            # GLM 判断诈骗
│   ├── SVM_Train.py           # SVM 训练
│   ├── SVM_Test_Finalwork.py  # SVM 测试
│   ├── BERT_Train.py          # BERT 训练
│   └── BERT_Test_Finalwork.py # BERT 测试
├── data/                      # 数据文件
│   ├── 测试集结果.csv          # 原始数据
│   ├── 训练集结果.csv
│   ├── promptattack_C1.csv    # 生成的对抗性样本
│   ├── promptattack_W1.csv
│   └── promptattack_S1.csv
├── models/                    # 模型文件
│   ├── svm_fraud_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bert_fraud_full_model.pth
├── results/                   # 实验结果
└── 运行结果图片.zip
```
bert_fraud_full_model.pth #模型过大通过网盘分享的文件：BERT模型.zip链接: https://pan.baidu.com/s/1eFiYG9__xmeS_4JAmhRe0A?pwd=rjrs 提取码: rjrs 复制这段内容后打开百度网盘手机App，操作更方便哦

## 作者
- 薛涵畅
- 深圳大学计算机科学与技术学院