对抗性数据改写在欺诈对话检测中的应用
1. 项目简介
本项目为《自然语言处理》课程大作业，研究对抗性数据改写对欺诈对话检测模型鲁棒性的影响。基于PromptAttack方法生成对抗样本，评估GLM、SVM、BERT三类模型在对抗攻击下的性能变化。

2. 环境要求
- Python 3.9+
- 依赖包：见 `requirements.txt`

3. 项目结构
nlp-fraud-detection-adversarial/
│
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
│
├── code/                        # 所有代码文件
│   ├── Prompts_Attack.py        # 生成对抗性样本的代码
│   ├── Prompts_Attack_1.py      # 中间步骤代码（可忽略）
│   ├── GLM_Test.py              # 调用GLM进行诈骗判断
│   ├── SVM_Train.py             # 训练SVM模型
│   ├── SVM_Test_Finalwork.py    # 用SVM测试对抗样本
│   ├── BERT_Train.py            # 训练BERT模型
│   └── BERT_Test_Finalwork.py   # 用BERT测试对抗样本
│
├── data/                        # 数据相关
│   ├── 测试集结果.csv           # 原始测试数据
│   ├── 训练集结果.csv           # 训练数据
│   ├── promptattack_C1.csv      # C1对抗样本
│   ├── promptattack_W1.csv      # W1对抗样本
│   └── promptattack_S1.csv      # S1对抗样本
│
├── models/                      # 训练好的模型
│   ├── svm_fraud_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── bert_fraud_full_model.pth  # 文件过大需单独处理
│
└── results/                     # 实验结果
    └── 运行结果图片.zip

bert_fraud_full_model.pth #模型过大通过网盘分享的文件：BERT模型.zip链接: https://pan.baidu.com/s/1eFiYG9__xmeS_4JAmhRe0A?pwd=rjrs 提取码: rjrs 复制这段内容后打开百度网盘手机App，操作更方便哦
4.作者
- 薛涵畅
- 深圳大学计算机科学与技术学院