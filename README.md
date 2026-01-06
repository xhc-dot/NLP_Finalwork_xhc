对抗性数据改写在欺诈对话检测中的应用
1. 项目简介
本项目为《自然语言处理》课程大作业，研究对抗性数据改写对欺诈对话检测模型鲁棒性的影响。基于PromptAttack方法生成对抗样本，评估GLM、SVM、BERT三类模型在对抗攻击下的性能变化。

2. 环境要求
- Python 3.9+
- 依赖包：见 `requirements.txt`

3. 项目结构
nlp-fraud-detection-adversarial/
│
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包列表
│
├── code/                     # 所有代码文件
│   ├── Prompts_Attack.py     # 生成对抗性样本的代码，运行时不同的输入会有不同的prompts，实现不同策略的对康熙样本的生成
│   ├── Prompts_Attack_1.py   # 在上面代码的基础上取出来一开始和解为的GLM大模型对对话的类型判断，是我的中间步骤代码，可忽视
│   ├── GLM_Test.py           # 调用GLM大模型对输入的对话进行诈骗/非诈骗判断，修改其中的值，可以进行对原始对话的判断和对对抗性样本对话的判断，可以得出准确率
│   ├── SVM_Train.py          # 使用给出的训练集训练SVM模型并对测试集中的数据进行测试的代码，还有保存训练好的模型的作用
│   ├── SVM_Test_Finalwork.py # 调用训练好的SVM模型对我生成的对抗性样本进行分类，得出结果的代码
│   ├── BERT_Train.py         # 使用给出的训练集训练BERT模型并对测试集中的数据进行测试的代码，还有保存训练好的模型的作用
│   └── BERT_Train.py         # 调用训练好的BERT模型对我生成的对抗性样本进行分类，得出结果的代码

│
├── data/                     # 数据相关
│   ├── 测试集结果.csv         # 原始数据
│   ├── 训练集结果.csv         # 训练集，用于训练SVM、BERT
│   └── promptattack_C1.csv   # 生成的对抗样本示例
│   └── promptattack_W1.csv   # 生成的对抗样本示例
│   └── promptattack_S1.csv   # 生成的对抗样本示例
│
├── models/                   # 训练好的模型
│   ├── svm_fraud_model.pkl
|   ├── tfidf_vectorizer.pkl
│   └── bert_fraud_full_model.pth #模型过大

│
├── results/                  # 实验结果
│   ├──运行结果图片.zip
└──

4.作者
- 薛涵畅
- 深圳大学计算机科学与技术学院