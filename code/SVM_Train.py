import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import joblib

def prepare_data(train_path, test_path):
    """
    准备数据并进行预处理
    返回：X_train_tfidf, y_train, X_test_tfidf, y_test, test_data, vectorizer
    """
    # 读取训练集和测试集数据
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 排除训练集中 specific_dialogue_content 为空或者 is_fraud 为空的数据
    train_data = train_data[train_data['specific_dialogue_content'].notna() & (train_data['specific_dialogue_content'] != '')]
    train_data = train_data[train_data['is_fraud'].notna()]

    # 排除测试集中 specific_dialogue_content 为空或者 is_fraud 为空的数据
    test_data = test_data[test_data['specific_dialogue_content'].notna() & (test_data['specific_dialogue_content'] != '')]
    test_data = test_data[test_data['is_fraud'].notna()]

    # 提取训练集的输入文本和真实标签
    X_train_text = train_data['specific_dialogue_content']
    y_train = train_data['is_fraud'].astype(int)

    # 提取测试集的输入文本和真实标签
    X_test_text = test_data['specific_dialogue_content']
    y_test = test_data['is_fraud'].astype(int)

    # 中文分词
    print("正在进行中文分词...")
    X_train_tokens = [' '.join(jieba.cut(text)) for text in X_train_text]
    X_test_tokens = [' '.join(jieba.cut(text)) for text in X_test_text]

    # TF-IDF 向量化
    print("正在进行TF-IDF向量化...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,  # 最小文档频率
        max_df=0.95,  # 最大文档频率
        ngram_range=(1, 2)  # 使用1-2元语法
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_tokens)
    X_test_tfidf = vectorizer.transform(X_test_tokens)
    
    print(f"特征维度: {X_train_tfidf.shape[1]}")
    
    return X_train_tfidf, y_train, X_test_tfidf, y_test, test_data, vectorizer

def train_svm_model(X_train, y_train):
    """
    训练SVM模型
    """
    print("正在训练SVM模型...")
    svm_model = SVC(
        kernel='linear',
        probability=True,
        class_weight='balanced',  # 处理类别不平衡
        random_state=42,
        verbose=True
    )
    svm_model.fit(X_train, y_train)
    print("模型训练完成！")
    return svm_model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    # 进行预测
    y_pred = model.predict(X_test)
    
    # 总准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 诈骗类样本（标签1）的准确率
    fraud_mask = y_test == 1
    if np.sum(fraud_mask) > 0:
        accuracy_fraud = accuracy_score(y_test[fraud_mask], y_pred[fraud_mask])
    else:
        accuracy_fraud = np.nan
    
    # 非诈骗类样本（标签0）的准确率
    nonfraud_mask = y_test == 0
    if np.sum(nonfraud_mask) > 0:
        accuracy_nonfraud = accuracy_score(y_test[nonfraud_mask], y_pred[nonfraud_mask])
    else:
        accuracy_nonfraud = np.nan
    
    print("=" * 60)
    print("模型评估结果:")
    print("=" * 60)
    print(f"总准确率: {accuracy:.4f}")
    print(f"诈骗类准确率: {accuracy_fraud:.4f}")
    print(f"非诈骗类准确率: {accuracy_nonfraud:.4f}")
    
    # 详细的分类报告
    print("\n分类报告:")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=['非诈骗', '诈骗']))
    
    # 混淆矩阵
    print("混淆矩阵:")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negative (非诈骗->非诈骗): {cm[0, 0]}")
    print(f"False Positive (非诈骗->诈骗): {cm[0, 1]}")
    print(f"False Negative (诈骗->非诈骗): {cm[1, 0]}")
    print(f"True Positive (诈骗->诈骗): {cm[1, 1]}")
    
    return y_pred

def save_model_pipeline(model, vectorizer, model_path, vectorizer_path):
    """
    保存完整的模型流水线
    """
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✅ 模型已保存至：{model_path}")
    print(f"✅ 向量化器已保存至：{vectorizer_path}")

def load_model_pipeline(model_path, vectorizer_path):
    """
    加载完整的模型流水线
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ 模型和向量化器加载完成！")
    return model, vectorizer

def predict_new_text(text, model, vectorizer):
    """
    预测新的文本
    """
    # 分词
    tokens = ' '.join(jieba.cut(text))
    # 向量化
    text_vectorized = vectorizer.transform([tokens])
    # 预测
    prediction = model.predict(text_vectorized)
    probability = model.predict_proba(text_vectorized)
    
    return {
        'prediction': '诈骗' if prediction[0] == 1 else '非诈骗',
        'confidence': probability[0][prediction[0]],
        'probabilities': {
            '非诈骗': probability[0][0],
            '诈骗': probability[0][1]
        }
    }

def main():
    # 数据准备与预处理
    train_path = 'D:\\自然语言处理\\实验一\\train_data.csv'
    test_path = 'D:\\自然语言处理\\实验一\\test_data.csv'
    
    print("开始数据预处理...")
    X_train_tfidf, y_train, X_test_tfidf, y_test, test_data, vectorizer = prepare_data(train_path, test_path)
    print(f"训练集大小: {X_train_tfidf.shape[0]}, 测试集大小: {X_test_tfidf.shape[0]}")
    print(f"诈骗样本比例 - 训练集: {y_train.mean():.2%}, 测试集: {y_test.mean():.2%}")
    
    # 训练 SVM 模型
    svm_model = train_svm_model(X_train_tfidf, y_train)
    
    # 保存模型和向量化器
    model_save_path = 'D:\\自然语言处理\\实验一\\svm_fraud_model.pkl'
    vectorizer_save_path = 'D:\\自然语言处理\\实验一\\tfidf_vectorizer.pkl'
    save_model_pipeline(svm_model, vectorizer, model_save_path, vectorizer_save_path)
    
    # 评估模型
    y_pred = evaluate_model(svm_model, X_test_tfidf, y_test)
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'dialogue': test_data['specific_dialogue_content'],
        'true_label': y_test,
        'predicted_label': y_pred
    })
    
    result_path = 'D:\\自然语言处理\\实验一\\prediction_results.csv'
    result_df.to_csv(result_path, index=False)
    print(f"✅ 预测结果已保存至：{result_path}")
    
    # 保存详细预测结果（包含置信度）
    y_prob = svm_model.predict_proba(X_test_tfidf)
    detailed_result_df = pd.DataFrame({
        'dialogue': test_data['specific_dialogue_content'],
        'true_label': y_test,
        'predicted_label': y_pred,
        'prob_non_fraud': y_prob[:, 0],
        'prob_fraud': y_prob[:, 1],
        'is_correct': (y_test == y_pred).astype(int)
    })
    
    detailed_result_path = 'D:\\自然语言处理\\实验一\\detailed_prediction_results.csv'
    detailed_result_df.to_csv(detailed_result_path, index=False)
    print(f"✅ 详细预测结果已保存至：{detailed_result_path}")
    
    # 示例：预测新文本
    print("\n" + "=" * 60)
    print("示例预测：")
    print("=" * 60)
    test_texts = [
        "你好，我是银行客服，需要您提供银行卡密码进行验证",
        "请问这个商品什么时候可以发货？",
        "恭喜您中奖了，请点击链接领取奖品"
    ]
    
    for text in test_texts:
        result = predict_new_text(text, svm_model, vectorizer)
        print(f"\n文本: {text}")
        print(f"预测结果: {result['prediction']}")
        print(f"置信度: {result['confidence']:.2%}")
        print(f"详细概率: 非诈骗 {result['probabilities']['非诈骗']:.2%}, 诈骗 {result['probabilities']['诈骗']:.2%}")

if __name__ == "__main__":
    main()