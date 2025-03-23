import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

warnings.filterwarnings('ignore')
import pickle
import os

# 设置绘图风格
plt.style.use('ggplot')
sns.set(style='whitegrid')

# 定义结果保存目录
results_dir = '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Group Project/GPv1/results'
os.makedirs(results_dir, exist_ok=True)
models_dir = '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Group Project/GPv1/models'
os.makedirs(models_dir, exist_ok=True)


# 1. 数据加载与探索
def load_and_explore_data(file_path):
    """加载数据并进行初步探索"""
    print("=" * 50)
    print("Data loading and exploration")
    print("=" * 50)

    # 读取CSV文件 - 使用传入的文件路径
    df = pd.read_csv(file_path)
    df['CLASS'] = df['CLASS'].str.strip()  # 去除空格
    # 显示数据基本信息
    print(f"Shape of the dataset: {df.shape}")
    print(df.head())

    # 数据类型和缺失值
    print("\nData Type and Missing Value:")
    print(df.info())

    # 统计描述
    print("\nNumerical feature Description:")
    print(df.describe())

    # 目标变量分布
    print("\nDistribution of target variable:")
    print(df['CLASS'].value_counts())
    print(df['CLASS'].value_counts(normalize=True) * 100)

    # 保存数据分布图
    plt.figure(figsize=(10, 6))
    sns.countplot(x='CLASS', data=df)
    plt.title('Distribution of target variable')
    plt.savefig(f"{results_dir}/class_distribution.png")

    return df


# 2. 数据预处理
def preprocess_data(df):
    """数据清洗与预处理"""
    print("=" * 50)
    print("Data Preprocessing")
    print("=" * 50)

    # 复制数据避免修改原数据
    data = df.copy()

    # 移除无用特征
    if 'ID' in data.columns:
        data = data.drop(['ID', 'No_Pation'], axis=1)

    # 检查缺失值
    print("\nMissing value statistics:")
    print(data.isnull().sum())

    # 检查重复值
    duplicates = data.duplicated().sum()
    print(f"\n重复记录数: {duplicates}")
    if duplicates > 0:
        data = data.drop_duplicates()
        print(f"移除重复记录后的数据形状: {data.shape}")

    # 编码性别变量
    data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})

    # 对目标变量编码
    le = LabelEncoder()
    data['CLASS_encoded'] = le.fit_transform(data['CLASS'])
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("\n目标变量编码映射:")
    print(class_mapping)

    # 检查异常值
    print("\n检查数值特征的异常值...")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['CLASS_encoded']]

    # 使用IQR方法检测异常值
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].shape[0]
        if outliers > 0:
            print(f"{col}: {outliers} 个异常值检测到")
            # 异常值处理：将超出范围的值截断至边界值
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

    # 保存预处理后的数据
    data.to_csv(f"{results_dir}/preprocessed_data.csv", index=False)

    # 返回处理后的特征和目标变量
    X = data.drop(['CLASS', 'CLASS_encoded'], axis=1)
    y = data['CLASS_encoded']
    feature_names = X.columns.tolist()

    return X, y, feature_names, class_mapping


# 3. 特征分析与选择
def analyze_and_select_features(X, y, feature_names):
    """分析特征重要性并选择最佳特征"""
    print("=" * 50)
    print("Feature Analysis and Selection")
    print("=" * 50)

    # 首先进行缺失值处理
    print("\n使用SimpleImputer处理特征中的缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names)

    # 相关性分析
    print("\nCalculating feature correlation matrix...")
    corr_matrix = X_imputed_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature correlation matrix')
    plt.savefig(f"{results_dir}/correlation_matrix.png")

    # 单变量特征选择 (ANOVA F-value)
    print("\nUsing ANOVA F-value to select features...")
    selector = SelectKBest(f_classif, k='all')
    selector.fit(X_imputed, y)
    feature_scores = pd.DataFrame({
        'Feature': feature_names,
        'F_Score': selector.scores_,
        'P_Value': selector.pvalues_
    })
    feature_scores = feature_scores.sort_values('F_Score', ascending=False)
    print(feature_scores)

    # 绘制特征重要性
    plt.figure(figsize=(12, 6))
    sns.barplot(x='F_Score', y='Feature', data=feature_scores)
    plt.title('Feature Importance (ANOVA F-Score)')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/feature_importance_anova.png")

    # 使用随机森林评估特征重要性
    print("\nUsing Random Forest to evaluate feature importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_imputed, y)
    rf_feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    })
    rf_feature_importance = rf_feature_importance.sort_values('Importance', ascending=False)
    print(rf_feature_importance)

    # 绘制随机森林特征重要性
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=rf_feature_importance)
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/feature_importance_rf.png")

    # 选择前k个最重要的特征
    k = min(10, len(feature_names))  # 选择最多10个特征
    top_features = rf_feature_importance.head(k)['Feature'].tolist()
    print(f"\n选择的前{k}个重要特征: {top_features}")

    return top_features, rf_feature_importance, feature_scores


# 4. 特征工程与数据准备
def prepare_data_for_modeling(X, y, top_features=None):
    """准备训练和测试数据"""
    print("=" * 50)
    print("Feature Engineering and Data Preparation")
    print("=" * 50)

    # 如果指定了特征列表，则只选择这些特征
    if top_features:
        X = X[top_features]
        print(f"使用选定的{len(top_features)}个特征: {top_features}")

    # 创建并应用新的imputer（避免特征名称不匹配问题）
    print("\n创建并应用新的imputer填充缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # 标准化特征
    print("\n标准化特征...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 检查各类别的样本数量
    class_counts = pd.Series(y).value_counts()
    print("\n每个类别的样本数量:")
    print(class_counts)

    # 检查是否有类只有1个样本
    min_samples_per_class = class_counts.min()
    if min_samples_per_class < 2:
        print(f"\n警告: 最小类别只有 {min_samples_per_class} 个样本，无法进行分层抽样。使用随机抽样替代。")

        # 移除只有一个样本的类别
        print("移除样本数量少于2的类别...")
        classes_to_keep = class_counts[class_counts >= 2].index
        mask = y.isin(classes_to_keep)
        X_scaled = X_scaled[mask]
        y = y[mask]

        print(f"移除后的数据形状: X: {X_scaled.shape}, y: {y.shape}")

        # 进行分层抽样
        print("\n划分训练集和测试集 (80%-20%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        # 进行分层抽样
        print("\n划分训练集和测试集 (80%-20%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print(f"训练集目标分布: \n{pd.Series(y_train).value_counts(normalize=True) * 100}")
    print(f"测试集目标分布: \n{pd.Series(y_test).value_counts(normalize=True) * 100}")

    return X_train, X_test, y_train, y_test, scaler, imputer, top_features


# 5. 模型训练与评估
def train_and_evaluate_models(X_train, X_test, y_train, y_test, class_mapping):
    """训练多个模型并评估其性能"""
    print("=" * 50)
    print("Model Training and Evaluation")
    print("=" * 50)

    # 定义模型 - 修复警告
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(max_iter=2000, random_state=42),  # 增加迭代次数
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # 超参数网格 - 移除deprecate的参数
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
            # 移除 'multi_class' 参数
        },
        'Random Forest': {
            'n_estimators': [50,100, 200],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 15]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    }

    # 存储模型结果
    results = {}
    best_models = {}

    # 训练并评估每个模型
    for name, model in models.items():
        print(f"\nModel Training: {name}...")

        # 使用网格搜索找到最佳超参数
        print(f"Hyperparameter tuning...")
        grid_search = GridSearchCV(
            model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        print(f"Best parameters: {best_params}")
        print(f"Cross-validation score: {grid_search.best_score_:.4f}")

        # 在测试集上评估
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        y_train_pred = best_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"Train-set Accuracy: {train_accuracy:.4f}")

        # **添加ROC-AUC曲线**
        # 二值化测试集标签（One-vs-Rest）
        classes = list(class_mapping.values())
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = len(classes)

        # 计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 绘制 ROC 曲线
        plt.figure(figsize=(10, 8))

        for i in range(n_classes):
            fpr_interp = np.linspace(0, 1, 100)  # 生成 100 个均匀分布的 FPR 点

            # 解决 FPR 不能有重复值的问题
            unique_fpr, unique_indices = np.unique(fpr[i], return_index=True)
            unique_tpr = tpr[i][unique_indices]

            # 进行插值
            interp_func = interp1d(unique_fpr, unique_tpr, kind='linear')  # 也可改为 'linear'
            tpr_interp = interp_func(fpr_interp)  # 计算对应的 TPR 值

            # 绘制平滑后的 ROC 曲线
            plt.plot(fpr_interp, tpr_interp, label=f'Class {classes[i]} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # 对角线
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (One-vs-Rest) for {name}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{name.replace(' ', '_')}_roc_curve.png")
        plt.close()

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # 计算ROC-AUC（多类别情况下使用OVR方法）
        if len(np.unique(y_test)) > 2:
            # 多类别情况
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            # 二类别情况
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        print(f"Test-set Accuracy: {accuracy:.4f}")
        print(f"Test-set Precision: {precision:.4f}")
        print(f"Test-set Recall: {recall:.4f}")
        print(f"Test-set F1-score: {f1:.4f}")
        print(f"Test-set ROC-AUC: {roc_auc:.4f}")

        # 保存结果
        results[name] = {
            'model': best_model,
            'params': best_params,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # 打印分类报告
        print("\nClassification Report:")
        class_labels = [f"{k} ({v})" for k, v in class_mapping.items()]
        print(classification_report(y_test, y_pred, target_names=class_labels))

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('Real Label')
        plt.title(f'{name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{name.replace(' ', '_')}_confusion_matrix.png")

        # 保存模型
        best_models[name] = best_model
        with open(f"{models_dir}/{name.replace(' ', '_')}_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)

        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)  # 训练集大小从10%到100%
        )

        # 计算平均准确率
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        # 绘制学习曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training Accuracy')
        plt.plot(train_sizes, val_mean, label='Validation Accuracy')
        plt.title(f'Learning Curve for {name}')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{name.replace(' ', '_')}_learning_curve.png")
        plt.close()

    # 比较所有模型性能
    compare_models(results)

    return results, best_models


# 6. 模型比较和可视化
def compare_models(results):
    """比较所有模型的性能并可视化"""
    print("=" * 50)
    print("Models Comparison")
    print("=" * 50)

    # 提取各模型的性能指标
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_comparison = pd.DataFrame(index=results.keys(), columns=metrics)

    for name, result in results.items():
        for metric in metrics:
            model_comparison.loc[name, metric] = result[metric]

    print("\nModel Performance Comparison:")
    print(model_comparison)

    # 找出最佳模型
    best_model_name = model_comparison['accuracy'].idxmax()
    print(f"\nBaseline Model: {best_model_name}")
    print(f"Accuracy: {model_comparison.loc[best_model_name, 'accuracy']:.4f}")

    # 可视化模型比较
    plt.figure(figsize=(14, 8))
    model_comparison.plot(kind='bar', figsize=(14, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Performance Score')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/model_comparison.png")

    # 绘制每个指标的比较图
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_comparison.index, y=model_comparison[metric])
        plt.title(f'{metric} Comparison')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/model_comparison_{metric}.png")


# 7. 创建特征重要性分析函数
def analyze_feature_importance(best_model, feature_names, model_name):
    """分析并可视化特征重要性"""
    print("=" * 50)
    print(f"{model_name} Feature Importance Analysis")
    print("=" * 50)

    # 获取特征重要性 (适用于树模型)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # 打印特征重要性
        print("\nFeature Importance Ranking:")
        for i, idx in enumerate(indices):
            print(f"{i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

        # 绘制特征重要性
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{model_name.replace(' ', '_')}_feature_importance.png")

    # 对于逻辑回归，可以查看系数
    elif hasattr(best_model, 'coef_'):
        # 对于二分类问题
        if len(best_model.classes_) == 2:
            coefficients = best_model.coef_[0]
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
            feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

            # 打印系数
            print("\nFeature Coefficients Ranking:")
            for i, row in feature_importance.iterrows():
                print(f"{i + 1}. {row['Feature']}: {row['Coefficient']:.4f}")

            # 绘制系数
            plt.figure(figsize=(12, 6))
            colors = ['green' if c > 0 else 'red' for c in feature_importance['Coefficient']]
            plt.bar(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
            plt.xticks(rotation=90)
            plt.title(f'{model_name} Feature Coefficients')
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{model_name.replace(' ', '_')}_coefficients.png")

        # 对于多分类问题
        else:
            plt.figure(figsize=(15, 10))
            for i, cls in enumerate(best_model.classes_):
                plt.subplot(len(best_model.classes_), 1, i + 1)
                plt.bar(feature_names, best_model.coef_[i])
                plt.title(f'Feature Coefficients for Class {cls}')
                plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{model_name.replace(' ', '_')}_multi_coefficients.png")

    else:
        print(f"Warning: Can't analyze feature importance directly for {model_name}.")


# 8. 主函数
def main():
    """主函数"""
    # 加载数据 - 使用正确的本地文件路径
    file_path = '/Users/zhujun/LU/Term-2/CDS524-Machine Learning for Business/Group Project/Diabetes Dataset/Dataset of Diabetes_updated .csv'
    df = load_and_explore_data(file_path)

    # 数据预处理
    X, y, feature_names, class_mapping = preprocess_data(df)

    # 特征分析与选择
    top_features, rf_importance, f_scores = analyze_and_select_features(X, y, feature_names)

    # 准备模型训练数据
    X_train, X_test, y_train, y_test, scaler, imputer, selected_features = prepare_data_for_modeling(X, y, top_features)

    # 更新类别映射以只包含训练数据中出现的类别
    unique_classes = np.unique(np.concatenate([y_train, y_test]))
    class_mapping = {k: v for k, v in class_mapping.items() if v in unique_classes}

    # 训练和评估模型
    results, best_models = train_and_evaluate_models(X_train, X_test, y_train, y_test, class_mapping)

    # 分析最佳模型的特征重要性
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = best_models[best_model_name]
    analyze_feature_importance(best_model, selected_features, best_model_name)

    print("\n分析完成！所有结果都保存在 'results' 目录中，模型保存在 'models' 目录中。")

    # 保存预处理信息，用于后续预测
    preprocessing_info = {
        'scaler': scaler,
        'imputer': imputer,
        'top_features': selected_features,
        'class_mapping': class_mapping
    }
    with open(f"{models_dir}/preprocessing_info.pkl", 'wb') as f:
        pickle.dump(preprocessing_info, f)

    return best_model, preprocessing_info


# 执行主函数
if __name__ == '__main__':
    best_model, preprocessing_info = main()