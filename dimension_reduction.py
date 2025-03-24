
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set(style='whitegrid')

results_dir = r'F:\各种作业\machine_learnning\Diabetes Dataset\results'
os.makedirs(results_dir, exist_ok=True)
models_dir = r'F:\各种作业\machine_learnning\Diabetes Dataset\models'
os.makedirs(models_dir, exist_ok=True)

# 预处理
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['CLASS'] = df['CLASS'].str.strip()
    
    data = df.drop(['ID', 'No_Pation'], axis=1)
    data = data.drop_duplicates()
    data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
    
    le = LabelEncoder()
    data['CLASS_encoded'] = le.fit_transform(data['CLASS'])
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    numeric_cols = [col for col in data.columns if col not in ['CLASS', 'CLASS_encoded']]
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
    
    X = data.drop(['CLASS', 'CLASS_encoded'], axis=1)
    y = data['CLASS_encoded']
    feature_names = X.columns.tolist()
    
    return X, y, feature_names, class_mapping

# 数据准备与降维
def prepare_data_with_dimensionality_reduction(X, y):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed = np.nan_to_num(X_imputed, nan=np.nanmean(X_imputed), posinf=np.nanmax(X_imputed), neginf=np.nanmin(X_imputed))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # SVD
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    k = 6
    X_svd = np.dot(U[:, :k], np.diag(S[:k]))
    
    # PCA
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_scaled)
    
    # Rotated PCA
    fa = FactorAnalyzer(n_factors=6, rotation='varimax')
    X_rotated_pca = fa.fit_transform(X_scaled)
    
    class_counts = pd.Series(y).value_counts()
    min_samples_per_class = class_counts.min()
    if min_samples_per_class < 2:
        classes_to_keep = class_counts[class_counts >= 2].index
        mask = y.isin(classes_to_keep)
        X_scaled = X_scaled[mask]
        X_svd = X_svd[mask]
        X_pca = X_pca[mask]
        X_rotated_pca = X_rotated_pca[mask]
        y = y[mask]
    
    data_splits = {}
    for name, X_data in [('Original', X_scaled), ('SVD', X_svd), ('PCA', X_pca), ('Rotated PCA', X_rotated_pca)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y, test_size=0.2, random_state=42, stratify=y
        )
        data_splits[name] = (X_train, X_test, y_train, y_test)
    
    return data_splits, scaler, imputer

# 模型训练
def train_and_evaluate_models(data_splits, class_mapping):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }

    param_grids = {
        'Logistic Regression': {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs'], 'multi_class': ['auto', 'ovr']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'Gradient Boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'Neural Network': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001], 'learning_rate': ['constant', 'adaptive']}
    }

    results = {}
    for method in data_splits:
        results[method] = {}
        X_train, X_test, y_train, y_test = data_splits[method]
        
        for name, model in models.items():
            print(f"Training {name} with {method} data...")
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=1)  # 修改为单线程
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            cm = confusion_matrix(y_test, y_pred)
            
            results[method][name] = {
                'f1': f1,
                'auc': auc_score,
                'cm': cm
            }
            
            with open(f"{models_dir}\\{name.replace(' ', '_')}_{method.replace(' ', '_')}_model.pkl", 'wb') as f:
                pickle.dump(best_model, f)
    
    return results

# 可视化
def compare_results(results):
    metrics = ['f1', 'auc']
    methods = list(results.keys())
    model_names = list(results[methods[0]].keys())
    
    comparison_df = pd.DataFrame(index=[f"{model} ({method})" for model in model_names for method in methods],
                                 columns=metrics)
    
    for method in methods:
        for model in model_names:
            comparison_df.loc[f"{model} ({method})", 'f1'] = results[method][model]['f1']
            comparison_df.loc[f"{model} ({method})", 'auc'] = results[method][model]['auc']
    
    print("\nF1 和 AUC 对比表:")
    print(comparison_df)
    comparison_df.to_csv(f"{results_dir}\\f1_auc_comparison.csv")

    for model in model_names:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        for i, method in enumerate(methods):
            cm = results[method][model]['cm']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{method}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        plt.suptitle(f'{model} Confusion Matrices', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{results_dir}\\{model.replace(' ', '_')}_confusion_matrices.png")
        plt.close()


def main():
    X, y, feature_names, class_mapping = load_and_preprocess_data(
        r'F:\各种作业\machine_learnning\Diabetes Dataset\Dataset of Diabetes .csv'
    )
    
    data_splits, scaler, imputer = prepare_data_with_dimensionality_reduction(X, y)
    results = train_and_evaluate_models(data_splits, class_mapping)
    compare_results(results)
    
    preprocessing_info = {'scaler': scaler, 'imputer': imputer, 'feature_names': feature_names, 'class_mapping': class_mapping}
    with open(f"{models_dir}\\preprocessing_info.pkl", 'wb') as f:
        pickle.dump(preprocessing_info, f)
    

if __name__ == '__main__':
    main()




# 可视化
plt.style.use('ggplot')
sns.set(style='whitegrid')
file_path = r'F:\各种作业\machine_learnning\Diabetes Dataset\results\f1_auc_comparison.csv'
results_df = pd.read_csv(file_path, index_col=0)
results_df.columns = ['F1 Score', 'AUC']

# F1 分数
plt.figure(figsize=(14, 6))
sns.barplot(x=results_df.index, y='F1 Score', data=results_df, palette='viridis')
plt.title('F1 Score Comparison Across Models and Methods', fontsize=16)
plt.xlabel('Model (Method)', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(r'F:\各种作业\machine_learnning\Diabetes Dataset\results\f1_score_comparison.png')
plt.close()

# AUC 值
plt.figure(figsize=(14, 6))
sns.barplot(x=results_df.index, y='AUC', data=results_df, palette='magma')
plt.title('AUC Comparison Across Models and Methods', fontsize=16)
plt.xlabel('Model (Method)', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(r'F:\各种作业\machine_learnning\Diabetes Dataset\results\auc_comparison.png')
plt.close()
