from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 医学术语列表（使用下划线连接复合词）
medical_terms = [
    "Blood_Sugar_Level", "HBA1C", "BMI", "Creatinine_Ratio", "Urea", "Cholesterol",
    "LDL", "HDL", "VLDL", "Triglycerides", "Fasting_Lipid_Profile", "Diabetic",
    "Non_Diabetic", "Pre_Diabetic", "Diabetes_Mellitus", "Type_2_Diabetes",
    "Type_1_Diabetes", "Insulin_Resistance", "Hyperglycemia", "Hypoglycemia",
    "Glycemic_Control", "Kidney_Function", "Cardiovascular_Risk", "Dyslipidemia",
    "Obesity", "Age", "Gender", "Diabetic_Nephropathy", "Diabetic_Retinopathy",
    "Diabetic_Neuropathy", "Fasting_Blood_Glucose", "Postprandial_Blood_Glucose",
    "Oral_Glucose_Tolerance_Test", "Insulin_Therapy", "Oral_Hypoglycemic_Agents",
    "Lifestyle_Modification", "Dietary_Intake", "Physical_Activity", "Metabolic_Syndrome",
    "Glycemic_Index"
]

# 将术语列表转换为一个长字符串
text = " ".join(medical_terms)

# 创建词云对象
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# 显示词云
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Dataset of Diabetes_updated.csv
# Column Names
# No. of Patient - 患者编号
# Sugar Level Blood - 血糖水平
# Age - 年龄
# Gender - 性别
# Creatinine ratio (Cr) - 肌酐比率
# Body Mass Index (BMI) - 体质指数
# Urea - 尿素
# Cholesterol (Chol) - 胆固醇
# Fasting lipid profile - 空腹血脂谱
#   Total - 总胆固醇
#   LDL - 低密度脂蛋白胆固醇
#   VLDL - 极低密度脂蛋白胆固醇
#   Triglycerides (TG) - 甘油三酯
#   HDL Cholesterol - 高密度脂蛋白胆固醇
# HBA1C - 糖化血红蛋白
# Class - 疾病分类
#   Diabetic - 糖尿病患者
#   Non-Diabetic - 非糖尿病患者
#   Predict-Diabetic - 糖尿病前期

# word cloud
# 1. 患者信息
# No. of Patient - 患者编号
# 解释：用于唯一标识每个患者的编号。
# Age - 年龄
# 解释：患者的年龄，是糖尿病风险的重要因素之一。
# Gender - 性别
# 解释：患者的性别，男性和女性在糖尿病的发病率和症状上可能有所不同。

# 2. 血糖水平
# Sugar Level Blood - 血糖水平
# 解释：血液中的葡萄糖浓度，是糖尿病诊断和管理的关键指标。
# HBA1C - 糖化血红蛋白
# 解释：反映过去2-3个月平均血糖水平的指标，用于评估糖尿病控制情况。

# 3. 肾功能
# Creatinine ratio (Cr) - 肌酐比率
# 解释：评估肾功能的指标，反映肾脏过滤废物的能力。
# Urea - 尿素
# 解释：血液中的尿素水平，也是评估肾功能的重要指标。

# 4. 体质指数（BMI）
# Body Mass Index (BMI) - 体质指数
# 解释：衡量体重与身高比例的指标，用于评估肥胖风险，肥胖是2型糖尿病的主要风险因素之一。

# 5. 血脂谱
# Cholesterol (Chol) - 胆固醇
# 解释：血液中的脂质，包括总胆固醇、LDL胆固醇和HDL胆固醇。
# Fasting lipid profile - 空腹血脂谱
# 解释：空腹状态下血液中脂质的分析，包括以下指标：
# Total - 总胆固醇
# 解释：血液中所有胆固醇的总和。
# LDL - 低密度脂蛋白胆固醇
# 解释：通常称为“坏胆固醇”，与心血管疾病风险相关。
# VLDL - 极低密度脂蛋白胆固醇
# 解释：携带甘油三酯的脂蛋白。
# Triglycerides (TG) - 甘油三酯
# 解释：血液中的脂肪，高水平与心血管疾病风险相关。
# HDL Cholesterol - 高密度脂蛋白胆固醇
# 解释：通常称为“好胆固醇”，有助于清除血液中的胆固醇。

# 6. 疾病分类
# Class - 疾病分类
# 解释：患者的糖尿病状态，可能为以下几种：
# Diabetic - 糖尿病患者
# 解释：确诊为糖尿病的个体。
# Non-Diabetic - 非糖尿病患者
# 解释：未确诊为糖尿病的个体。
# Predict-Diabetic - 糖尿病前期
# 解释：血糖水平高于正常但尚未达到糖尿病诊断标准的个体。