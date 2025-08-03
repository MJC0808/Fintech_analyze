# 필수 패키지 로드
import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb

# --- 데이터 로드 ---
print("--- 데이터 로드 ---")
google_path = 'https://drive.google.com/uc?id='
file_id = '1cA2bkyBdPvNFX8yiL-kyczqlfv8YvjLK'
output_name = 'train.csv'
print(f"{output_name} 다운로드 중...")
gdown.download(google_path + file_id, output_name, quiet=False)
print("다운로드 완료.")

train_df = pd.read_csv('train.csv')
print(f'원본 train 데이터 크기: {train_df.shape}')
print(train_df.head())

# --- 데이터 탐색 및 전처리 ---
print("\n--- 데이터 탐색 및 전처리 ---")

# 특정 컬럼만 실습 대상으로 제한 (예시: Time, V1~V9, Amount, Class)
# 실제 사용 시에는 Feature Engineering이나 Feature Selection 과정을 거쳐야 함
selected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'Amount', 'Class']
train_df = train_df[selected_columns]
print(f'선택된 {len(selected_columns)}개 컬럼으로 제한 후 train 데이터 크기: {train_df.shape}')
print(train_df.head())

# 결측치 확인
print("\n결측치 현황:")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0]) # 결측치가 있는 컬럼만 출력

# 각 컬럼 값 확인 및 시각화
# Time 컬럼
plt.figure(figsize=(10, 4))
train_df['Time'].plot(title='Time 컬럼 분포')
plt.xlabel('인덱스')
plt.ylabel('Time 값')
plt.show()
print("주석: Time 컬럼은 거래 발생 시각을 나타냅니다. 시간 경과에 따른 거래 패턴 변화를 볼 수 있으며, 사기 거래가 특정 시간대에 집중될 수도 있습니다.")

# Amount 컬럼
plt.figure(figsize=(10, 4))
train_df['Amount'].plot(title='Amount 컬럼 분포')
plt.xlabel('인덱스')
plt.ylabel('Amount 값')
plt.show()
print("주석: Amount 컬럼은 거래 금액을 나타냅니다. 대부분의 거래는 소액이지만, 이상치 또는 특정 금액대의 패턴이 있을 수 있습니다.")

# Class 컬럼 (이상 거래 여부)
print("\nClass 컬럼 값 분포:")
print(train_df['Class'].value_counts())
print("\nClass 컬럼 비율:")
print(train_df['Class'].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
train_df['Class'].value_counts().plot(kind='bar', title='Class 분포 (0:정상, 1:이상)')
plt.xticks(rotation=0)
plt.xlabel('Class')
plt.ylabel('거래 수')
plt.show()
print("주석: Class 0(정상)이 Class 1(이상)에 비해 압도적으로 많습니다. 이는 전형적인 불균형 데이터셋으로, 모델 학습 시 이 문제를 고려해야 합니다.")

# 기초 통계량 확인
print("\n--- 기초 통계량 ---")
print(train_df.describe())

# --- 학습 데이터와 검증 데이터 분리 ---
print("\n--- 학습 데이터와 검증 데이터 분리 ---")

X = train_df.drop('Class', axis=1)
y = train_df['Class']

# 불균형 데이터셋이므로 stratify를 사용하여 Class 비율 유지
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 참고: 원래 코드는 특정 건수로 분리했지만, 비율로 분리하는 것이 일반적이고 유연함 (예: test_size=0.2)
# 원래 문제의 280000:4807 분리 (test_size 약 0.0168)를 반영하려면 test_size를 직접 계산하여 넣어야 함
# test_size = 4807 / len(train_df)

print(f'X_train 크기: {X_train.shape}, y_train 크기: {y_train.shape}')
print(f'X_valid 크기: {X_valid.shape}, y_valid 크기: {y_valid.shape}')
print(f'y_train Class 비율:\n{y_train.value_counts(normalize=True)}')
print(f'y_valid Class 비율:\n{y_valid.value_counts(normalize=True)}')

# --- 단일 LightGBM 모델 학습 ---
print("\n--- 단일 LightGBM 모델 학습 ---")

lgbm_params = {
    'n_estimators': 1000, # 초기 n_estimators를 더 높여서 early stopping의 기회 제공
    'learning_rate': 0.05,
    'num_leaves': 31,
    'objective': 'binary', # 이진 분류
    'random_state': 42,
    'n_jobs': -1, # 모든 코어 사용
    'colsample_bytree': 0.8, # 컬럼 샘플링
    'subsample': 0.8, # 로우 샘플링
    'reg_alpha': 0.1, # L1 정규화
    'reg_lambda': 0.1, # L2 정규화
    'verbose': -1, # 학습 과정 메시지 비활성화
    'boosting_type': 'gbdt',
    # 'boost_from_average': False, # 불균형 데이터 학습의 경우 성능 개선 가능 (LightGBM 2.1.0 이상)
}

lgbm_model = lgb.LGBMClassifier(**lgbm_params)
print("LightGBM 모델 학습 시작...")
lgbm_model.fit(X_train, y_train,
               eval_set=[(X_valid, y_valid)],
               eval_metric='auc',
               callbacks=[lgb.early_stopping(100, verbose=False)]) # 100회 동안 성능 개선 없으면 조기 종료

# --- 단일 LightGBM 모델 예측 및 평가 ---
print("\n--- 단일 LightGBM 모델 예측 및 평가 ---")

y_pred_lgbm = lgbm_model.predict_proba(X_valid)[:, 1] # 클래스 1(이상 거래)에 대한 예측 확률
valid_df_lgbm = X_valid.copy()
valid_df_lgbm['Class'] = y_valid
valid_df_lgbm['pred_proba'] = y_pred_lgbm

lgbm_roc_auc = roc_auc_score(y_valid, y_pred_lgbm)
print(f'단일 LightGBM ROC AUC Score = {lgbm_roc_auc:.4f}')

# Confusion Matrix, Precision, Recall, F1-Score (임계값 0.5 기준)
y_pred_class_lgbm = (y_pred_lgbm > 0.5).astype(int)
print("\n단일 LightGBM 혼동 행렬:")
print(confusion_matrix(y_valid, y_pred_class_lgbm))
print(f"정확도 (Accuracy): {accuracy_score(y_valid, y_pred_class_lgbm):.4f}")
print(f"정밀도 (Precision): {precision_score(y_valid, y_pred_class_lgbm):.4f}")
print(f"재현율 (Recall): {recall_score(y_valid, y_pred_class_lgbm):.4f}")
print(f"F1-Score: {f1_score(y_valid, y_pred_class_lgbm):.4f}")

# --- 단일 LightGBM 변수 중요도 시각화 ---
print("\n--- 단일 LightGBM 변수 중요도 시각화 ---")
lgbm_feature_importance = pd.DataFrame({
    'feature': lgbm_model.feature_name_,
    'importance': lgbm_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=lgbm_feature_importance.head(len(lgbm_feature_importance))) # 모든 변수 출력
plt.title('단일 LightGBM 변수 중요도')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.tight_layout()
plt.show()

# --- 모델 저장 및 불러오기 (단일 LightGBM) ---
print("\n--- 단일 LightGBM 모델 저장 및 불러오기 ---")
save_object_lgbm = [lgbm_model, lgbm_params, valid_df_lgbm]
with open(file='lgbm_model.pickle', mode='wb') as f:
    pickle.dump(save_object_lgbm, f)
print("단일 LightGBM 모델 저장 완료: lgbm_model.pickle")

with open(file='lgbm_model.pickle', mode='rb') as f:
    loaded_lgbm_model, loaded_lgbm_params, loaded_valid_df_lgbm = pickle.load(f)
print("단일 LightGBM 모델 불러오기 완료.")

# 불러온 모델로 검증
loaded_y_pred_lgbm = loaded_lgbm_model.predict_proba(X_valid)[:, 1]
loaded_lgbm_roc_auc = roc_auc_score(y_valid, loaded_y_pred_lgbm)
print(f'불러온 단일 LightGBM ROC AUC Score = {loaded_lgbm_roc_auc:.4f}')

# --- 앙상블 모델 학습 (VotingClassifier) ---
print("\n--- 앙상블 모델 학습 (VotingClassifier) ---")

# Model 1: Logistic Regression
lr_params = {
    'solver': 'liblinear', # 작은 데이터셋에 적합, L1/L2 정규화 지원
    'random_state': 42,
    'class_weight': 'balanced' # 불균형 데이터셋에 유리
}
model_lr = LogisticRegression(**lr_params)
print("Logistic Regression 학습 시작...")
model_lr.fit(X_train, y_train)
print("Logistic Regression 학습 완료.")

# Model 2: Random Forest Classifier
rf_params = {
    'n_estimators': 100,
    'criterion': 'entropy',
    'max_depth': 10, # 과적합 방지를 위해 max_depth를 제한
    'random_state': 42,
    'class_weight': 'balanced' # 불균형 데이터셋에 유리
}
model_rf = RandomForestClassifier(**rf_params)
print("Random Forest Classifier 학습 시작...")
model_rf.fit(X_train, y_train)
print("Random Forest Classifier 학습 완료.")

# Model 3: LightGBM (위에서 학습한 lgbm_model 재사용 또는 새로 정의)
# 여기서는 새로운 파라미터로 다시 정의 (예시)
lgbm_ensemble_params = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'num_leaves': 30,
    'objective': 'binary',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    'boosting_type': 'gbdt',
    # 'is_unbalance': True # LightGBM의 불균형 데이터 처리 옵션 (objective가 'binary'일 때)
}
model_lgbm_ensemble = lgb.LGBMClassifier(**lgbm_ensemble_params)
print("Ensemble용 LightGBM 학습 시작...")
model_lgbm_ensemble.fit(X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        eval_metric='auc',
                        callbacks=[lgb.early_stopping(100, verbose=False)])
print("Ensemble용 LightGBM 학습 완료.")

# 앙상블 모델 정의 및 학습 (Soft Voting: 예측 확률을 평균)
final_model = VotingClassifier(estimators=[('lr', model_lr), ('rf', model_rf), ('lgbm', model_lgbm_ensemble)], voting='soft', n_jobs=-1)
print("\nVotingClassifier 앙상블 모델 학습 시작...")
final_model.fit(X_train, y_train)
print("VotingClassifier 앙상블 모델 학습 완료.")

# --- 앙상블 모델 예측 및 평가 ---
print("\n--- 앙상블 모델 예측 및 평가 ---")

y_pred_ensemble = final_model.predict_proba(X_valid)[:, 1]
valid_df_ensemble = X_valid.copy()
valid_df_ensemble['Class'] = y_valid
valid_df_ensemble['pred_proba'] = y_pred_ensemble

ensemble_roc_auc = roc_auc_score(y_valid, y_pred_ensemble)
print(f'앙상블 모델 ROC AUC Score = {ensemble_roc_auc:.4f}')

# Confusion Matrix, Precision, Recall, F1-Score (임계값 0.5 기준)
y_pred_class_ensemble = (y_pred_ensemble > 0.5).astype(int)
print("\n앙상블 모델 혼동 행렬:")
print(confusion_matrix(y_valid, y_pred_class_ensemble))
print(f"정확도 (Accuracy): {accuracy_score(y_valid, y_pred_class_ensemble):.4f}")
print(f"정밀도 (Precision): {precision_score(y_valid, y_pred_class_ensemble):.4f}")
print(f"재현율 (Recall): {recall_score(y_valid, y_pred_class_ensemble):.4f}")
print(f"F1-Score: {f1_score(y_valid, y_pred_class_ensemble):.4f}")

# --- 앙상블 모델 변수 중요도 시각화 (RandomForest 기준) ---
print("\n--- 앙상블 모델 변수 중요도 시각화 (RandomForest 기준) ---")
# 앙상블 모델은 직접적인 feature_importance를 제공하지 않으므로, 개별 모델 중 하나를 사용
# 여기서는 RandomForest 모델의 변수 중요도를 사용
rf_ensemble_feature_importance = pd.DataFrame({
    'feature': model_rf.feature_names_in_,
    'importance': model_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=rf_ensemble_feature_importance)
plt.title('앙상블 모델 (RandomForest 기준) 변수 중요도')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.tight_layout()
plt.show()

# --- 모델 저장 및 불러오기 (앙상블) ---
print("\n--- 앙상블 모델 저장 및 불러오기 ---")
save_object_ensemble = [final_model, y_valid, y_pred_ensemble] # 앙상블 모델, 실제값, 예측 확률 저장
with open(file='ensemble_model.pickle', mode='wb') as f:
    pickle.dump(save_object_ensemble, f)
print("앙상블 모델 저장 완료: ensemble_model.pickle")

with open(file='ensemble_model.pickle', mode='rb') as f:
    loaded_ensemble_model, loaded_y_valid, loaded_y_pred_ensemble = pickle.load(f)
print("앙상블 모델 불러오기 완료.")

# 불러온 모델로 검증
reloaded_ensemble_roc_auc = roc_auc_score(loaded_y_valid, loaded_y_pred_ensemble)
print(f'불러온 앙상블 모델 ROC AUC Score = {reloaded_ensemble_roc_auc:.4f}')

print("\n--- 프로그램 종료 ---")