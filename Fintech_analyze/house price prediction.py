# 필수 패키지 로드
import gdown
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import lightgbm as lgb
import optuna

# --- 데이터 로드 및 초기 준비 ---
print("--- 데이터 로드 및 초기 준비 ---")

# 파일 다운로드
google_path = 'https://drive.google.com/uc?id='
file_id = '1oL3KU8zMxI8AfANJSeI-jHCQHZeiJ9mW'
output_name = 'train.csv'
print(f"{output_name} 다운로드 중...")
gdown.download(google_path + file_id, output_name, quiet=False)
print("다운로드 완료.")

# 파일 불러오기
train_df = pd.read_csv('train.csv')
print(f'원본 train 데이터 크기: {train_df.shape}')
print(train_df.head())

# --- 데이터 탐색 및 전처리 (EDA & Preprocessing) ---
print("\n--- 데이터 탐색 및 전처리 ---")

# SalePrice 분포 시각화 (로그 변환 전후)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(train_df['SalePrice'], kde=True)
plt.title('SalePrice 분포 (원본)')
plt.subplot(1, 2, 2)
sns.histplot(np.log1p(train_df['SalePrice']), kde=True)
plt.title('SalePrice 분포 (로그 변환 후)')
plt.tight_layout()
plt.show()
print("주석: SalePrice는 오른쪽으로 치우쳐진 분포를 보입니다. 로그 변환을 통해 정규 분포에 가깝게 만들어 모델 성능을 향상시킬 수 있습니다.")

# SalePrice 로그 변환 (타겟 변수)
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])

# Id 컬럼 제거 (예측에 불필요)
if 'Id' in train_df.columns:
    train_df = train_df.drop('Id', axis=1)
    print("Id 컬럼 제거.")

# 결측치 확인
missing_counts = train_df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
print("\n결측치 현황:")
print(missing_counts)

# 결측치 비율이 높은 컬럼 제거 (예: 50% 이상)
# 여기서는 예시로 80%를 넘는 경우 제거. 실제는 도메인 지식과 함께 판단.
cols_to_drop_high_na = missing_counts[missing_counts / len(train_df) > 0.8].index.tolist()
if cols_to_drop_high_na:
    train_df = train_df.drop(columns=cols_to_drop_high_na)
    print(f"결측치 비율이 높은 컬럼 제거: {cols_to_drop_high_na}")

# 수치형/범주형 컬럼 분리
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_df.select_dtypes(include='object').columns.tolist()

# SalePrice는 타겟이므로 특징에서 제외
if 'SalePrice' in numerical_cols:
    numerical_cols.remove('SalePrice')

print(f"\n수치형 컬럼 ({len(numerical_cols)}개): {numerical_cols[:5]}...") # 앞 5개만 출력
print(f"범주형 컬럼 ({len(categorical_cols)}개): {categorical_cols[:5]}...") # 앞 5개만 출력

# 범주형 컬럼 레이블 인코딩 (LightGBM은 정수형 범주형 변수 처리 가능)
# 결측치는 SimpleImputer를 통해 처리되므로, LabelEncoder 적용 전 NaN 처리.
for col in categorical_cols:
    # NaN이 아닌 값에 대해서만 인코딩 수행
    unique_non_nan_values = train_df[col].dropna().unique()
    if len(unique_non_nan_values) > 0: # 유효한 값이 있는 경우에만 인코딩
        le = LabelEncoder()
        train_df.loc[train_df[col].notna(), col] = le.fit_transform(train_df.loc[train_df[col].notna(), col])
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce') # 인코딩 후 숫자 타입으로 변환

print("\n--- 인코딩 및 초기 처리 후 데이터 정보 ---")
train_df.info()

# --- 특징 선택 (Feature Selection) ---
print("\n--- 특징 선택 ---")
# 여기서는 모든 컬럼을 활용하고, 모델의 feature_importance로 중요도 확인.
# 더 정교한 특징 선택은 모델 중요도 분석 후 중요도 낮은 컬럼 제거, 상관관계 분석 후 다중공선성 제거 등을 통해 수행 가능.
features = [col for col in train_df.columns if col != 'SalePrice']
target = 'SalePrice'

print(f"학습에 사용될 특징 컬럼 수: {len(features)}")
print(f"타겟 컬럼: {target}")

X = train_df[features]
y = train_df[target]

# --- 모델 파이프라인 구성 ---
print("\n--- 모델 파이프라인 구성 ---")

# 수치형 컬럼의 결측치는 중앙값으로 대체하고 스케일링
# 범주형 컬럼의 결측치는 최빈값으로 대체 (LabelEncoded 되어 숫자가 되었으므로)
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# 각 컬럼 타입에 맞는 트랜스포머 적용
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ],
    remainder='passthrough' # 나머지 컬럼은 그대로 통과
)

# LightGBM 모델과 파이프라인 연결
# `LGBMRegressor`는 `n_jobs=-1`로 설정하여 모든 코어를 사용합니다.
# `objective='rmse'`는 LightGBM 내부적으로 'regression_l1' (MAE) 대신 'regression_l2' (MSE)의 RMSE 버전을 사용하도록 지시합니다.
# `random_state`로 재현성을 확보합니다.
# `verbose=-1`로 학습 과정 메시지를 최소화합니다.
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', lgb.LGBMRegressor(random_state=42, verbose=-1))])

print("모델 파이프라인 구성 완료.")

# --- 교차 검증 (K-Fold Cross-Validation) ---
print("\n--- 교차 검증 (K-Fold Cross-Validation) ---")

# K-Fold 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5-Fold 사용, 셔플

oof_preds = np.zeros(len(X)) # Out-Of-Fold 예측 저장
models = [] # 각 폴드에서 학습된 모델 저장
rmse_scores = []
mae_scores = []
r2_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"\n--- 폴드 {fold+1}/{kf.n_splits} 학습 시작 ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # 파이프라인 학습 (전처리 + 모델)
    # n_estimators, learning_rate 등은 Optuna에서 튜닝할 것이므로 초기값 사용
    current_params = {
        'regressor__n_estimators': 1000,
        'regressor__learning_rate': 0.05,
        'regressor__num_leaves': 31,
        'regressor__objective': 'rmse',
        'regressor__random_state': 42,
        'regressor__verbose': -1
    }
    model_pipeline.set_params(**current_params)
    model_pipeline.fit(X_train, y_train,
                       regressor__eval_set=[(preprocessor.fit_transform(X_val), y_val)], # 파이프라인 내에서 전처리 적용
                       regressor__callbacks=[lgb.early_stopping(100, verbose=False)])

    # 예측 및 평가
    fold_preds = model_pipeline.predict(X_val)
    fold_rmse = mean_squared_error(y_val, fold_preds, squared=False)
    fold_mae = mean_absolute_error(y_val, fold_preds)
    fold_r2 = r2_score(y_val, fold_preds)

    oof_preds[val_index] = fold_preds
    models.append(model_pipeline)
    rmse_scores.append(fold_rmse)
    mae_scores.append(fold_mae)
    r2_scores.append(fold_r2)

    print(f"폴드 {fold+1} RMSE: {fold_rmse:.4f}, MAE: {fold_mae:.4f}, R2: {fold_r2:.4f}")

print("\n--- 교차 검증 결과 요약 ---")
print(f"평균 RMSE: {np.mean(rmse_scores):.4f} (Std: {np.std(rmse_scores):.4f})")
print(f"평균 MAE: {np.mean(mae_scores):.4f} (Std: {np.std(mae_scores):.4f})")
print(f"평균 R2: {np.mean(r2_scores):.4f} (Std: {np.std(r2_scores):.4f})")

# OOF 예측을 사용하여 전체 데이터셋에 대한 최종 RMSE 계산
final_oof_rmse = mean_squared_error(y, oof_preds, squared=False)
print(f"전체 OOF RMSE: {final_oof_rmse:.4f}")

# 변수 중요도 (마지막 폴드 모델 기준)
# LightGBM 모델 객체를 파이프라인에서 추출해야 합니다.
final_lgbm_model = models[-1].named_steps['regressor']
# 전처리 후 컬럼명 추출이 복잡하므로, 인코딩된 특징 컬럼 기준으로 중요도 확인
feature_importance_df = pd.DataFrame({
    'feature': final_lgbm_model.feature_name_,
    'importance': final_lgbm_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n변수 중요도 (마지막 폴드 모델 기준):")
print(feature_importance_df.head(10)) # 상위 10개만 출력

# 변수 중요도 시각화
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
plt.title('변수 중요도 (상위 15개)')
plt.xlabel('중요도')
plt.ylabel('변수')
plt.tight_layout()
plt.show()

# --- 모델 저장 및 불러오기 (가장 좋은 폴드 모델 또는 마지막 폴드 모델 저장) ---
print("\n--- 모델 저장 및 불러오기 ---")
# 여기서는 마지막 폴드 모델을 저장합니다.
save_file_name = 'best_house_price_model.pickle'
with open(file=save_file_name, mode='wb') as f:
    pickle.dump(models[-1], f) # 파이프라인 전체 저장
print(f"최종 모델 (파이프라인 포함)이 '{save_file_name}'에 저장되었습니다.")

# 저장된 모델 불러오기
with open(file=save_file_name, mode='rb') as f:
    loaded_pipeline = pickle.load(f)
print(f"모델이 '{save_file_name}'에서 로드되었습니다.")

# 로드된 모델로 예측 검증 (X의 일부 사용)
sample_X_val = X.iloc[val_index].head() # 마지막 검증 세트의 일부
sample_y_val = y.iloc[val_index].head()
loaded_preds = loaded_pipeline.predict(sample_X_val)
print("\n로드된 모델을 사용한 샘플 예측:")
print(f"실제 SalePrice (로그): {sample_y_val.values}")
print(f"예측 SalePrice (로그): {loaded_preds}")
# 원래 스케일로 변환하여 비교
print(f"실제 SalePrice (원래 스케일): {np.expm1(sample_y_val.values)}")
print(f"예측 SalePrice (원래 스케일): {np.expm1(loaded_preds)}")


# --- 하이퍼파라미터 튜닝 (Optuna) ---
print("\n--- 하이퍼파라미터 튜닝 (Optuna) ---")

# Optuna 목적 함수 정의 (교차 검증 포함)
def objective(trial, X, y):
    params = {
        'regressor__n_estimators': trial.suggest_int('n_estimators', 1000, 7000, step=100),
        'regressor__learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'regressor__num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'regressor__max_depth': trial.suggest_int('max_depth', 5, 15),
        'regressor__min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
        'regressor__subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'regressor__colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'regressor__reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'regressor__reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'regressor__objective': 'rmse',
        'regressor__random_state': 42,
        'regressor__verbose': -1,
        'regressor__n_jobs': -1
    }

    # 파이프라인의 regressor 단계에 파라미터 설정
    model_pipeline.set_params(**params)

    # 교차 검증을 통해 RMSE 계산
    kf_tuning = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmse_list = []

    for train_idx, val_idx in kf_tuning.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model_pipeline.fit(X_train_fold, y_train_fold,
                           regressor__eval_set=[(preprocessor.fit_transform(X_val_fold), y_val_fold)],
                           regressor__callbacks=[lgb.early_stopping(100, verbose=False)],
                           regressor__eval_metric='rmse') # 명시적으로 eval_metric 설정

        fold_preds = model_pipeline.predict(X_val_fold)
        fold_rmse_list.append(mean_squared_error(y_val_fold, fold_preds, squared=False))

    return np.mean(fold_rmse_list) # 평균 RMSE 반환

# Optuna 스터디 객체 생성
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())

# 튜닝 실행
# n_trials를 늘려 더 많은 하이퍼파라미터 조합 탐색 가능
# timeout을 설정하여 튜닝 시간 제한 가능
print(f"\nOptuna 하이퍼파라미터 튜닝 시작 (최대 50회 시도 또는 3600초 타임아웃)...")
study.optimize(lambda trial: objective(trial, X, y), n_trials=50, timeout=3600, show_progress_bar=True)

print(f'\nOptuna가 찾은 최적 파라미터: {study.best_trial.params}')
print(f'튜닝 중 최적 교차 검증 RMSE: {study.best_value:.4f}')

# --- 최적 파라미터로 최종 모델 학습 ---
print("\n--- 최적 파라미터로 최종 모델 학습 ---")

# Optuna가 찾은 최적 파라미터로 최종 파이프라인 설정
best_params_flat = {'regressor__' + k: v for k, v in study.best_trial.params.items()}
model_pipeline.set_params(**best_params_flat)

# 전체 학습 데이터로 최종 모델 재학습
model_pipeline.fit(X, y)

# --- Optuna 시각화 ---
print("\n--- Optuna 시각화 ---")

try:
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.show()
    print("최적화 히스토리 플롯 생성됨.")
except Exception as e:
    print(f"최적화 히스토리 플롯 생성 실패: {e}")

try:
    fig_slice = optuna.visualization.plot_slice(study)
    fig_slice.show()
    print("슬라이스 플롯 생성됨.")
except Exception as e:
    print(f"슬라이스 플롯 생성 실패: {e}")

try:
    # 튜닝된 파라미터 중 두 개를 선택하여 등고선 플롯 생성
    if len(study.best_trial.params) >= 2:
        contour_params = list(study.best_trial.params.keys())[:2]
        fig_contour = optuna.visualization.plot_contour(study, contour_params)
        fig_contour.show()
        print(f"'{contour_params[0]}'와 '{contour_params[1]}'에 대한 등고선 플롯 생성됨.")
    else:
        print("등고선 플롯을 생성할 파라미터가 충분하지 않습니다.")
except Exception as e:
    print(f"등고선 플롯 생성 실패: {e}")

print("\n--- 프로그램 종료 ---")