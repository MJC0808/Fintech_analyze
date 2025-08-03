# --- 1. 필수 패키지 임포트 ---
import gdown          # Google Drive에서 파일 다운로드를 위한 패키지
import pandas as pd   # 데이터 조작 및 분석을 위한 패키지
import numpy as np    # 수치 계산을 위한 패키지
import matplotlib.pyplot as plt # 데이터 시각화를 위한 패키지
import seaborn as sns # Matplotlib 기반의 고급 데이터 시각화를 위한 패키지
import warnings       # 경고 메시지 관리를 위한 패키지
import pickle         # Python 객체를 직렬화/역직렬화 (모델 저장/불러오기) 하기 위한 패키지

from sklearn.metrics import precision_score, recall_score, f1_score # 모델 평가 지표 (정밀도, 재현율, F1 점수)
from sklearn.preprocessing import LabelEncoder # 범주형 데이터를 수치형으로 변환하기 위한 패키지
import lightgbm as lgb # LightGBM 머신러닝 모델

# 경고 메시지 출력 억제 (깔끔한 출력을 위함)
warnings.filterwarnings('ignore')

# --- 2. 평가 지표 예제 ---
print("--- 평가 지표 예제 ---")
y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] # 실제 값
y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0] # 예측 값

# 예시를 통한 수동 계산:
# TP (True Positives, 실제 1, 예측 1): 3개 (인덱스 5, 6, 7)
# FP (False Positives, 실제 0, 예측 1): 1개 (인덱스 4)
# FN (False Negatives, 실제 1, 예측 0): 2개 (인덱스 8, 9)

print(f'정밀도 (Precision): {precision_score(y_true, y_pred):.2f} (수동 계산: {3/(3+1):.2f})')
print(f'재현율 (Recall): {recall_score(y_true, y_pred):.2f} (수동 계산: {3/(3+2):.2f})')
print(f'f1 score: {f1_score(y_true, y_pred):.2f} (수동 계산: {2 * (0.75 * 0.6)/(0.75 + 0.6):.2f})')
print("\n")

# --- 3. 데이터 로딩 및 초기 전처리 ---
print("--- 데이터 로딩 및 초기 전처리 ---")

# 파일 다운로드
google_path = 'https://drive.google.com/uc?id='
file_id = '1j0qGYojlW9cgRghZZUv4gWLBFjR8W8nG'
output_name = 'train.csv'
print(f"{output_name} 파일 다운로드 중...")
gdown.download(google_path + file_id, output_name, quiet=False)
print("다운로드 완료.\n")

# 파일 로드
train = pd.read_csv(output_name)

# 데이터 미리보기: 임의의 5개 컬럼 출력
# 'ID', 'Age', 'Occupation', 'Annual_Income', 'Credit_Score' 컬럼을 선택하여 출력
print("원본 데이터 미리보기 (선택된 컬럼의 상위 5행):")
print(train[['ID', 'Age', 'Occupation', 'Annual_Income', 'Credit_Score']].head())
print("\n")

# 결측치 처리: 모든 NaN 값을 0으로 채우기 (요청에 따라)
print("결측치 채우기 전 결측치 확인:")
print(train.isnull().sum()[train.isnull().sum() > 0]) # 결측치가 있는 컬럼만 표시
train_filled = train.fillna(0)
print("\n결측치가 0으로 채워졌습니다.")

# 'Name' 컬럼은 일반적으로 특징으로 유용하지 않으므로 삭제
train_drop = train_filled.drop(columns=['Name'])

# train_drop 데이터 미리보기 (상위 5행)
print("NaN 값을 채우고 'Name' 컬럼을 삭제한 후 데이터 미리보기 (상위 5행):")
print(train_drop.head())
print("\n")

# 중복 행 확인
print(f"중복 행 개수: {train_drop.duplicated().sum()}")
print("\n")

# 초기 데이터 정보 확인
print("초기 전처리 후 데이터 구조:")
train_drop.info()
print("\n")

# 기초 통계량 확인
print("수치형 컬럼의 기초 통계량:")
print(train_drop.describe().T)
print("\n")

# --- 4. 탐색적 데이터 분석 (EDA) 및 특징 공학 ---
print("--- EDA 및 특징 공학 ---")

# 직업별 데이터 수 확인
print("직업별 값 개수 및 고유 직업 수:")
print(train_drop['Occupation'].value_counts().head())
print(f"고유 직업 수: {train_drop.Occupation.nunique()}")
print("\n")

# 신용 등급별 월별 총 균등할부(EMI) 평균 금액 확인
print("Credit_Score별 Total_EMI_per_month 평균:")
print(train_drop.groupby('Credit_Score')['Total_EMI_per_month'].mean().reset_index())
print("\n")

# 보유 카드 수별 건수
print("보유 카드 수 값 개수 (정규화 전 상위 10개):")
print(train_drop['Num_Credit_Card'].value_counts().head(10))

# Num_Credit_Card 정규화 (Z-점수 표준화)
# 정규화(Normalization)는 데이터의 스케일을 조정하여 특정 피처의 값이 다른 피처의 값에 비해 너무 크거나 작아서 모델 학습에 미치는 영향이 불균형해지는 것을 방지하는 과정입니다.
# 특히 선형 모델이나 경사 하강법 기반의 모델에서 효과적이며, 데이터를 특정 범위(예: 0~1)로 맞추거나 평균이 0이고 표준편차가 1인 표준 정규 분포 형태로 변환합니다.
# 이 코드에서는 Z-score 정규화(Standardization)를 사용하여 데이터를 평균 0, 표준편차 1로 변환하고 있습니다.
train_drop['Num_Credit_Card'] = (train_drop['Num_Credit_Card'] - train_drop['Num_Credit_Card'].mean()) / train_drop['Num_Credit_Card'].std()
print("\n보유 카드 수 값 개수 (정규화 후 상위 10개):")
print(train_drop['Num_Credit_Card'].value_counts().head(10))
print("\n")

# 특수 문자가 포함된 수치형 컬럼 정리 및 타입 변환
print("수치형 컬럼 정리 중 ('_' 제거 및 타입 변환):")
for col in ['Annual_Income', 'Age', 'Num_of_Loan', 'Outstanding_Debt']:
    if train_drop[col].dtype == 'object': # 데이터 타입이 문자열/객체인 경우에만 처리
        train_drop[col] = train_drop[col].apply(lambda x: str(x).replace('_', ''))
        if col == 'Annual_Income' or col == 'Outstanding_Debt':
            train_drop[col] = train_drop[col].astype(float)
        else: # Age, Num_of_Loan
            train_drop[col] = train_drop[col].astype(int)
print("수치형 컬럼 정리가 완료되었습니다.")
print("수치형 컬럼 정리 후 기초 통계량:")
print(train_drop.describe().T)
print("\n")

# 수치형 변수의 히스토그램 시각화
print("수치형 특징의 히스토그램을 표시합니다 (bins=50). 플롯을 닫아야 다음 단계로 진행합니다.")
# bins는 히스토그램의 막대(bin) 개수를 의미합니다.
# bins 값이 작을수록 막대 폭이 넓어져 데이터 분포를 더 넓은 구간으로 뭉뚱그려 보여주고,
# bins 값이 클수록 막대 폭이 좁아져 데이터 분포를 더 세분화된 구간으로 자세히 보여줍니다.
# 적절한 bins 값은 데이터의 특성과 분석 목적에 따라 달라질 수 있습니다.
train_drop.hist(bins=50, figsize=(15, 7))
plt.tight_layout()
plt.suptitle('수치형 특징의 히스토그램', y=1.02) # 전체 그림에 제목 추가
plt.show()

# 예측 대상 데이터인 Credit_Score의 분포를 살펴보기 위한 countplot 시각화
print("Credit_Score의 countplot을 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
plt.figure(figsize=(6, 3))
sns.countplot(x=train_drop['Credit_Score'])
plt.title('Credit_Score 분포')
plt.xlabel('신용 점수')
plt.ylabel('개수')
plt.show()

# 다른 범주형 변수(예: 'Occupation', 'Month')에 대한 countplot 시각화
print("Occupation의 countplot을 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
plt.figure(figsize=(12, 6))
# 범주가 많을 경우 가독성을 위해 값의 빈도 순으로 정렬
sns.countplot(y=train_drop['Occupation'], order=train_drop['Occupation'].value_counts().index)
plt.title('직업 분포')
plt.xlabel('개수')
plt.ylabel('직업')
plt.show()

print("Month의 countplot을 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
plt.figure(figsize=(8, 4))
sns.countplot(x=train_drop['Month'], order=train_drop['Month'].value_counts().index)
plt.title('월 분포')
plt.xlabel('월')
plt.ylabel('개수')
plt.show()
print("\n")

# LabelEncoder를 사용하여 범주형 데이터를 수치형으로 변환
print("LabelEncoder를 사용하여 범주형 컬럼을 수치형으로 변환 중:")
le = LabelEncoder()
# 인코딩이 필요한 범주형 특징 컬럼 정의
# 'Credit_Score'는 타겟 변수이며, 나머지는 특징 변수입니다.
categorical_features = ['ID', 'Customer_ID', 'Month', 'SSN', 'Occupation',
                        'Changed_Credit_Limit', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

for col in categorical_features:
    if col in train_drop.columns:
        train_drop[col] = le.fit_transform(train_drop[col].astype(str)) # 일관된 인코딩을 위해 문자열 타입으로 변환

# 'Credit_Score' (타겟 변수)도 LabelEncoder를 사용하여 변환
# 참고: 원본 코드는 Credit_Score를 직접 변환하며 원래 범주를 보존하지 않습니다.
# LabelEncoder는 알파벳 순으로 Good: 0, Standard: 1, Bad: 2 등으로 매핑될 수 있습니다.
# 필요한 경우 원래 매핑을 확인해야 합니다.
if 'Credit_Score' in train_drop.columns:
    train_drop['Credit_Score'] = le.fit_transform(train_drop['Credit_Score'])

print("레이블 인코딩 후 데이터 구조:")
train_drop.info()
print("\n")

# 모든 전처리 후 상위 5행 데이터 출력
print("최종 전처리된 데이터 미리보기 (상위 5행):")
print(train_drop.head())
print("\n")

# --- 5. 모델 학습 및 평가 ---
print("--- 모델 학습 및 평가 ---")

# 데이터를 학습 및 검증 세트로 분리
# 학습 데이터와 검증 데이터를 분리하는 건수는 정해져 있지 않습니다. 다양한 크기로 분리하여 학습과 검증해 보시기 바랍니다.
# 예를 들어, sklearn.model_selection의 train_test_split을 사용하여 무작위로 분리할 수도 있습니다.
# 여기서는 원본 코드의 순차적 분리 방식을 유지합니다.
train_data = train_drop[:-100] # 마지막 100행을 제외한 모든 행을 학습 데이터로 사용
valid_data = train_drop[-100:] # 마지막 100행을 검증 데이터로 사용

print(f"학습 데이터 형태: {train_data.shape}")
print(f"검증 데이터 형태: {valid_data.shape}")

# 학습 데이터 준비 (특징 및 타겟)
x_train = train_data.drop(columns=['Credit_Score']) # 특징 (타겟 컬럼 제외)
y_train = train_data['Credit_Score'] # 타겟 ('Credit_Score' 컬럼)

print(f"x_train 형태: {x_train.shape}, y_train 형태: {y_train.shape}")

# 검증 데이터 준비 (특징 및 타겟)
x_valid = valid_data.drop(columns=['Credit_Score']) # 검증용 특징
y_valid = valid_data['Credit_Score'] # 검증용 타겟

print(f"x_valid 형태: {x_valid.shape}, y_valid 형태: {y_valid.shape}")
print("\n")

# LightGBM을 위한 사용자 정의 평가 지표 (마이크로 F1-스코어)
def lgbm_microf1(truth, predictions):
    # 예측값은 각 클래스에 대한 확률이므로, (샘플 수, 클래스 수) 형태로 재구성합니다.
    # argmax를 사용하여 가장 높은 확률을 가진 클래스를 선택합니다.
    pred_labels = predictions.reshape(truth.shape[0], -1).argmax(axis=1)
    f1 = f1_score(truth, pred_labels, average='micro') # 마이크로 평균 F1-스코어 계산
    return ('lgbm_microf1', f1, True) # (지표 이름, 지표 값, 높을수록 좋음)

# LightGBM 모델 파라미터 정의
# 학습 파라메터는 정해져 있지 않습니다. 다양한 크기로 변경하여 학습한 후 검증결과를 확인해 보세요.
params = {
    'n_estimators': 500,     # 부스팅 라운드 수 (트리 개수)
    'learning_rate': 0.01,   # 과적합을 방지하기 위한 학습률 (각 트리의 기여도를 축소)
    'num_leaves': 30,        # 한 트리에 가질 수 있는 최대 잎 노드 수
    'objective': 'multiclass', # 다중 클래스 분류를 위한 목적 함수
    'metric': 'multi_logloss', # 학습 중 평가 지표
    'num_class': 3,          # 타겟 클래스 수 (Credit_Score: Bad, Standard, Good)
    'random_state': 42,      # 결과 재현성을 위한 시드 값
    'verbosity': -1,         # 학습 중 상세 출력 억제
    'n_jobs': -1             # 사용 가능한 모든 CPU 코어 사용
}

# LightGBM 분류기 초기화 및 학습
print("LightGBM 모델 학습 중...")
model = lgb.LGBMClassifier(**params)
model.fit(x_train, y_train,
          eval_set=[(x_valid, y_valid)], # 검증 세트 설정
          eval_metric=lgbm_microf1,      # 사용자 정의 평가 지표 사용
          callbacks=[lgb.early_stopping(10, verbose=False)]) # 검증 지표가 10라운드 동안 개선되지 않으면 조기 종료
print("모델 학습 완료.\n")

# 검증 세트에 대한 예측 수행
y_pred = model.predict(x_valid)
valid_data['pred'] = y_pred # 예측 결과를 검증 데이터 DataFrame에 추가

print("예측 결과가 추가된 검증 데이터 (상위 5행):")
print(valid_data.head())
print("\n")

# 모델 성능 평가 (F1-스코어)
score = f1_score(y_valid, y_pred, average='micro')
print(f'검증 세트에서의 모델 F1-스코어: {score:.4f}\n')

# 특징 중요도 확인
val_imp = pd.DataFrame(model.feature_importances_, index=model.feature_name_, columns=['imp'])
print("상위 10개 특징 중요도:")
print(val_imp.sort_values(by='imp', ascending=False).head(10))

# 특징 중요도 시각화
print("\n특징 중요도 플롯을 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
val_imp_sorted = val_imp.sort_values(by='imp', ascending=True) # 가로 막대 그래프를 위해 오름차순 정렬
plt.figure(figsize=(10, len(val_imp_sorted) * 0.4)) # 동적인 그림 높이 설정
val_imp_sorted['imp'].plot(kind='barh') # 가독성을 위해 가로 막대 그래프 사용
plt.title('특징 중요도')
plt.xlabel('중요도')
plt.ylabel('특징')
plt.tight_layout()
plt.show()

# --- 6. 모델 저장 및 로딩 ---
print("--- 모델 저장 및 로딩 ---")

# 저장할 객체 정의 (모델, 파라미터, 검증 데이터)
save_object = [model, params, valid_data]

# 객체 저장
model_filepath = 'my_credit_score_model.pickle'
with open(file=model_filepath, mode='wb') as f:
    pickle.dump(save_object, f)
print(f"모델 및 관련 객체가 '{model_filepath}'에 저장되었습니다.\n")

# 저장된 객체 불러오기
with open(file=model_filepath, mode='rb') as f:
    load_object = pickle.load(f)
print(f"모델 및 관련 객체가 '{model_filepath}'에서 로드되었습니다.\n")

# 로드된 객체 분리
loaded_model = load_object[0]
loaded_params = load_object[1]
loaded_valid_data = load_object[2]

# 로드된 모델을 사용하여 예측 수행
# 검증 데이터의 구조가 변경될 수 있으므로, x_valid는 loaded_valid_data에서 파생된 것을 사용하는 것이 좋습니다.
# 일관된 평가를 위해 초기 예측에 사용된 원본 x_valid를 사용합니다.
loaded_valid_data['pred_loaded'] = loaded_model.predict(x_valid)

print("로드된 모델의 예측 결과가 추가된 검증 데이터 (상위 5행):")
print(loaded_valid_data.head())
print("\n")

# 로드된 모델 평가
score_loaded = f1_score(loaded_valid_data['Credit_Score'], loaded_valid_data['pred_loaded'], average='micro')
print(f'로드된 모델의 F1-스코어: {score_loaded:.4f}\n')

print("프로그램 실행 완료.")