# --- 1. 필수 패키지 임포트 ---
import gdown          # Google Drive에서 파일 다운로드를 위한 패키지
import pandas as pd   # 데이터 조작 및 분석을 위한 패키지
import numpy as np    # 수치 계산을 위한 패키지
import matplotlib.pyplot as plt # 데이터 시각화를 위한 패키지
import seaborn as sns # Matplotlib 기반의 고급 데이터 시각화를 위한 패키지
import warnings       # 경고 메시지 관리를 위한 패키지
import pickle         # Python 객체를 직렬화/역직렬화 (모델 저장/불러오기) 하기 위한 패키지
import networkx as nx # 그래프 시각화를 위한 패키지 (협업 필터링 시각화에 사용)

from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도 계산
from sklearn.metrics import precision_score, recall_score, f1_score # 모델 평가 지표
from sklearn.preprocessing import LabelEncoder # 범주형 데이터를 수치형으로 변환

import lightgbm as lgb # LightGBM 머신러닝 모델

# 경고 메시지 출력 억제 (깔끔한 출력을 위함)
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Colab 외 환경 또는 초기 설정 셀 실행 후)
try:
    import matplotlib.font_manager as fm
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' # NanumGothic 폰트 경로 (시스템에 따라 다를 수 있음)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
except:
    print("한글 폰트 설정 중 오류 발생. 기본 폰트로 시각화됩니다.")


# --- 2. 평가 지표 예제 (개념 이해를 위한 코드) ---
def demonstrate_metrics():
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
def load_and_preprocess_data():
    print("--- 데이터 로딩 및 초기 전처리 ---")

    # 파일 다운로드: 상품 데이터
    google_path = 'https://drive.google.com/uc?id='
    product_file_id = '10s__zTv--ecAWxaW4Qfr4gk4vKJtksMG'
    product_output_name = 'train_product.csv'
    print(f"{product_output_name} 파일 다운로드 중...")
    gdown.download(google_path + product_file_id, product_output_name, quiet=False)

    # 파일 다운로드: 사용자 액션 데이터
    action_file_id = '1Cg4jXj3YcsoZk5_WZ7WnaHyaJitx9wWD'
    action_output_name = 'train_actions.csv'
    print(f"{action_output_name} 파일 다운로드 중...")
    gdown.download(google_path + action_file_id, action_output_name, quiet=False)
    print("다운로드 완료.\n")

    # 다운로드 파일 불러오기
    train_product = pd.read_csv(product_output_name, encoding='utf-8')
    # train_actions는 euc-kr 인코딩으로 불러옴
    train_actions = pd.read_csv(action_output_name, encoding='euc-kr')

    # 데이터 미리보기 및 결측치/중복 확인
    print("상품 데이터 형태:", train_product.shape)
    print("상품 데이터 미리보기:\n", train_product.head())
    print("상품 데이터 결측치:\n", train_product.isnull().sum())
    print("상품 데이터 중복:\n", train_product.duplicated().sum())
    print("\n")

    print("사용자 액션 데이터 형태:", train_actions.shape)
    print("사용자 액션 데이터 미리보기:\n", train_actions.head())
    print("사용자 액션 데이터 결측치:\n", train_actions.isnull().sum())
    print("사용자 액션 데이터 중복:\n", train_actions.duplicated().sum())
    print("\n")

    # 상품구분 Top4 확인 및 시각화
    top_product_type = train_product['상품구분'].value_counts().head(4)
    print("상품구분 Top 4:\n", top_product_type)

    # 파이 차트로 상품구분의 최다 개수 4개를 그려본다.
    plt.figure(figsize=(4, 4))
    plt.pie(top_product_type.values, labels=top_product_type.index, autopct='%1.1f%%', startangle=90)
    plt.title('Top 4 상품구분 분포')
    plt.show()
    # 시각화 결과에 대한 생각:
    # 이 파이 차트는 금융 상품 중 어떤 유형이 가장 많은 비중을 차지하는지 직관적으로 보여줍니다.
    # 예를 들어, '상품A'가 전체 상품 중 절반 이상을 차지한다면, 해당 상품이 주력 상품이거나 고객의 관심이 높다는 것을 알 수 있습니다.
    # 이는 향후 추천 시스템 개발 시 특정 상품군에 더 많은 리소스를 할당하거나,
    # 비중이 낮은 상품군에 대한 추천 전략을 고민하는 데 활용될 수 있습니다.
    print("\n")

    # 데이터 병합
    train_actions_with_name = train_actions.merge(train_product, on='상품ID', how='left')
    print("상품 정보가 병합된 액션 데이터 형태:", train_actions_with_name.shape)
    print("병합된 데이터 미리보기:\n", train_actions_with_name.head())
    print("\n")

    return train_product, train_actions, train_actions_with_name

# --- 4. 협업 필터링 추천 시스템 ---
def collaborative_filtering_recommender(train_actions_with_name):
    print("--- 협업 필터링 추천 시스템 ---")

    # 상품구분별 추천건수 확인
    print("상품구분별 추천건수:")
    product_type_counts = train_actions_with_name.groupby('상품구분')['추천점수'].count().reset_index()
    product_type_counts.rename(columns={'추천점수': '추천건수'}, inplace=True)
    print(product_type_counts)
    print("\n")

    # 상품별 추천건수 확인
    num_action_df = train_actions_with_name.groupby('상품ID').count()['추천점수'].reset_index()
    num_action_df.rename(columns={'추천점수':'추천건수'}, inplace=True)
    print("상품별 추천건수:\n", num_action_df)

    # 상품별 평균 추천점수 확인
    avg_action_df = train_actions_with_name.groupby('상품ID').mean()['추천점수'].reset_index()
    avg_action_df.rename(columns={'추천점수':'추천 점수평균'}, inplace=True)
    print("상품별 평균 추천점수:\n", avg_action_df)

    # 상품별 추천건수와 평균 추천점수 병합
    popular_df = num_action_df.merge(avg_action_df, on='상품ID')
    print("상품별 추천건수 및 평균 추천점수 병합:\n", popular_df)
    print("\n")

    # 추천점수 기반 시각화
    plt.figure(figsize=(10, 4))
    scatter = plt.scatter(popular_df['추천 점수평균'], popular_df.index, c=popular_df.index, cmap='cool', s=50)
    plt.xlabel('평균 추천점수')
    plt.ylabel('상품ID') # 인덱스 대신 상품ID를 명확히 표시
    plt.colorbar(label='상품 인덱스', orientation='vertical')
    plt.title('상품별 평균 추천점수 분포')
    plt.show()
    print("\n")

    # 다양한 횟수의 추천을 한 사용자 선택 분석
    # 사용자의 추천 활동이 1회 이상인 사용자만 필터링
    # 예를 들어, `x = train_actions_with_name.groupby('사용자ID').count()['추천점수'] > 2` 와 같이 변경하여
    # 2회 이상 추천한 사용자만 선택하여 분석 결과를 비교할 수 있습니다.
    print("사용자 활동 필터링 (최소 1회 추천):")
    min_actions_threshold = 1 # 이 값을 변경하여 다양한 시도를 할 수 있습니다. (예: 2, 3 등)
    x = train_actions_with_name.groupby('사용자ID').count()['추천점수'] >= min_actions_threshold
    cf_users = x[x].index
    print(f"최소 {min_actions_threshold}회 이상 추천한 사용자 ID:\n", cf_users)

    # 선택된 사용자를 대상으로 데이터 선별
    filtered_rating = train_actions_with_name[train_actions_with_name['사용자ID'].isin(cf_users)]
    print("필터링된 사용자 액션 데이터 형태:", filtered_rating.shape)
    print("필터링된 사용자 액션 데이터 미리보기:\n", filtered_rating.head())
    print("\n")

    # 다양하게 일정 상품만을 선택해서 분석
    # 모든 상품을 대상으로 분석할 것인지 결정
    # 예를 들어, `y = filtered_rating.groupby('상품ID').count()['추천점수'] > 1` 와 같이 변경하여
    # 1회 이상 추천된 상품만 선택하여 분석 결과를 비교할 수 있습니다.
    print("상품 활동 필터링 (최소 1회 추천된 상품):")
    min_product_actions_threshold = 1 # 이 값을 변경하여 다양한 시도를 할 수 있습니다. (예: 2, 3 등)
    y = filtered_rating.groupby('상품ID').count()['추천점수'] >= min_product_actions_threshold
    famous_productid = y[y].index
    print(f"최소 {min_product_actions_threshold}회 이상 추천된 상품 ID:\n", famous_productid)

    # 선택된 사용자의 정보에서 선택된 상품을 대상으로 최종 작업 데이터 선택
    final_ratings = filtered_rating[filtered_rating['상품ID'].isin(famous_productid)]
    print("최종 필터링된 데이터 형태:", final_ratings.shape)
    print("최종 필터링된 데이터 미리보기:\n", final_ratings.head())
    print("\n")

    # 사용자-상품 피벗 테이블 생성
    pt = final_ratings.pivot_table(index='상품ID', columns='사용자ID', values='추천점수')
    print("피벗 테이블 (초기):\n", pt)
    # 결측치는 추천하지 않은 경우이니 0 으로 채우기
    pt.fillna(0,inplace=True)
    print("피벗 테이블 (결측치 0으로 채움):\n", pt)
    print("\n")

    # 유사도 점수 테이블 생성 (코사인 유사도)
    similarity_scores = cosine_similarity(pt)
    print("유사도 점수 테이블 (일부):\n", similarity_scores[:3, :3]) # 일부만 출력
    print("\n유사도 점수 히트맵:")
    plt.figure(figsize=(6, 5))
    sns.heatmap(similarity_scores, annot=True, cmap='viridis', xticklabels=pt.index, yticklabels=pt.index)
    plt.title('상품 간 코사인 유사도')
    plt.xlabel('상품ID')
    plt.ylabel('상품ID')
    plt.show()
    print("\n")

    # 상품ID를 이용한 추천 함수
    def get_recommend_by_productid(product_id, top_n=5):
        # 협업 필터링 테이블에서 입력된 상품의 index 위치 찾기
        if product_id not in pt.index:
            print(f"오류: 상품ID {product_id}는 피벗 테이블에 존재하지 않습니다.")
            return []
        index = np.where(pt.index == product_id)[0][0]
        # 유사도 점수 테이블에서 입력된 상품의 index 위치 값 가져오기
        product_similarity_scores = similarity_scores[index]
        # 위치 값과 유사도 점수를 튜플로 묶고 리스트형 자료형으로 변환
        product_similarity_scores_list = list(enumerate(product_similarity_scores))
        # 유사도 점수를 기준으로 내림차순 정렬하고 자신을 제외한 나머지를 선택
        # 람다 함수 `lambda x: x[1]`는 튜플의 두 번째 요소(유사도 점수)를 기준으로 정렬함을 의미
        similar_items = sorted(product_similarity_scores_list, key=lambda x: x[1], reverse=True)[1:top_n+1]
        return similar_items

    # 다양한 상품과 유사한 상품을 구해서 서로 비교
    # 상품ID 1과 유사한 상품 구하기
    print("상품ID 1과 유사한 상품:")
    k_items_1 = get_recommend_by_productid(1)
    if k_items_1:
        for i_idx, score in k_items_1:
            print(f'상품ID: {pt.index[i_idx]}, 유사도: {score:.4f}')

    # 상품ID 3과 유사한 상품 구하기 (예시)
    print("\n상품ID 3과 유사한 상품:")
    k_items_3 = get_recommend_by_productid(3)
    if k_items_3:
        for i_idx, score in k_items_3:
            print(f'상품ID: {pt.index[i_idx]}, 유사도: {score:.4f}')
    print("\n")

    # 상품 관계 네트워크 그리기
    def plot_recommend_by_productid_graph(product_id):
        # 입력된 상품ID와 유사한 상품 구하기
        similar_items = get_recommend_by_productid(product_id)
        if not similar_items:
            return

        # 네트터크 객체 생성
        G = nx.Graph()
        G.add_node(product_id) # 중심 노드 추가
        # 유사 상품 노드 및 엣지 추가
        for i, score in similar_items:
            G.add_edge(product_id, pt.index[i], weight=score)

        pos = nx.spring_layout(G, k=0.5, iterations=50) # 노드 위치 설정 (k 값 조정으로 노드 간 간격 조절)
        labels = {node: f'상품{node}' for node in G.nodes()} # 노드 레이블 (한글 포함)
        weights = [G[u][v]['weight'] * 5 for u, v in G.edges()] # 유사도에 따라 엣지 두께 조절

        plt.figure(figsize=(6, 6))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=1500, font_size=10,
                node_color='skyblue', font_color='black', font_weight='bold',
                width=weights, edge_color='gray', alpha=0.7, edge_cmap=plt.cm.Blues)
        plt.title(f"상품ID <{product_id}>에 대한 상품 유사도 네트워크", fontsize=14)
        plt.show()

    print("상품ID 1에 대한 상품 유사도 네트워크를 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
    plot_recommend_by_productid_graph(1)
    print("상품ID 3에 대한 상품 유사도 네트워크를 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
    plot_recommend_by_productid_graph(3) # 다른 상품에 대해서도 시각화

    # 유사 상품들의 평균 추천점수 분포 시각화
    def plot_mean_recommend_by_productid(product_id):
        # 입력된 상품ID와 유사한 상품 구하기
        similar_items = get_recommend_by_productid(product_id)
        if not similar_items:
            return

        similar_productids = [pt.index[i[0]] for i in similar_items]
        # 각 유사 상품의 평균 추천 점수를 계산
        average_ratings = [final_ratings[final_ratings['상품ID'] == prod_id]['추천점수'].mean() for prod_id in similar_productids]

        plt.figure(figsize=(7, 4))
        plt.bar([f'상품{pid}' for pid in similar_productids], average_ratings, color='skyblue')
        plt.xlabel('유사 상품ID')
        plt.ylabel('평균 추천점수')
        plt.title(f'상품ID <{product_id}>과 유사한 상품들의 평균 추천점수')
        plt.ylim(0, 5) # 추천 점수 범위에 따라 y축 조정
        plt.show()

    print("상품ID 1과 유사한 상품들의 평균 추천점수 분포를 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
    plot_mean_recommend_by_productid(1)
    print("\n")

    # 협업 필터링 결과 저장 (피벗 테이블)
    cf_model_filepath = 'cf_pivot_table.pickle'
    with open(file=cf_model_filepath, mode='wb') as f:
        pickle.dump(pt, f)
    print(f"협업 필터링 피벗 테이블이 '{cf_model_filepath}'에 저장되었습니다.\n")

    return pt, get_recommend_by_productid

# --- 5. 인공지능 추천 모델링 (분류 문제) ---
def ai_recommendation_model(train_actions_with_name):
    print("--- 인공지능 추천 모델링 (분류 문제) ---")

    # 작업용 데이터 프레임으로 복사
    train_ai = train_actions_with_name.copy()
    print("AI 모델링을 위한 데이터 미리보기:\n", train_ai.head())
    print("AI 모델링을 위한 데이터 형태:", train_ai.shape)
    print("\n")

    # 문제 재정의: 상품ID를 종속변수로, 나머지를 독립변수로 설계
    # 즉, 사용자와 액션 정보를 기반으로 어떤 상품이 추천될지 (상품ID)를 예측

    # '추천일시' 컬럼 날짜형으로 변경
    train_ai['추천일시'] = pd.to_datetime(train_ai['추천일시'])
    # 추천일시 정보에서 데이터 분리 (년월일시분초)
    train_ai['yyyy'] = train_ai['추천일시'].apply(lambda x : x.year)
    train_ai['month'] = train_ai['추천일시'].apply(lambda x : x.month)
    train_ai['day'] = train_ai['추천일시'].apply(lambda x : x.day)
    train_ai['hh'] = train_ai['추천일시'].apply(lambda x : x.hour)
    train_ai['mm'] = train_ai['추천일시'].apply(lambda x : x.minute)
    train_ai['ss'] = train_ai['추천일시'].apply(lambda x : x.second)
    print("시간 정보 분리 후 데이터 미리보기:\n", train_ai.head())
    print("\n")

    # 상품명 정수형 인코딩 (fit과 transform을 한 번에)
    le_product_name = LabelEncoder()
    train_ai['상품명'] = le_product_name.fit_transform(train_ai['상품명'])
    print("상품명 정수형 인코딩 후 데이터 미리보기:\n", train_ai.head())

    # 상품구분 정수형 인코딩 (fit과 transform을 한 번에)
    le_product_type = LabelEncoder()
    train_ai['상품구분'] = le_product_type.fit_transform(train_ai['상품구분'])
    print("상품구분 정수형 인코딩 후 데이터 미리보기:\n", train_ai.head())
    print("\n")

    # 학습에 필요한 컬럼만 선택
    # 학습에 일부 컬럼만 사용할 필요가 있을 수 있습니다. 일부 컬럼명을 이용하여 데이터를 선택하여 학습한 후 다른 결과와 비교해 보세요.
    # 예시: train_ = train_ai[['사용자ID', '추천점수', '상품명', 'yyyy', 'month', '상품ID']]
    # 여기서는 모든 추출된 특징과 사용자ID, 추천점수를 사용합니다.
    features = ['사용자ID', '추천점수', '상품명', '상품구분', 'yyyy', 'month', 'day', 'hh', 'mm', 'ss']
    target = '상품ID'
    train_final_ai = train_ai[features + [target]]
    print("AI 모델 학습을 위한 최종 컬럼 선택 후 데이터 형태:", train_final_ai.shape)
    print("AI 모델 학습을 위한 최종 데이터 미리보기:\n", train_final_ai.head())
    print("\n")

    # 데이터 섞기 (전체를 대상으로 무작위 표본 샘플)
    train_final_ai = train_final_ai.sample(frac=1, random_state=42).reset_index(drop=True)
    print("데이터 섞기 후 데이터 미리보기:\n", train_final_ai.head())
    print("\n")

    # 학습 데이터와 검증 데이터 분리
    # 학습 데이터와 검증 데이터를 분리하는 건수는 정해져 있지 않습니다. 다양한 크기로 분리하여 학습과 검증해 보시기 바랍니다.
    # 현재 55건의 데이터 중 45건 학습, 10건 검증으로 사용합니다.
    split_point = 45 # 이 값을 변경하여 다양한 분리 시도 가능 (예: 40, 50 등)
    train_data_ai = train_final_ai[:split_point]
    valid_data_ai = train_final_ai[split_point:]

    print(f"학습 데이터 형태: {train_data_ai.shape}")
    print(f"검증 데이터 형태: {valid_data_ai.shape}")

    # 학습용 데이터 준비 (특징 및 타겟)
    x_train_ai = train_data_ai[features]
    y_train_ai = train_data_ai[target]

    print(f"x_train_ai 형태: {x_train_ai.shape}, y_train_ai 형태: {y_train_ai.shape}")

    # 검증용 데이터 준비 (특징 및 타겟)
    x_valid_ai = valid_data_ai[features]
    y_valid_ai = valid_data_ai[target]

    print(f"x_valid_ai 형태: {x_valid_ai.shape}, y_valid_ai 형태: {y_valid_ai.shape}")
    print("\n")

    # 예측 대상 미리보기
    print("y_train_ai (예측 대상) 미리보기:\n", y_train_ai[:6])
    print(f"y_train_ai 최대값: {max(y_train_ai)}")
    print("y_valid_ai (예측 대상) 미리보기:\n", y_valid_ai)
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
    # 'num_class'는 상품ID의 고유 개수 (7개)에 맞게 설정
    params_ai = {
        'n_estimators': 500,     # 부스팅 라운드 수 (트리 개수)
        'learning_rate': 0.01,   # 과적합을 방지하기 위한 학습률
        'num_leaves': 30,        # 한 트리에 가질 수 있는 최대 잎 노드 수
        'objective': 'multiclass', # 다중 클래스 분류를 위한 목적 함수
        'metric': 'multi_logloss', # 학습 중 평가 지표
        'num_class': train_final_ai[target].nunique(), # 타겟 클래스 수 (상품ID의 고유 값 개수)
        'random_state': 42,      # 결과 재현성을 위한 시드 값
        'verbosity': -1,         # 학습 중 상세 출력 억제
        'n_jobs': -1             # 사용 가능한 모든 CPU 코어 사용
    }

    # LightGBM 분류기 초기화 및 학습
    print("LightGBM 모델 학습 중...")
    ai_model = lgb.LGBMClassifier(**params_ai)
    ai_model.fit(x_train_ai, y_train_ai,
              eval_set=[(x_valid_ai, y_valid_ai)], # 검증 세트 설정
              eval_metric=lgbm_microf1,          # 사용자 정의 평가 지표 사용
              callbacks=[lgb.early_stopping(10, verbose=False)]) # 검증 지표가 10라운드 동안 개선되지 않으면 조기 종료
    print("AI 모델 학습 완료.\n")

    # 검증 세트에 대한 예측 수행
    y_pred_ai = ai_model.predict(x_valid_ai)
    valid_data_ai['pred'] = y_pred_ai # 예측 결과를 검증 데이터 DataFrame에 추가

    print("AI 모델 예측 결과가 추가된 검증 데이터:\n", valid_data_ai)
    print("\n")

    # AI 모델 성능 평가 (F1-스코어)
    score_ai = f1_score(y_valid_ai, y_pred_ai, average='micro')
    print(f'AI 모델 F1-스코어 (검증 세트): {score_ai:.4f}\n')

    # 변수 중요도 확인
    val_imp_ai = pd.DataFrame(ai_model.feature_importances_, index=ai_model.feature_name_, columns=['imp'])
    print("AI 모델 상위 10개 변수 중요도:")
    print(val_imp_ai.sort_values(by='imp', ascending=False).head(10))

    # 변수 중요도 시각화
    print("\nAI 모델 변수 중요도 플롯을 표시합니다. 플롯을 닫아야 다음 단계로 진행합니다.")
    val_imp_ai_sorted = val_imp_ai.sort_values(by='imp', ascending=True) # 가로 막대 그래프를 위해 오름차순 정렬
    plt.figure(figsize=(10, len(val_imp_ai_sorted) * 0.5)) # 동적인 그림 높이 설정
    val_imp_ai_sorted['imp'].plot(kind='barh') # 가독성을 위해 가로 막대 그래프 사용
    plt.title('AI 모델 특징 중요도')
    plt.xlabel('중요도')
    plt.ylabel('특징')
    plt.tight_layout()
    plt.show()
    print("\n")

    # AI 모델 저장
    ai_model_filepath = 'ai_recommender_model.pickle'
    with open(file=ai_model_filepath, mode='wb') as f:
        pickle.dump([ai_model, params_ai, valid_data_ai], f)
    print(f"AI 모델 및 관련 객체가 '{ai_model_filepath}'에 저장되었습니다.\n")

    return ai_model, le_product_name, le_product_type # 인코더도 반환하여 예측 시 디코딩에 활용 가능

# --- 메인 실행 함수 ---
def main():
    demonstrate_metrics() # 평가 지표 예제 실행

    # 데이터 로딩 및 초기 전처리
    train_product, train_actions, train_actions_with_name = load_and_preprocess_data()

    # 협업 필터링 추천 시스템 실행
    pt, get_recommend_by_productid_func = collaborative_filtering_recommender(train_actions_with_name.copy())

    # 협업 필터링 모델 로딩 예제
    print("--- 저장된 협업 필터링 모델 로딩 및 확인 ---")
    try:
        with open(file='cf_pivot_table.pickle', mode='rb') as f:
            loaded_pt = pickle.load(f)
        print("로드된 피벗 테이블 미리보기:\n", loaded_pt.head())
    except FileNotFoundError:
        print("저장된 협업 필터링 피벗 테이블 파일을 찾을 수 없습니다.")
    print("\n")


    # 인공지능 추천 모델링 실행
    ai_model, le_product_name, le_product_type = ai_recommendation_model(train_actions_with_name.copy())

    # AI 모델 로딩 및 검증 예제
    print("--- 저장된 AI 모델 로딩 및 검증 ---")
    try:
        with open(file='ai_recommender_model.pickle', mode='rb') as f:
            loaded_ai_object = pickle.load(f)
        loaded_ai_model = loaded_ai_object[0]
        loaded_params_ai = loaded_ai_object[1]
        loaded_valid_data_ai = loaded_ai_object[2]

        # x_valid_ai를 다시 준비 (main 함수 내에서 직접 참조하지 않으므로)
        features = ['사용자ID', '추천점수', '상품명', '상품구분', 'yyyy', 'month', 'day', 'hh', 'mm', 'ss']
        # train_actions_with_name을 다시 전처리하여 x_valid_ai 생성
        temp_train_ai = train_actions_with_name.copy()
        temp_train_ai['추천일시'] = pd.to_datetime(temp_train_ai['추천일시'])
        temp_train_ai['yyyy'] = temp_train_ai['추천일시'].apply(lambda x : x.year)
        temp_train_ai['month'] = temp_train_ai['추천일시'].apply(lambda x : x.month)
        temp_train_ai['day'] = temp_train_ai['추천일시'].apply(lambda x : x.day)
        temp_train_ai['hh'] = temp_train_ai['추천일시'].apply(lambda x : x.hour)
        temp_train_ai['mm'] = temp_train_ai['추천일시'].apply(lambda x : x.minute)
        temp_train_ai['ss'] = temp_train_ai['추천일시'].apply(lambda x : x.second)
        temp_train_ai['상품명'] = le_product_name.fit_transform(temp_train_ai['상품명'])
        temp_train_ai['상품구분'] = le_product_type.fit_transform(temp_train_ai['상품구분'])
        temp_train_ai_shuffled = temp_train_ai.sample(frac=1, random_state=42).reset_index(drop=True)
        temp_valid_data_ai = temp_train_ai_shuffled[45:]
        x_valid_ai_for_load_test = temp_valid_data_ai[features]
        y_valid_ai_for_load_test = temp_valid_data_ai['상품ID']

        loaded_valid_data_ai['pred_loaded'] = loaded_ai_model.predict(x_valid_ai_for_load_test)
        print("로드된 AI 모델의 예측 결과가 추가된 검증 데이터 (상위 5행):")
        print(loaded_valid_data_ai.head())

        score_loaded_ai = f1_score(y_valid_ai_for_load_test, loaded_valid_data_ai['pred_loaded'], average='micro')
        print(f'로드된 AI 모델의 F1-스코어: {score_loaded_ai:.4f}\n')

    except FileNotFoundError:
        print("저장된 AI 모델 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"AI 모델 로딩 및 검증 중 오류 발생: {e}")

    print("모든 프로그램 실행 완료.")

if __name__ == "__main__":
    main()