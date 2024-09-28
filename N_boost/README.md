# 제 1강

## 1. 정형데이터(Structured Data)
데이터베이스(DB)와 같이 정해진 구조에 따라서 정리되어 있는 데이터. 이는 엑셀의 표와 같이 고정된 열(column)과 행(row)을 가지고 있습니다. 정형 데이터의 중요성은 그 구조 덕분에 검색이나 분석이 용이하다는 점에 있습니다.

## 비정형데이터(Unstructured Data)
미리 정의된 형식이나 모델이 없는 데이터. ex)txt문서, 이메일, 소셜 미디어 게시물, 동영상, 오디오파일, 웹 페이지

## 데이터의 이해, 문제 이해
데이터의 이해는 데이터의 특성, 형태, 분포 등을 파악하는 것을 의미합니다. 문제 이해는 해당 데이터를 바탕으로 어떤 문제를 해결하려 하는지, 그 문제의 정의와 목표를 명확히 이해하는 것입니다.

## 평가지표 이해
모델의 성능을 측정하는 기준을 의미합니다. 분류와 회귀 문제에서 사용하는 평가지표는 다릅니다.

### 분류, 회귀 평가지표
분류 - 값이 범주형 데이터인 경우에 사용되며, 정확도, 정밀도, 재현율 등이 평가지표로 사용.
회귀 - 예측 값이 연속형 데이터인 경우에 사용되며, 평균 제곱 오차(MSE), 평균 절대 오차(MAE) 등이 평가지표로 사용.

### Confusion Matrix(혼돈 행렬)
분류 문제에서 모델의 성능을 평가하는 방법 중 하나. True positive, False positive, True negative, False negative 이렇게 4가지 경우로 구분하여 모델의 성능을 평가.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/06846a3a-6c10-4f89-a1fe-73323131265c)


- Accuracy(정확성)
정확도는 모델이 예측한 결과가 실제 값과 얼마나 일치하는지를 나타내는 지표입니다. 전체 예측 중에서 올바르게 예측한 비율을 의미합니다.
(TP+TN)(=맞춘 경우의 수)/(전체 경우의 수)

- Precision(정밀도)
정밀도는 모델이 True로 예측한 것 중에서 실제로 True인 것의 비율을 의미합니다. ex)스팸메일 - 실수로 스팸을 놓치더라도 일반메일을 스팸처리 하지 않도록 하는 것에 초점
TP/(TP +FP)

-  Recall
재현율은 실제 True인 것 중에서 모델이 True로 예측한 것의 비율을 의미합니다. ex)악성종양을 일반종양으로 잘못 분류하면 안된다.
TP/(TP + FN)

- ROC AUC(수신자 조작 특성)
ROC 곡선은 False Positive Rate에 대한 True Positive Rate의 그래프입니다. AUC는 이 ROC 곡선의 면적을 의미하며, 값이 1에 가까울수록 모델의 성능이 좋다고 판단합니다. 

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/3ae0888c-8305-4305-afb0-4a318055cee4)



# 제 2강 정형데이터 분류 EDA

## EDA(Exploratory DAta Analysis) 탐색적 데이터 분석
데이터의 주요 특성을 탐색하거나 데이터에 숨겨진 패턴을 찾는데 도움이 되는 기법으로, 데이터를 탐색하고 가설을 세우고 증명하는 과정
- 데이터 체크 : 데이터 확인. 크기, 변수의 수, 데이터 일부를 확인하여 형태와 변수를 파악함
- 기술 통계량 확인 : 경향, 분포, 분산 등 기술통계량을 확인. 평균 중앙값 최빈값 표준편차 범위 등을 확인하여 전반적인 특성을 파악
- 결측치 확인
- 이상치 확인
- 데이터 시각화 : ex)히스토그램, 박스 플롯, 산점도
- 상관 분석 : 변수 간의 상관 관계를 분석. 피처선택이나 모델 구성에 중요한 영향
  
## 연속형 변수(Continuous Variable)
숫자로 구성된 변수로, 그 값이 연속적인 범위 내에서 어떤 값이든 가질 수 있는 변수를 의미. ex)키, 몸무게, 온도.

## 범주형 변수(Categorical Variable)
명확하게 구분되는 카테고리나 범주로 나뉘는 변수. 일반적으로 문자열이나 정수값이며 그 숫자도 특정 카테고리를 나타냄.ex)성별, 학년, 혈액형
- 순서형 변수(Ordinal Variable) : ex)학년
- 명목형 변수(Nominal Variable) : ex)혈액형
  
## EDA 가설 검정
가설이 올바른지 판단하는 과정
- 가설 설정 : 데이터 분석 목적과 EDA를 통해 얻은 통찰력을 바탕으로 검증하고자 하는 가설을 설정.
- 분석 방법 선택 : 적절한 통계방법 선택. ex)t-검정, 카이제곱 검정, ANOVA [본 강의에선 다루지 않음]
- 분석 수행 : p-값 등의 통계량을 계산. 
- 결과 해석 : p-값이 5% 미만일 경우 대립가설을 채택
  
***p-(p-value) : 귀무가설(null - hypothesis)가 참일 경우 극단적인 결과를 얻을 확률. 우리가 관찰한 결과가 우연히 발생한 확률***

# 제 3강 - 데이터 전처리
데이터 분석이나 모델 학습에 앞서 데이터를 적절하게 가공하는 과정

## Scaling
변수의 범위를 조정하는 과정. 각 변수가 가진 값의 범위가 다르면 모델이 변수를 공정하게 학습하지 못함.**연속형 데이터 처리에 사용**
- Min-Max Scaling (정규화) :  데이터의 최소값과 최대값을 사용하여 데이터를 0과 1 사이의 범위로 스케일링 **(X - Xmin) / (Xmax - Xmin)**
- Standard Scaling (표준화) :  데이터의 평균을 0, 표준편차를 1로 만들어 데이터의 분포를 표준 정규 분포로 만드는 방법 **(X - X평균) / X표준편차**
- Robust Scaling : 데이터의 중앙값(median)과 IQR(Interquartile Range, Q3 - Q1)을 사용 **(X - X중앙값) / IQR**
  ***아래로 내려올수록 이상치의 영향을 적게 받음***
  
## 로그 변환(Log Transformation)
데이터의 왜도를 줄이고 정규분포에 가깝게 만듬. **비율이나 백분율 데이터에 주로 사용**
## Quantile Transformation
분위수를 이용하여 데이터를 재배치하여 데이터를 원하는 분포를 원하는 분포로 변환

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/e8489ed1-47e5-4a2e-b3df-57ef3096b91f)

## Binning
데이터를 여러 구간으로 나누는 방법. **연속형 범주 ->범주형 변수**. overfitting방지.
- 등간격 비닝(Equal-width Binning) : 전체 값의 범위를 동일한 너비의 구간으로 나눔.ex)나이 변수를 10살 간격으로 나눔.
 
 ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/e9762fe2-33d3-4c3b-b17e-b80867332f1d)

- 등빈도 비닝(Equal-frequency Binning) : 각 구간에 동일한 개수의 데이터가 포함되도록 구간을 나눔.ex)소득 변수를 사분위수로 나눔.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/327138b5-f84c-4caa-b1a5-97b8489791f8)

## Encoding
범주형 데이터를 숫자형 데이터로 변환.
- 원-핫 인코딩(One-Hot Encoding): 범주형 변수의 각 범주를 이진변수(0 or 1)로 변환.ex)'색깔'범주 中 빨강, 파랑, 노랑 이라면 '색깔_빨강', '색깔_파랑', '색깔_녹색'
- 라벨 인코딩(Label Encoding): 범주형 변수의 각 범주를 고유한 정수로 변환. ex) '색깔'범주 中 빨강, 파랑, 노랑 이라면 각각을 0,1,2로 변환
- 빈도 인코딩(Frequency Encoding) : 범주형 변수의 각 범주를 전체 데이터에서 차지하는 빈도로 변환하는 방법. **특정 범주의 빈도가 중요한 정보를 가질 때 유용**
- 타겟 인코딩(Target Encoding) : 범주형 변수의 각 범주를 타겟 변수의 평균값으로 변환. **범주와 타겟 변수 사이의 관계를 인코딩에 반영, 예측성능up**.데이터 누출방지작업이 필요
- 임베딩(Embedding) : 범주형 변수의 각 범주를 고차원 벡터로 변환.**딥러닝에 활용, 자연어 처리에 효과적**.ex)단어 임베딩(word embedding)

### 임베딩 中 World2vec
주변 단어들이 주어졌을 때 해당 단어를 예측하거나, 반대로 해당 단어가 주어졌을 때 주변 단어를 예측하는 방식.**비슷한 문맥에서 사용되는 단어는 비슷한 벡터값을 가져 의미간 관계를 잘 표현함**

# 제 3-2강 데이터 전처리 - 결측치 처리
특정 변수의 값이 누락된 상태.**적절하게 처리하지 않으면 데이터 분석의 결과를 왜곡하거나 모델의 성능을 저하시킬 수 있음**

## pattern
결측치가 무작위로 발생한 것인지 특정 패턴을 따라 발생한것인지 파악.

## Univariate
단일 변수의 정보만을 사용하여 결측치를 처리. ex) 특정값으로 채우기. [수치형 변수 - 평균/중앙값],[범주형 변수 - 최빈값]
- 제거 : 결측치가 많을 경우 많은 데이터를 손실. 결측치 비율이 매우 낮을 때, 결측치가 무작위로 발생했을 때, 데이터가 충분히 많을 때
- 평균값 삽입 : 수치형 변수에 적용. 데이터 분포를 크게 왜곡시키지 않음. 이상치의 영향을 크게 받음.
- 중위값 삽입 : 수치형 변수에 적용. 이상치의 영향을 덜 받음.
- 상수값 삽입 : 수치/범주에 모두 적용. 결측치가 특정 의미를 가질 때 유용

## Multivariate
다변량 정보를 사용하여 결측치를 처리. ex)다른 변수들 값에 따라 결측치를 예측하는 머신러닝 모델 사용.
- 회귀분석 : 결측치가 있는 변수를 종속 변수로, 결측치가 없는 변수를 독립 변수로 하는 회귀모델을 학습하여 결측치를 예측.**선형 관계에선 잘 반영하나 비선형 관계는 잘 반영하지 못함**.ex)나이에 결측치, 성별 소득엔 노결측이면 나이 =a, 성별 +b 소득 +c 와 같은 회귀모델을 학습
- KNN(K- Nearest Neighbors) : 결측치가 있는 행과 가장 가까운 k개의 행에서 평균값 또는 중앙값으로 결측치를 채움.**비선형 관계에도 강하며 범주형 변수에도 적용 가능** 계산 비용이 높으며 이상치 영향을 받음.

### Multivarited의 합리적 접근법(예시)
- 행정구역인구, 관할소방서 인원 : 각 행정구역별 각 연도의 편균값으로 대체
- 강수량 : 비가오는날이 적기 때문에 0으로 대체
- 토지이용상황명, 도로측면명, 용도지역지구명 : 최빈값으로 대체
- 온도, 습도, 풍속 : 각 행정구역별 해댕 일의 값으로 대체
- 풍향 : 16방위를 참조하여 범주화 진행이후 해당일의 풍향으로 대체

## 이상치(Outlier)
데이터 분포에서 상대적으로 동떨어진 값으로 다른 값들과 많이 다른 값. 데이터 수집과정에서 오류로 발생할 수도 있고 특이한 실제 현상을 반영하는 경우도 있음.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/75b06ab8-21ed-4900-9f69-504b823d7f3c)


### 이상치 탐색
- Z-score : 데이터 평균에서 표준편차의 3배 이상 떨어진 데이터를 이상치로 보는 방법
- IQR(Interquartile Range, 사분위수 범위) : 1사분위수(Q1)에서 1.5IQR이하 또는 3사분위수(Q3)에서 1.5IQR이상 떨어진 데이터를 이상치로 보는 방법

# 제 4강 머신러닝
컴퓨터가 학습할 수 있도록 하는 인공지능의 분야.

## Underfitting & Overfitting
- fit : 데이터를 잘 설명할 수 있는 능력. 적절하게 설명하지 못하거나 노이즈까지 학습하여 일반화할 수 있음.
  
### Regularization (정규화)
머신러닝 모델의 복잡성을 제어하여 오버피팅을 방지하는 기법.
- Early stopping : 모델의 성능이 개선되지 않을 때 학습을 중단.**정형 데이터에서도 사용 가능.**

  ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/8a9918f2-2a3b-4a98-adc1-526eb5723bf5)

- Parameter norm penalty : 모델의 가중치에 제한을 두는 방법.**정형 데이터에서도 사용 가능.**

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/9948bcd4-c330-49cf-a07b-7b0ff3419581)

- Data augmentation : 기존의 학습 데이터를 변형하여 새로운 학습 데이터를 생성. ex)이미지 회전 및 반전.**정형 데이터에서도 사용 가능.**
- Noise robustness : 학습 데이터에 노이즈를 추가하여 모델이 노이즈에 강건하도록 만듬.
- Label smoothing : 학습 데이터의 레이블을 약간 변경하여 모델이 너무 확신하지 않도록 만듬.

  ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/6f14702c-4bd2-4178-a6fc-d4c2fc50ff38)

- Dropout : 신경망의 일부 뉴런을 무작위로 비활성화시켜 학습을 진행하여, 모델이 특정 뉴런에 과도하게 의존하는 것을 방지하고 일반화 성능을 향상.**정형 데이터에서도 사용 가능.**

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/53352424-8d63-4c03-9087-9e6ca6066f94)

- Batch normalization : 학습 과정에서 배치단위로 데이터의 분포를 정규화하여 학습 속도를 향상시키고, 초기 가중치 선택에 대한 의존성을 줄임.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/ee4e0cca-b15b-4e0d-922b-67bd657bfdf8)

## Validation strategy (검증 전략)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/1f9f8a12-5510-451f-87ea-b981b590dd75)

- Test : 모델의 최종 선능을 평가하는데 사용하며, 모델학습과 검증과정에서 전혀 사용되어선 안됨.
- 검증 : 모델의 성능을 평가하고 하이퍼파라미터를 튜닝하는데 사용. 일반화 되었는지 확인 후 모델 설정을 조절함. **test 와 유사한 구조**
- train 
### Holdout Validation
데이터셋을 훈련과 검증으로 나누고, 훈련세트로 모델을 학습시킨 후 검증세트로 모델의 성능을 평가하는 방법

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/e2c307a3-7243-46db-905e-e6f56d1c575d)

### Cross Validation
데이터 셋을 여러 부분으로 나누고 각 폴드를 검증세트로 사용하면서 모델을 여러 번 학습하고 평가하는 방법
- k-Fold 
- Stratified K-Fold
- Group K-fold
- Time series

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/a1a021f4-68d3-45ee-a3ff-2c00f0194091)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/6b256e81-02c4-4b13-a5ca-9d28fc39e6a9)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/9b361dec-f693-4c6c-9e05-e80a35d00a8d)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/c06ef81f-c47d-40ca-9435-868e42a4abbc)


## Reproducibility (재현성)
동일한 조건 하에서 동일한 결과를 얻을 수 있는 능력

- fix seed : 무작위성을 가지는 알고리즘에서 항상 동일한 결과를 얻기 위해 사용하는 기법.**모델의 성능을 평가하거나 타인이 실험을 재현할 수 있게 함**

## Machine learning workflow
머신러닝 모델을 개발하는 모든 과정. **문제정의 - 데이터 수집 - 데이터 전처리 - 데이터 분석 및 시각화 - 모델 선택 및 훈련 - 모델 검증 및 테스트 - 모델 베포 및 모니터링**

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/e1354490-f798-45c7-915c-d749c929fb6f)

# 제 5강 tree model
머신러닝 알고리즘 중 하나로, 의사결정트리라고도 함. 데이터를 분석하여 데이터 사이의 패턴을 나무 구조로 표현

## Decision Tree (의사결정나무)
칼럼(feature) 값들을 어떠한 기준으로 그룹을 나누어 목적에 맞는 의사결정을 만드는 방법.하나의 질문으로 YES or NO로 결정을 내려서 분류

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/40698e3d-284f-43fc-9bc7-0a1343b9fadb)

## Bagging & Boosting
앙상블 학습 기법을 사용할때 주로 쓰이는 방법. 여러 모델을 학습시켜 그 결과를 결합하여 단일 모델보다 더 좋은 성능을 얻는 것이 목표.
- Bagging (Bootstrap Aggregating) : 원래 데이터 셋에서 랜덤하게 복원추출(bootstrap)한 여러 서브셋을 만들고, 각각의 서브셋으로 모델을 독립적으로 학습시킨 후 이 모델들의 예측을 결합(aggregating).**모델의 분산을 줄이는데 효과적으로 과적합을 방지**.ex)랜덤 포레스트
  
  ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/bc9969b1-f6cb-4310-8144-6aee72052304)
  ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/36838265-da88-42f0-8fd4-bf7a513f9511)
  ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/722f1588-3bd3-4f03-a86a-2be40e37c2df)

- Boosting : 여러개의 성능이 낮은 모델을 순차적으로 학습시키되, 각 단계에서 이전 모델이 잘못 예측한 데이터에 대해 더 높은 가중치를 주어 다음 모델을 학습.**모델의 편향을 줄이는데 효과적**.ex)Gradient Boosting
  
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/b8ae3d82-539c-4255-8a50-a58f617b9d72)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/153ff833-5cbe-40a1-86c4-0a40f028f7b0)

## LightGBM, XGBoost, CatBoost
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/d9f1c58c-721a-42b9-ab97-31517de641a7)

### LightGBM (Gradient Boosting Framework)
알고리즘의 메모리 사용량이 적고 실행속도가 빠름. Leaf-wise트리 분할방법을 사용하는데, 기존의 Level-wise보다 더 복잡한 트리를 만들지만, 예측 오차를 더 많이 줄일 수 있다.

### XGBoost (Extreme Gradient Boosting)
속도와 성능을 최적화하고, 누럭된 값 처리 및 정규화 기능 제공. 병렬처리 기능을 사용하여 학습과 분류가 빠름

### CatBoost (Category Boosting)
범주형 데이터에 강함. 범주형 변수를 자동으로 변환하는 기능. 데이터 전처리 과정의 간소화. 과적합 방지 기능 내장

## hyper parameter
파라미터와 구분하여 사용자가 딥러닝을 위해 설정하는 값들을 모두 지칭.
- Max_depth (트리의 깊이) : 트리의 깊이를 결정하는 하이퍼파라미터. 깊이가 깊을수록 모델이 복잡해지고 overfitting의 위험이 커짐.
- Min_sample_leaf (리프 노드의 최소샘플 수) : 이 값이 클수록 트리의 분기가 줄어들어 모델이 단순해짐.
- Learning_rate (학습률) : 부스팅 모델에서는 각 트리가 예측에 기여하는 정도를 조절하는 학습률이 중요. 일반적으로 학습률이 낮을수록 더 많은 트리가 필요하지만, 보다 안정적인 모델을 얻을 수 있음.
  
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/e51b3eed-b8cb-4e84-a8d3-c94336d6ae99)

- N_estimators (트리의 수) : 부스팅 모델에서는 학습을 위해 생성되는 트리의 수를 결정. 트리의 수가 많을수록 모델은 복잡해지고 overfitting의 위험이 커짐.

# 7강 8차시 피처 엔지니어링(Feature Engineering)
머신 러닝 알고리즘이 작동할 수 있도록 Feature들을 만드는 과정으로 데이터에 대한 도메인 지식을 이용.**모델 정확도를 높이기 위함**

## pandas Group By Aggregation
여러개의 관련 데이터를 그룹화하여 그룹별 통계를 계산하는 방법. ex)기존 구매고객 id에서 구매금액의 평균,max,min등의 새로운feature값을 사용.**시계열 데이터, 카테고리형 데이터**
- total sum Feature : 여러 피처의 값을 합산하여 새로운 feature를 생성.**피처의 총합이 중요한 의미를 가질때 사용**
 ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/0c0e7812-b7a8-4d72-9004-0865e575ec51)
- price/total/quantity sum Feature
- price/total/quantity count Feature : 수량
- price/total/quantity mean Feature : 평균값
- price/total/quantity min Feature : 최소값
- price/total/quantity max Feature : 최대값
- price/total/quantity std Feature : 표준편차
- price/total/quantity skew Feature : 비대칭도(쏠림)**왼쪽 - 음의 값/ 오른쪽 - 양의 값/ 대칭 - 0**

** Cross Validation을 이용한 Out of Fold 예측
모델 학습 시 교차검증(여러개의 폴드로 나누고, 각 폴드를 차례대로 검증)을 적용하여 각 데이터 포인트에 대한 예측값을 생성.**모델 성능의 정확한 평가, 스태킹과 같은 앙상블방법에 사용**

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/fc894115-9ddb-4c31-a92c-ead4e1ae3545)

- 앙상블 방법 : 여러개의 기본 모델을 학습시킨 후 이들의 예측을 조합하는 방법. **개별 모델의 성능이 한정적일 때도 더 높은 성능을 얻을 수 있음**
- 스태킹(Stacking) : 여러개의 다른 모델들의 예측값을 새로운 데이터로 사용하여 추가적으로 다른 모델(메타 모델 또는 2단계 모델)을 학습시키는 방법.**overfitting주의하기**

** LightGBM Early Stopping

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/f783522c-ae40-41c5-8bdd-6f4d51dd27e0)

# 7강 9차시 Feature Importance
머신러닝 모델에서 피처가 예측을 수행하는 데 얼마나 중요한 역할을 하는지 측정하는 방법.
- model specific : 특정 모델의 내부 구조를 사용하여 피처 중요도를 계산.ex)결정트리 기반 알고리즘(랜덤 포레스트, 그래디언트 부스팅)**특정 모델에만 사용 가능** **머신 러닝 모델 자체에서 피처 중요도 계산이 가능할때 채택**
- model agnostic : 특정 피처의 값을 무작위로 섞어서 모델의 성능 변화를 측정하여 피처 중요도를 계산.**모든 모델에 사용 가능하지만, 복잡한 상호작용을 완벽반영못함**

## Boosting tree
일반적으로 feature 중요도를 feature가 모델에 포함된 트리에서 분할에 사용된 횟수, 분할에 사용된 피처로 인해 감소한 손실 함수의 총량 등으로 측정.디폴트 계산 방식은 weight
-LightGBM : Gradient Boosting 알고리즘을 기반으로 한 모델. 
[1 - split방식 : 피처가 트리의 분기점에서 사용된 횟수를 기반으로 중요도 계산] 
[2 - gain 방식 : 각 피처가 모델의 손실을 감소시키는 데 얼마나 기여했는지 평가] **gain방식은 피처의 효과가 큰 분기점에서 더 많이 사용된 피처를 중요하게 간주**
- XGBoost : Gradient Boosting 알고리즘을 기반으로 한 모델. 각 피처가 모델 성능 향상에 얼마나 기여했는지 평가하는데 사용할 수 있는 피처 중요도를 계산.**피처가 트리의 분기점에서 사용된 횟수를 기반으로 함. 즉 피처가 많이 사용될수록 그 피처의 중요도가 높아짐**

## CatBoost 피처 중요도
'PredictionValuesChange' 방식으로 특정 피처를 사용하여 트리의 분기점을 만드는 경우, 그 피처가 예측값을 얼마나 변화시키는지 측정. 값이 클수록 해당 피처가 중요하다고 판단. **범주형 피쳐를 잘 처리함** 인자의type, 디폴트는 FeatureImportance

## Permutation 피처 중요도
 특정 피처의 값을 무작위로 섞어 모델 성능의 변화를 관찰하여 피처의 중요도를 측정.

## 피처 선택 (Feature Selection)
모델 성능 향상, 과적합 방지, 계산 효율성 향상 등을 위해 중요한 피처를 선택, 중요하지 않은 피처를 제거.
- Filter : 통계적 측정 방법을 통해 피처간의 상관 관계를 계산.**빠르고 간단하지만 선택된 피처가 반드시 성능 향상에 기여한다는 보장은 없음**ex)카이제곱 테스트, ANOVA, 상관 계수. 빠른 장점 때문에 전처리 과정에서 많이 사용

  ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/d61b1ac5-5e6d-4da8-a865-cc267b5cf4df)

- Wrapper : 예측 모델을 사용하여, 특정 모델에 대해 피처의 부분 집합을 반족적으로 선택하고, 모델의 성능을 통해 가장 좋은 피처 집합을 찾음.**성능에 직접 기여하는 피처를 찾을 수 있지만 계산 비용이 높음**ex)순방향 선택, 후방향제거, 재귀적피처 제거

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/d19bf0d1-c7b4-4ded-a170-d464dbf12630)

- Embedded : 학습 과정에서 피처 선택이 이루어 짐. **계산 비용이 적고, 모델 성능에 직접적으로 기여하는 피처를 선택 가능**ex)Lasso, Ridge 회귀, 트리 기반 모델

# 10차시 하이퍼 파라미터 (Hyper Parameter) 튜닝
머신러닝 모델의 학습 과정을 제어하는 변수. 학습 과정에서 결정되지 않고 사용자가 미리 설정해야 함. 모델의 성능 최적화에 큰 영향을 미침.

## 하이퍼 파라미터 튜닝 방법
- Manual Search : 자동화 툴을 사용하지 않고 매뉴얼하게 실험할 하이퍼 마라미터 셋을 정하고 하나씩 바꿔가면서 테스트하는 방식
- Grid Search : 모든 가능한 하이퍼파라미터 조합을 시도해보는 방법.
- Random Search : 하이퍼파라미터의 값들을 무작위로 선택하는 방법.
- Bayesian optimization : 베이지안 통계를 이용하여, 이전의 탐색 결과를 바탕으로 다음에 탐색할 값을 예측하는 효율적 탐색 방법.

## Boosting tree 하이퍼 파라미터
앙상블 방법 중 하나로, 여러개의 결정 트리를 순차적으로 학습

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/2a7b62f2-706c-4e7f-9c71-3a32449b3017)

## Optuna
하이퍼 파라미터 튜닝 프레임워크. 베이지안 최적화를 기반으로 복잡한 하이퍼파라미터 공간에서도 효율적 탐색.**병렬계산, 학습중단, 학습 과정의 시각화 기능 제공** 자동으로 최적화.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/30db81d1-30ed-4a81-91b3-43e149fe5f75)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/e38074cd-8825-478a-b167-889f7a867f93)

Storage API를 사용해서 하이퍼 파라미터 검색 결과 저장 가능. RDB, Redis와 같은 Persistent 저장소에 하이퍼 파라미터 탐색 결과를 저장하여, 다음에 다시 이어서 탐색이 가능함.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/36505822-3810-42c9-adc2-0d8ab9ce4f12)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/9b49ea50-dca2-4513-9d7f-deb40af1efbc)

# 11차시 10강 앙상블

## Ensemble learning
여러 개의 학습 알고리즘을 사용하여 더 좋은 예측 성능을 얻는 방법. 다수의 의견이 하나의 의견보다 더 좋은 결정을 내릴 수 있다는 집단지성의 원리를 이용

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/5f738b8a-2984-4561-b8b1-3256c06d7512)

## Ensemble 기법 종류
- Voting : 서로 다른 알고리즘 model을 조합해서 사용하여 도출해 낸 결과물에 대하여 최종 투표.Hard(결과물에 대한 최종 값을 투표로 결정)/Soft(최종 결과물이 나올 확률 값을 다 더해 최종 결과물에 대한 확률을 구한 뒤 도출) vote로 나눔.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/a27c2348-e11a-481a-a40d-e684ecdedb00)

- Boosting : 약한 학습기를 순차적으로 학습시키면서, 잘못 예측한 데이터에 가중치를 높여 다음 모델이 더 잘 예측하도록 함. ex)Gradient Boosting, XGBoost, LigjtGBM

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/0f5388da-de8e-4394-bb11-1489cbf61d39)

- Stacking : 여러 다른 모델들의 예측 결과를 새로운 모델의 입력으로 사용하여 최종 예측

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/a9b16368-2b83-4b4b-9066-f93cdf34cf86)

- Bagging : 동일한 알고리즘에 대해 데이터의 서브셋을 다르게 하여 여러 모델을 학습시키고, 그 결과를 투표/평균 내어 결정. ex) 랜덤 포레스트

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/ae4e284c-b2ec-46f3-b3e4-d971b93f95be)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/4a641a6d-3faf-4790-9257-74087c4b7e62)
![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/b53db6ca-f9b2-4569-89c0-855bff2e4875)


# 단어/용어-개념 정리
 **1강**
## 피처(feature)
개별 독립 변수를 의미. 데이터의 '특징'이라고 볼 수 있다. ex)집값 예측 모델 中 방의 개수, 집의 크기, 위치 등 과 같은 정보를 피처라고 한다.
## 데이터 인스턴스
데이터 세트의 개별 항목을 의미. ex)테이블 형식의 데이터 中 한 행을 하나의 데이터 인스턴스라고 본다.
- 피처와 데이터 인스턴스의 차이
데이터 인스턴스(전체 데이터 셋에서 하나의 행)는 개별 관찰 사례를 , 피처(그 행의 각 열)는 그 관찰 사례의 특. ex) 성적 기록부 中 성적정보는 데이터인스턴스, 이름, 수학성적 영어성적 등의 항목이 피처. 
## 컬럼
수직으로 배열된 데이터의 집합. 다른 말로는 '열'. 주로 데이터의 특정 피처나 속성(attribute)을 나타내며, 각 컬럼은 동일한 유형의 데이터(ex-숫자, 문자열)을 포함.ex)성적기록부 中 학생이름, 수학점수, 영어 점수 등 각각의 항목은 하나의 컬럼
## 집계(Aggregation)
여러 데이터를 합쳐서 그룹별로 요약된 정보를 제공하는 과정. 종류)평균, 합계, 최대값, 최소값, 카운트
## 시계열(Time Series)
시간 순서에 따라 측정된 데이터의 연속된 순서.시간의 흐름에 따른 패턴이나 추세 등을 분석하여 미래의 값을 예측하는데 활용. ex)일일 주가, 월별 판매량, 연간 기온 변화
## 학습(train)
학습 데이터 세트는 모델이 학습하는데 사용되는 데이터로, 이 데이터를 바탕으로 모델이 패턴을 학습하고, 예측 방법을 결정한다.
## 검증(Validation)
모델의 성능을 평가하기 위한 데이터로, 모델의 하이퍼파라미터(머신러닝 모델의 학습 과정을 제어하는 매개변수)를 조정하거나 overfiting을 방지한다.
## 테스트(Test)
최종 성능을 평가하는데 사용되는 데이터로, 모델이 실제 알 수 없는 데이터에 대해 어떻게 예측하는지 평가하는데 사용.
## 베이스 라인 모델
머신러닝 프로젝트에서 초기 참조점 또는 최소한의 성능 기준을 설정하는 모델.
- 성능 비교 : 복잡한 모델의 성능이 더 나은지 판단하기 위한 기준점
- 문제 이해 : 간단한 모델을 통해 문제의 복잡성 및 특성을 파악하는데 도움을 줌.
## 선형회귀(Linear Regression)
종속변수 y와 한 개 이상의 독립변수 x와의 선형상관관계를 모델링하는 회귀분석 기법. 데이터포인트들 사이의 관계를 잘 설명하는 직선(고차원에서는 평면)을 찾는 것을 목표.

![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/001983cd-0f7c-4ccb-959b-0d9578a72521)

## 결정 트리(Decision Tree)
데이터를 분류하거나 회귀 예측을 수행하는 간단하지만 강력한 머신러닝 알고리즘. 특정 피처가 어떻게 예측에 영향을 미치는지 직관적으로 파악 가능.ex)설명가능한 AI 구현
## 레이블 생성 함수
머신러닝에서 지도 학습을 수행하기 위해 필요한 타겟 변수, 레이블을 생성하는 함수를 의미.ex)스팸 메일 필터링모델학습中 각 이메일이 스팸인지 아닌지 나타내는 레이블이 필요.

 **2강**
## 데이터 전처리(Data processing)
데이터 분석이나 모델 학습에 앞서 데이터를 적절하게 가공하는 과정
## description
데이터셋의 특성이나 함수, 클래스 등의 동작과 속성.ex)데이터셋의 description은 해당 데이터셋의 전반적인정보, 각 변수의 의미, 데이터의 출처 등을 포함
## Summary
데이터의 기술통계량.ex)평균, 중앙값 최빈값, 최소값, 최대값, 범위, 표준편차, 1사분위수, 3사분위수 등 데이터 중심 경향, 분포, 변동성 등을 요약하여 나타낸 통계량

 **3강**
 ## 자연어
 사람들이 일상 생활에서 사용하는 언어.

  **4강**
  ## 사분위수(Quartile)
  데이터를 크기 순서대로 나열했을 때, 4등분 하는 지점. **데이터의 분와 이상치를 파악하는데 사용** **데이터의 중심경향, 퍼짐정도, 비대칭도 파악에 유용.** 박스플롯(Box plot)을 그릴때 사용
  - 제1,2,3사분위수(Q1,Q2,Q3) : 전체 데이터를 오름차순으로 정렬했을 때,각각 하위 25%값, 50%값, 75%값

 **7강**
 ## weight
 특정 피처가 트리 구조 내에서 얼마나 많이 사용되었는지 나타냄. 각 피처가 트리의 분기에 얼마나 많이 참여했는지를 계산하여 피처 중요도를 도출
 
 ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/98f770d6-8849-4d0a-8e1d-4f46826f92b9)
 ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/1f59e435-8f7f-427a-b882-377a7dcd1298)
 ![image](https://github.com/Oh-JunTaek/N_boost/assets/143782929/6c022c0e-a770-4e94-9e48-0f371b3b9259)


