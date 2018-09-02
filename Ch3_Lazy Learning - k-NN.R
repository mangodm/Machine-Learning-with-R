## -- 예제: k-NN으로 유방암 진단

# 0. 데이터셋 설명

# 1. 데이터 탐색 및 준비

wbcd <- read.csv('https://www.dropbox.com/s/1wbnzleuiahnjkm/wisc_bc_data.csv?dl=1')
str(wbcd)

# 모델링에 불필요한 아이디 변수 제거
wbcd <- wbcd[-1]

# diagnosis 변환
wbcd$diagnosis <- factor(wbcd$diagnosis,
                         levels = c('B', 'M'),
                         labels = c('Benign', 'Malignant'))

# diagnosis별 빈도
table(wbcd$diagnosis)

# diagnosis별 상대빈도
prop.table(table(wbcd$diagnosis)) # 62.7% vs. 37.3% 

# 특징 값 정규화(normalization)

# min-max 정규화
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x))) # 괄호 주의!
}

# 정규화 정상 작동 여부 테스트
normalize(c(1, 2, 3, 4, 5))
normalize(c(10, 20, 30, 40, 50))

# lapply를 활용한 변수 정규화 자동화
wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))

summary(wbcd_n) # 변수들이 0 - 1 범위에 있음을 확인

# 2. 데이터 준비: 훈련 및 테스트 데이터셋 생성

# 데이터 분할: 469개 / 100개 
wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ] # cf) 데이터가 시간순 정렬 혹은 비슷한 값으로 묶인 경우 랜덤 샘플링 필요

# 클래스 레이블 저장
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

# install.packages('class')
library(class) # 분류를 위한 기본 R 함수 제공하는 패키지

# 3. 데이터로 모델 훈련

# k값 지정(469개의 제곱근(k = 21)으로 시도)
wbcd_test_pred <- knn(train = wbcd_train,
                      test = wbcd_test,
                      cl = wbcd_train_labels,
                      k = 21)

# 4. 모델 성능 평가

# install.packages('gmodels')
library(gmodels) # 교차표 생성을 위한 패키지 

CrossTable(x = wbcd_test_labels,
           y = wbcd_test_pred,
           prop.chisq = FALSE) # 종양 100개 중 2개가 잘못 분류

# cf) 예측: 거짓 긍정 비율과 거짓 부정 비율 간의 균형 맞추기
# cf) 어떤 오류가 domain에서 위험한 오류인 지를 판단하여 train 개선 방향 수립

# 5. 모델 성능 개선 

# 1) z-점수 표준화: 사전에 정의된 최솟값/최댓값이 없으므로 극값이 중심 방향으로 축소 x 
wbcd_z <- as.data.frame(scale(wbcd[-1])) # cf) scale 함수는 dataframe에 적용 가능

summary(wbcd_z)

# 4.에서와 같은 방법으로 적합 후 성능 평가
wbcd_train <- wbcd_z[1:469, ]
wbcd_test <- wbcd_z[470:569, ] 

# 클래스 레이블 저장
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

wbcd_test_pred <- knn(train = wbcd_train,
                      test = wbcd_test,
                      cl = wbcd_train_labels,
                      k = 21)

CrossTable(x = wbcd_test_labels,
           y = wbcd_test_pred,
           prop.chisq = FALSE) # 95%만 정확히 분류 

# 2) k 대체 값 테스트
