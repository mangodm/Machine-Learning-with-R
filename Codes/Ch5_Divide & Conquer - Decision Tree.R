## -- 예제: C5.0 의사결정 트리를 활용한 위험 은행 대출 식별

# 0. 데이터셋 설명

# 1. 데이터 탐색 및 준비

credit <- read.csv('https://www.dropbox.com/s/w43k5qii8yf9els/credit.csv?dl=1')
str(credit) # 1000 obs. of 17 variables:

table(credit$default) # no: 700 / yes: 300 | 대출의 전체 30% => default(채무불이행)

# 2. 데이터 준비: 훈련 및 테스트 데이터셋 생성

# 데이터 분할: 900개 / 100개 (랜덤 샘플링 활용)
set.seed(123)
train_sample <- sample(1000, 900) # 1000개 중 정수 벡터 900개 샘플링
str(train_sample)

credit_train <- credit[train_sample, ]
credit_test <- credit[-train_sample, ]

prop.table(table(credit_train$default)) # 70.3 vs. 29.7
prop.table(table(credit_test$default)) # 67 vs. 33

# 3. 데이터로 모델 훈련

# install.packages('C50')
library(C50) # C5.0 알고리즘을 구현해주는 패키지

# 모델 생성 
credit_model <- C5.0(credit_train[-17], credit_train$default) # default를 target factor/vector로 제공 
credit_model # 트리의 기본 데이터 확인

# 트리 결정 확인
summary(credit_model)

# cf) 처음 세 줄에 대한 해석
# 1) 수표 계좌 잔고를 모르거나 200DM 이상이면 'no'로 분류
# 2) 그렇지 않으면 수표 계좌 잔고가 0보다 작거나 1에서 200DM 이하인 경우
# 3) 대출 이력이 완벽이나 매우 좋음이면 'yes'로 분류
# -- 괄호 안의 숫자: 결정 기준에 부합하는 예시 개수와 결정으로 부정확하게 분류된 숫자
# -- 논리에 반하는 결정: a) 데이터의 실제 패턴 OR b) 통계적 이상치의 영향 

# 4. 모델 성능 평가

credit_pred <- predict(credit_model, credit_test)
library(gmodels) # 교차표 생성을 위한 패키지 

CrossTable(x = credit_test$default,
           y = credit_pred,
           prop.chisq = FALSE,
           propr.c = FALSE,
           prop.r = FALSE,
           dnn = c('actual default', 'predicted default')) # 정확도: 73% / 오류율: 27%

# 5. 모델 성능 개선 

# 1) 부스팅 활용
credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10)
credit_boost10 # average tree size 작아짐: 57 => 47.5
summary(credit_boost10)

# 테스트 데이터 적합 
credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('actual default', 'predicted default')) # 오류율: 18%

# cf) 부스팅이 쉽다면(?) 혹은 성능을 개선한다면(?) 왜 디폴트 옵션으로 사용하지 않는가?
# 1) 의사결정 트리를 한번 만드는 데 시간이 많이 걸린다면 여러 개의 트리 생성 => 비현실적
# 2) 훈련 데이터에 노이즈가 많으면 부스팅으로 개선 x

# 2) 비용 행렬(cost matrix) 활용

# 비용 행렬 생성
mat.dim <- list(c('no', 'yes'), c('no', 'yes'))
names(mat.dim) <- c('predicted', 'actual')
mat.dim

# 오류 유형별 페널티 부여 
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2,
                     dimnames = mat.dim)
error_cost 
# cf) 비용 행렬에 대한 설명
# 1) 분류기가 no 또는 yes를 정확히 분류하면 비용 할당 x
# 2) 거짓 부정: 비용 4 / 거짓 긍정: 비용 1

# 모형 적합
credit_cost <- C5.0(credit_train[-17], credit_train$default,
                    costs = error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)

CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

26/33 # 실제 default의 79%를 default로 예측