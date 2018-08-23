## -- 예제: SVM으로 OCR 수행

# 1. 데이터 탐색 및 준비
letters <- read.csv('https://www.dropbox.com/s/1sxvff8al4a1nwe/letterdata.csv?dl=1')
str(letters) # 20000 x 17 

# cf 1) SVM 학습자 => 모든 특징이 수치형이어야 함 + 각 특징이 아주 작은 구간으로 값 조정되어야 함.
# cf 2) 필요한 경우 데이터 표준화 필요

letters_train <- letters[1:16000, ]
letters_test <- letters[16001:20000, ]

# 2. 데이터에 대한 모델 훈련
# install.packages('kernlab')
library(kernlab)

# 분류기 구축
letter_classifier <- ksvm(letter ~., data = letters_train,
                          kernel = 'vanilladot')

# 훈련 파라미터와 모델 적합성에 대한 기본정보 확인
letter_classifier

# 3. 모델 성능 평가
# 문자 분류 모델로 테스트 데이터셋 예측
letter_predictions <- predict(letter_classifier, letters_test)

# 분류 정확도 확인
table(letter_predictions, letters_test$letter)
agreement <- letter_predictions == letters_test$letter

# 문자를 정확히 식별해냈는지 여부 확인 
table(agreement)

# 비율로 계산
prop.table(table(agreement))

# 4. 모델 성능 향상
# 가우시안 RBF 기반 SVM 훈련

# seed 고정
set.seed(12345)
letter_classifier_rbf <- ksvm(letter ~ ., data = letters_train,
                              kernel = 'rbfdot')

# 테스트셋 데이터 예측
letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test)

# 결과 확인
agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf)
prop.table(table(agreement_rbf)) # 93%까지 상승

# cf) 성능 수준 향상 방안 1) 다른 커널 테스트, 2) 제약의 비용 파라미터 C 변화

