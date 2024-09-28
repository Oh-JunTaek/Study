[TODO] 코드 구현 : 경사하강법을 위한 데이터 분리
앞으로 우리는 numpy 를 이용해 정답값을 예측해보는 선형회귀(Linear Regression) 모델을 구현해보고자 합니다. 그 첫번째 단계로 데이터를 준비해보겠습니다. 아래와 같이 데이터가 주어져 있을 때, 경사하강법을 위한 데이터를 분리해보세요.
# 프로그램 실행 예시
# 출력 결과

# print(x_train, x_train.shape)
# [1. 2. 3. 4. 5. 6.] (6,)

# print(y_train, y_train.shape)
# [10. 20. 30. 40. 50. 60.] (6,)

    import numpy as np

xy = np.array([[1., 2., 3., 4., 5., 6.],
              [10., 20., 30., 40., 50., 60.]])

## 코드시작 ##

x_train = xy[0, :]    
y_train = xy[1, :]   


## 코드종료 ##

print(x_train, x_train.shape)
print(y_train, y_train.shape)


[TODO] 코드 구현 : train, weight, bias 정리
위에서 분리한 x_train 데이터와 계산될 weight, bias 값을 정의해보세요. 여기서 구현한 weight와 bias는 linear regression을 계산할 때, 직선의 기울기와 y 절편의 값이 됩니다.

numpy 내의 random 함수를 이용해보세요.

# 프로그램 실행 예시
# 출력 결과

# print(beta_gd, bias)
# [0.53764546] [0.71495179]    # randoma을 사용해 숫자를 만들어냈기 때문에, 출력결과는 다를 수 있습니다. 형태 위주로 확인해주세요.

## 코드시작 ##
beta_gd = np.random.rand(1)    
bias = np.random.rand(1)       



## 코드종료 ##

print(beta_gd, bias)

[TODO] 코드 구현 : 경사하강법 구현
이제 최종적으로 linear regression을 경사하강법으로 학습하는 코드를 구현해봅시다. 경사하강법 구현을 위한 학습 Loop를 만들어보세요. 그리고 100회 반복했을 때의 결과를 출력해보세요.

단, Error는 차이의 제곱을 이용해서 계산해주세요.
Gradient 값은 우리가 예측하는 각 변수에 대한 미분값입니다.

learning_rate = 0.001

## 코드시작 ##

epochs = 1000
for i in range(epochs):
    # 예측값 계산
    y_pred = beta_gd * x_train + bias

    # 에러 계산
    error = y_train - y_pred

    # Gradient 계산
    beta_grad = -(2/len(x_train)) * sum(x_train * error)
    bias_grad = -(2/len(x_train)) * sum(error)

    # weight와 bias 업데이트
    beta_gd = beta_gd - learning_rate * beta_grad
    bias = bias - learning_rate * bias_grad

    # 100회 반복마다 결과 출력
    if i % 100 == 0:
        cost = sum(error**2)
        print(f"Epoch ( {i}/{epochs}) cost : {cost}, w: {beta_gd[0]}, b:{bias[0]}")

## 코드종료 ##
