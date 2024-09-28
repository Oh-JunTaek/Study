# 미션 1 코드 작성 : 캐글에 가입하고, 프로필명 공유하기
print('eunma3533')

# 미션 2 코드 작성 : data에서 price 컬럼을 완전히 삭제하기
data.drop('price', axis=1, inplace=True)

# 미션 3-1 코드 작성 : data에 isna와 sum을 적용하여 각 컬럼의 결측치 수를 확인해보세요.

missing = data.isna().sum()
print(missing)

missing_ratio = missing / len(data)
print(missing_ratio)