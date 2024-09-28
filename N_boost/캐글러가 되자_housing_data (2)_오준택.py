# 미션 3-2 코드 작성 : data에 결손치를 missingno 라이브러리를 이용하여 시각화 해보세요.
import missingno as msno

msno.matrix(data)
plt.show()

# 미션 4 코드 작성 : 로그변환을 수행해보세요.

# 치우친 분포의 컬럼을 저장해 두기
skew_columns = ['bedrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_lot15', 'sqft_living15']

for c in skew_columns:
    data[c] = np.log1p(data[c])