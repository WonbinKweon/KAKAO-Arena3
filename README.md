# KAKAO-Arena3

## 곡 추천
사용된 추천 알고리즘은 최근 연구에서 여러 딥러닝 모델들을 압도하는 성능을 보이는 것으로 알려진, K-NN 입니다 [1].
구체적으로, 두 플레이리스트의 similarity는 아래와 같이 계산됩니다.

$$s_ij = \frac{R_i \cdot R_j}{||R_i||||R_j|| + S}$$

위 식에 따라, Implicit CF rating matrix (playlist num x song num)의 경우, playlist i, j의 row vecotor R_i, R_j의 cosine similarity로 두 playlist의 similarity를 계산하며,
이후, K nearest playlist에 담겨있는 song 정보를 종합하여, 최종 추천에 반영합니다.

CF 정보 외에, tag/title, artist, album 정보가 추가로 similarity 계산에 사용되었으며,
각 feature의 다양한 조합으로 만들어진 추천 결과를 종합하여 앙상블을 수행한 것이 최종 제출된 결과에 해당합니다.

앙상블에 사용된 feature 조합 및 validation 성능은 아래와 같습니다.





[1] Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches, RecSys 19
