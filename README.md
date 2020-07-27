# KAKAO-Arena3

# 알고리즘 설명
## 곡 추천
사용된 추천 알고리즘은 최근 연구에서 여러 딥러닝 모델들을 압도하는 성능을 보이는 것으로 알려진, K-NN 입니다 [1].
구체적으로, 두 플레이리스트의 similarity는 아래와 같이 계산됩니다.

$s_ij = \frac{R_i \cdot R_j}{||R_i||||R_j|| + S}$

위 식에 따라, Implicit CF rating matrix (playlist num x song num)의 경우, playlist i, j의 row vecotor R_i, R_j의 cosine similarity로 두 playlist의 similarity를 계산하며,
이후, K nearest playlist에 담겨있는 song 정보를 종합하여, 추천에 사용합니다.
추가적으로, 기타 feature를 기반으로한 re-ranking 등의 후처리 기법이 사용되었습니다.

CF 정보 외에, tag/title, artist, album 정보가 추가로 similarity 계산에 사용되었으며,
각 feature의 다양한 조합으로 만들어진 추천 결과를 종합하여 앙상블을 수행한 것이 최종 제출된 결과에 해당합니다.

앙상블에 사용된 feature 조합 및 validation 성능은 아래와 같습니다.


|                  | feature                        | 평가              |  
|:--- | ---: | :---: |  
| 1             | CF only            | 0.3088 |  
| 2           | CF + tag/title            | 0.3094 |
| 3           | CF + artist            | 0.3080 |
| 4           | CF + artist + tag/title           | 0.3085 |
| 5           | CF + album          | 0.3092 |
| 6           | CF + album + tag/title         | 0.3095 |
| 7           | CF + album + artist         | 0.3075 |
| 8           | CF + tag/title2 (전처리 차이)         | 0.3092 |
| 앙상블 결과           | 1 - 8 종합        | 0.3124 |

앙상블에는 최근 연구에서 좋은 결과를 보인 ranking importance scheme (exp(-ranking/T)) 을 사용하였습니다.

## tag 추천
tag 추천 역시 곡 추천과 동일한 K-NN 메소드를 사용합니다.

# 코드 설명
## 1. 전처리


## 2. 곡 추천
song_inference.py 를 실행하면 됩니다. 곡 추천 결과 파일은 ensemble 폴더에 ensemble_A, ensemble_CD 파일입니다.

구체적으로, 위에 서술된 8개의 feature조합에 대한 예측 수행 후, 앙상블한 결과를 저장합니다.
한 파일에서 앙상블을 돌리기 위해서, feature 조합들에 대한 결과를 메모리에 올린채 반복문을 수행합니다.
저희가 사용한 머신은 메모리가 상당히 커서, 별 지장이 없었으나, 만에하나 가용 메모리 초과 문제가 있을 경우, 각각 저장된 결과를 앙상블에 사용하면 됩니다.
각각 저장된 결과는 ensemble 폴더에 들어있습니다 (song_x).

## 3. tag 추천


## 4. 결과 종합
앙상블 된 song/tag 결과를 종합하여 최종 파일을 만듭니다.


### 코드 실행 관련 문제가 있을 경우 seongku@postech.ac.kr 로 연락주시면 감사하겠습니다.

[1] Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches, RecSys 19
