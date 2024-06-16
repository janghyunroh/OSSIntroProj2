# OSSIntroProj2
인하대학교 24-1학기 오픈소스 소프트웨어개론 프로젝트2 저장소

# 프로젝트 목표
주어진 데이터셋을 이용하여 사용자를 세 유형의 그룹으로 군집화한 뒤, 
각 유형의 그룹인 유저가 좋아할만한 영화를 10개를 추려 출력하는 것이 목표입니다.

어떤 영화를 좋아할지는 각 그룹 사용자들의 평점과
6가지 집계 기법을 이용하여 판단합니다.

각 그룹 별로 각 기법을 사용하여 추려낸 top 10 list 18개를 최종 출력하게 됩니다. 

# 데이터셋 
https://grouplens.org/datasets/movielens/1m/

# 사용법
이 저장소를 clone한 후 main.py 파일을 실행시키면 됩니다. 

# 알고리즘 구현

(자세한 코드 작성에 대한 설명은 주석에 포함되어 있습니다.)

1. 데이터 불러오기 및 행렬 생성

2. 행렬 기반 사용자 클러스터링

3. 클러스터링 결과 시각화 

4. 각 클러스터에 대해 6가지 집계 기법을 적용한 뒤 상위 10개의 영화 id 추출

5. 최종 출력


