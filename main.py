import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import logging
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 로그 출력 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. ratings.dat 파일을 기반으로 User x Item 행렬 생성
def create_user_item_matrix(file_path):
    
    logging.info("Loading and processing the ratings data.")
    start_time = time.time()

    # 데이터 로딩
    data = pd.read_csv(file_path, sep='::', header=None, engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    # user-item NP array 생성
    # UserID: 1~6040, MovieID: 1~3952
    # 평점을 매기지 않은 경우 0으로 채움
    user_item_matrix = np.zeros((6040, 3952)) 
    
    #ndarray에 데이터 대입
    #matrix[userid][movieid] = rating
    for row in data.itertuples():
        user_item_matrix[row[1]-1, row[2]-1] = row[3]
    
    end_time = time.time()
    logging.info(f"User-item matrix created. Time taken: {end_time - start_time:.2f} seconds.")
    
    
    return user_item_matrix

# 2. KMeans 클러스터링을 통해 유저를 3개의 그룹으로 나눔
# User는 3952개의 Item에 대한 평점을 매김
# 각 User는 3952차원의 벡터로 표현됨
# KMeans 클러스터링을 통해 User를 3개의 그룹으로 나눔
def perform_kmeans_clustering(user_item_matrix, n_clusters=3):
    logging.info("Starting KMeans clustering.")
    start_time = time.time()

    #scaler = StandardScaler()
    #user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)
    
    # KMeans 클러스터링
    # 3개의 클러스터로 수행
    # 6040명의 사용자를 3개의 그룹으로 군집화
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    #clusters = kmeans.fit_predict(user_item_matrix_scaled) - scaling을 시도해봤지만 결과가 좋지 않아서 scaling을 하지 않음
    clusters_without_scaling = kmeans.fit_predict(user_item_matrix)

    end_time = time.time()
    logging.info(f"KMeans clustering completed. Time taken: {end_time - start_time:.2f} seconds.")
    return clusters_without_scaling

# 3. 클러스터링 결과를 시각화
# 이 부분은 시각화를 위해 별도로 TSNE 차원 축소 기법에 대해 조사를 진행했습니다.
def visualize_clusters(user_item_matrix, clusters):
    logging.info("Starting cluster visualization.")
    start_time = time.time()

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(user_item_matrix)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter, ticks=range(max(clusters) + 1))
    plt.title("t-SNE visualization of Clusters")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")

    end_time = time.time()
    logging.info(f"Cluster visualization completed. Time taken: {end_time - start_time:.2f} seconds.")
    plt.show()

# 4. 집계 기법을 사용하여 그룹별 상위 10개 상품을 추천
# 각 클러스터(3 군집), 각 집계 기법(6가지)에 따른 상위 10개의 추천 영화를 반환 (3x6 = 18 개의 len=10짜리 list)
def recommend_top_items(user_item_matrix, clusters, n_clusters=3, top_n=10):
    
    logging.info("Starting recommendation process.")
    start_time = time.time()
    
    # 각 클러스터별 추천 결과를 저장할 딕셔너리 생성
    recommendations = defaultdict(dict)
    #aggregation_methods = ['average', 'additive_utilitarian', 'simple_count', 'approval_voting', 'borda_count', 'copeland_rule']
    
    
    for cluster in range(n_clusters):
        
        start_time_for_each_cluster = time.time()
        
        print(f"\n============ Processing recommendations for cluster {cluster} ============")
        
        cluster_data = user_item_matrix[clusters == cluster]
        
        # 각 클러스터에 속한 사용자 수 출력
        print('number of users in cluster: ', len(cluster_data))
        
        # Average - 해당 클러스터의 사용자가 평가한 평균 평점 기반 top 10
        print("Calculating average scores...")
        avg_scores = np.mean(cluster_data, axis=0)
        avg_top_items = np.argsort(avg_scores)[::-1][:top_n]
        recommendations[cluster]['average'] = avg_top_items
        
        # Additive Utilitarian - 해당 클러스터의 사용자가 평가한 평점의 합 기반 top 10
        print("Calculating additive utilitarian scores...")
        add_util_scores = np.sum(cluster_data, axis=0)
        add_util_top_items = np.argsort(add_util_scores)[::-1][:top_n]
        recommendations[cluster]['additive_utilitarian'] = add_util_top_items
        
        # Simple Count - 해당 클러스터의 중 실제로 평가한 사용자의 수 기반 top 10
        print("Calculating simple count scores...")
        simple_count_scores = np.count_nonzero(cluster_data, axis=0) # 0이 아닌 값의 개수를 세어줌
        simple_count_top_items = np.argsort(simple_count_scores)[::-1][:top_n]
        recommendations[cluster]['simple_count'] = simple_count_top_items
        
        # Approval Voting - 해당 클러스터에서 4점 이상으로 평가한 사용자의 수 기반 top 10
        print("Calculating approval voting scores...")
        approval_voting_scores = np.sum(cluster_data >= 4, axis=0)
        approval_voting_top_items = np.argsort(approval_voting_scores)[::-1][:top_n]
        recommendations[cluster]['approval_voting'] = approval_voting_top_items
        
        # Borda Count - 각 영화에 대해 사용자들의 평점 순위를 매겨 누적 점수를 계산하여 top 10 선정
        print("Calculating Borda count scores...")
        borda_scores = np.zeros(cluster_data.shape[1])
        for user_ratings in cluster_data:
            ranked_indices = np.argsort(user_ratings)[::-1]
            for rank, idx in enumerate(ranked_indices):
                borda_scores[idx] += rank
        borda_top_items = np.argsort(borda_scores)[::-1][:top_n]
        recommendations[cluster]['borda_count'] = borda_top_items
        
        # Copeland Rule - 해당 영화에 대해 다른 영화와 비교하여 승리한 횟수 - 패배한 횟수를 계산하여 top 10 선정
        # 처음엔 그냥 double 반복문 썼는데 너무 오래 걸림 - vectorization 사용으로 수정!
        print("Calculating Copeland rule scores...")
        copeland_scores = np.zeros(cluster_data.shape[1])
        for i in range(cluster_data.shape[1]):
            comparisons = cluster_data[:, i][:, np.newaxis] > cluster_data
            wins = np.sum(comparisons, axis=0)
            losses = np.sum(~comparisons, axis=0) - 1  # 자기 자신과의 비교를 제외
            copeland_scores[i] = np.sum(wins - losses)
        
        copeland_top_items = np.argsort(copeland_scores)[::-1][:top_n]
        recommendations[cluster]['copeland_rule'] = copeland_top_items
        
        end_time_for_each_cluster = time.time()
        print(f"Recommendations for cluster {cluster} completed. Time taken: {end_time_for_each_cluster - start_time_for_each_cluster:.2f} seconds.\n")
    
    end_time = time.time()
    logging.info(f"Recommendation process completed. Total Time taken: {end_time - start_time:.2f} seconds.")
    return recommendations



# 4. 추천 결과를 DataFrame으로 변환
# DataFrame은 Method x Cluster의 shape을 띄고 있으며 각 데이터는 상위 10개 추천 영화 list임.
def recommendations_to_dataframe(recommendations):
    records = []
    for method in recommendations[0].keys():
        method_records = {'Method': method}
        for cluster in recommendations.keys():
            method_records[f'Cluster {cluster}'] = list(recommendations[cluster][method])
        records.append(method_records)
    
    df = pd.DataFrame(records)
    return df

# 메인 함수
def main():
    
    # 파일 경로 지정
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'ml-1m', 'ratings.dat')
    
    # User-item matrix 생성
    user_item_matrix = create_user_item_matrix(file_path)
    
    # 각 유저는 3952차원의 벡터로 표현됨(미평가 시 0)
    # KMeans 클러스터링을 통해 유저를 3개의 그룹으로 나눔
    clusters = perform_kmeans_clustering(user_item_matrix)
    
    # 클러스터링 결과 시각화
    visualize_clusters(user_item_matrix, clusters)
    
    # 각 군집, 각 집계 방식 별 상위 10개 추천 영화
    recommendations = recommend_top_items(user_item_matrix, clusters)
    
    # DataFrame으로 변환 및 출력
    print('\n============ 기법 별 추천 결과 DataFrame ============\n')
    recommendations_df = recommendations_to_dataframe(recommendations)
    print(recommendations_df, end='\n\n')
    
    #출력 1 - 기법 별 각 군집에 대한 추천 결과
    print('\n============ 출력1 - 기법 별 추천 결과 ============')
    for Method in recommendations_df['Method']:
        print(f"Method: {Method}")
        for cluster in range(3):
            print(f"  Cluster {cluster}: {recommendations_df.iloc[0][f'Cluster {cluster}']}")
    
    #출력 2 - 군집 별 각 집계 기법에 대한 추천 결과
    print('\n============ 출력2 - 클러스터 별 추천 결과 ============')
    for cluster, methods in recommendations.items():
        print(f"Cluster {cluster} Recommendations:")
        for method, items in methods.items():
            strOut = '%21s' % method + ': '
            print(strOut, items)
            #print(f"  {method}: {items}")

if __name__ == "__main__":
    main()
