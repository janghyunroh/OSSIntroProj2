import os
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import defaultdict

from config import CONFIG
from loader import load_user_data, load_movie_data
from kmeans import create_user_item_matrix, perform_kmeans_clustering, visualize_clusters

# Streamlit 설정
st.set_page_config(page_title="영화 추천 시스템", layout="wide")



# 5. 추천 아이템 함수 (영화 제목 매핑)
def recommend_top_items_with_titles(user_item_matrix, clusters, movie_dict, n_clusters=3, top_n=10):
    recommendations = defaultdict(list)
    for cluster in range(n_clusters):
        cluster_data = user_item_matrix[clusters == cluster]
        avg_scores = np.mean(cluster_data, axis=0)
        top_items = np.argsort(avg_scores)[::-1][:top_n]
        for movie_id in top_items:
            movie_id += 1  # 영화 ID는 0부터 시작하므로 +1
            if movie_id in movie_dict:
                title = movie_dict[movie_id]["Title"]
                genres = movie_dict[movie_id]["Genres"]
                recommendations[cluster].append((title, genres))
    return recommendations

def get_recommendations(model, user_data, selected_user_id, n_recommendations=10):
    # 선택된 사용자의 데이터 추출
    selected_user_data = user_data[user_data['UserID'] == selected_user_id].drop('UserID', axis=1).values
    # 이웃 찾기
    distances, indices = model.kneighbors(selected_user_data, n_neighbors=n_recommendations + 1)
    # 추천 영화 ID 반환 (본인을 제외한 가장 가까운 이웃)
    return indices[0][1:]

# 태그 형식으로 추천 영화 출력
def display_recommendations_as_tags(recommendations, movie_dict):
    st.markdown("<h3>추천 영화 목록</h3>", unsafe_allow_html=True)
    for movie_id in recommendations:
        if movie_id in movie_dict:
            title = movie_dict[movie_id]["Title"]
            genres = movie_dict[movie_id]["Genres"]
            # HTML 태그 스타일 적용
            st.markdown(
                f"""
                <p style="font-size:16px;">
                    🎬 <strong>{title}</strong>
                    <span style="background-color:#f0f0f0; color:#333; padding:3px 8px; border-radius:5px; font-size:14px;">
                        {genres}
                    </span>
                </p>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"영화 ID {movie_id}에 대한 정보가 없습니다.")
# Streamlit 메인 앱
def main():
    st.title("영화 추천 시스템")

    # 파일 경로 정의
    user_file_path = "user_ratings.csv"  # 사용자 데이터 (예: UserID와 영화 평점 데이터)
    movies_file_path = "movies.dat"  # 영화 메타데이터

    # 데이터 로드
    user_data = load_user_data(CONFIG['user_path'])
    rating_data = load_user_data(CONFIG['rating_path'])
    movie_dict = load_movie_data(CONFIG['movie_path'])

    if os.path.exists(ratings_file_path) and os.path.exists(movies_file_path):
        st.success(f"`ratings.dat` 및 `movies.dat` 파일이 로드되었습니다!")

        # user-item matrix 생성
        # User x Item 행렬 생성
        with st.spinner("User-Item Matrix 생성 중..."):
            user_item_matrix = create_user_item_matrix(ratings_file_path)
            st.success("User-Item Matrix가 성공적으로 생성되었습니다!")


        # 클러스터링 수행
        n_clusters = st.slider("클러스터 개수 선택", min_value=2, max_value=10, value=3, step=1)
        if st.button("클러스터링 수행"):
            with st.spinner("KMeans 클러스터링 수행 중..."):
                clusters = perform_kmeans_clustering(user_item_matrix, n_clusters=n_clusters)
                st.success("클러스터링이 완료되었습니다!")
            
            # 클러스터링 결과 시각화
            st.subheader("🔍 클러스터링 시각화")
            st.write(f"선택한 클러스터 개수: {n_clusters}")
            fig = visualize_clusters(user_item_matrix, clusters)
            st.pyplot(fig)

            # 추천 결과 표시
            st.subheader("🎯 추천 결과")
            with st.spinner("추천 결과 생성 중..."):
                recommendations = recommend_top_items_with_titles(user_item_matrix, clusters, movie_dict, n_clusters=n_clusters)
                for cluster, movies in recommendations.items():
                    st.write(f"**클러스터 {cluster} 추천 영화**")
                    for title, genres in movies:
                        st.write(f"- {title} ({genres})")

        # 사용자 선택
        user_ids = user_data['UserID'].unique()
        selected_user_id = st.selectbox("사용자 선택", options=user_ids)



    # 추천 결과 얻기
    recommendations = get_recommendations(model, user_data, selected_user_id)

    # 추천 영화 출력
    display_recommendations_as_tags(recommendations, movie_dict)

if __name__ == "__main__":
    main()

# Streamlit 인터페이스
st.title("🎥 영화 추천 시스템")

# 고정된 파일 경로 설정
ratings_file_path = os.path.join(os.getcwd(), "ml-1m/ratings.dat")
movies_file_path = os.path.join(os.getcwd(), "ml-1m/movies.dat")

if os.path.exists(ratings_file_path) and os.path.exists(movies_file_path):
    st.success(f"`ratings.dat` 및 `movies.dat` 파일이 로드되었습니다!")
    
    # 영화 데이터 로드
    movie_dict = load_movie_data(movies_file_path)
    
    

    # 클러스터링 수행
    n_clusters = st.slider("클러스터 개수 선택", min_value=2, max_value=10, value=3, step=1)
    if st.button("클러스터링 수행"):
        with st.spinner("KMeans 클러스터링 수행 중..."):
            clusters = perform_kmeans_clustering(user_item_matrix, n_clusters=n_clusters)
            st.success("클러스터링이 완료되었습니다!")
        
        # 클러스터링 결과 시각화
        st.subheader("🔍 클러스터링 시각화")
        st.write(f"선택한 클러스터 개수: {n_clusters}")
        fig = visualize_clusters(user_item_matrix, clusters)
        st.pyplot(fig)

        # 추천 결과 표시
        st.subheader("🎯 추천 결과")
        with st.spinner("추천 결과 생성 중..."):
            recommendations = recommend_top_items_with_titles(user_item_matrix, clusters, movie_dict, n_clusters=n_clusters)
            for cluster, movies in recommendations.items():
                st.write(f"**클러스터 {cluster} 추천 영화**")
                for title, genres in movies:
                    st.write(f"- {title} ({genres})")
else:
    st.error("`ratings.dat` 또는 `movies.dat` 파일이 현재 경로에 존재하지 않습니다. 파일을 추가하세요.")
