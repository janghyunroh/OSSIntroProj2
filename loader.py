import pandas as pd

# 데이터 로드 함수
def load_user_data(file_path):
    user_data = pd.read_csv(
        file_path,
        sep="::",
        header=None,
        engine="python",
        names=["UserID","Gender|","Age","Occupation","Zip-code"]
    )
    return user_data

def load_rating_data(file_path):
    movie_data = pd.read_csv(
        file_path,
        sep="::",
        header=None,
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"]
    )
    return movie_data

def load_movie_data(file_path):
    # movies.dat 파일 읽기
    movie_data = pd.read_csv(
        file_path,
        sep="::",
        header=None,
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1"
    )
    # MovieID를 key로 한 딕셔너리 생성
    movie_dict = {
        row.MovieID: {"Title": row.Title, "Genres": row.Genres}
        for _, row in movie_data.iterrows()
    }
    return movie_dict