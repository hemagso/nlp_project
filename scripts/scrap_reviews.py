from metascrapper.review import UserReviewScrapper

index_path = "../data/reviews/game_index.csv"
scrapper = UserReviewScrapper()
scrapper.run(index_path)
scrapper.to_csv("../data/reviews/reviews.csv")