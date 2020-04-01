from metascrapper.index import IndexScrapper
import sensitive

user_agent = "Metascrapper ({email})".format(email=sensitive.EMAIL)
index_scrapper = IndexScrapper(user_agent=user_agent)

index_scrapper.run()
index_scrapper.to_csv("../data/reviews/game_index.csv")
