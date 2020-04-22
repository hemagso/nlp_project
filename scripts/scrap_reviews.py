from metascrapper.review import UserReviewScrapper
import os

index_path = "../data/reviews/game_index"
reviews_path = "../data/reviews/reviews"

for index_part_filename in os.listdir(index_path):
    slice_id = int(index_part_filename[15:17])
    reviews_part_path = os.path.join(reviews_path, "game_index_part{0:02d}.csv".format(slice_id))
    if os.path.exists(reviews_part_path):
        print("Skipping", slice_id)
    else:
        print("Retrieving part", slice_id)
        index_part_path = os.path.join(index_path, index_part_filename)
        print(index_part_path)
        scrapper = UserReviewScrapper()
        scrapper.run(index_part_path)
        scrapper.to_csv(reviews_part_path)