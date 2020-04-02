import requests
from requests.compat import urljoin
from bs4 import BeautifulSoup
from datetime import datetime
import re
import csv
from time import sleep
from .exceptions import RequestException

# TODO: Add docstrings

class UserReviewScrapper(object):
    def __init__(self, user_agent="Metascrapper"):
        self.reviews_root = "http://www.metacritic.com"
        self.user_agent = user_agent
        self.data = []

    def run(self, index_path, sleep_time=2, retry_time=60):
        url_list = UserReviewScrapper._read_index_file(index_path)
        for game_url in url_list:
            while True:  # TODO: Add a maximum number of attempts
                try:
                    self.data += self._get_game_reviews(game_url, sleep_time=sleep_time)
                    break
                except Exception as e:  # TODO: Make this exception clause more specific
                    print(e)
                    print("Exception raised when processing game {game_url}".format(game_url=game_url))
                    print("Retrying in {retry_time} seconds".format(retry_time=retry_time))
                    sleep(retry_time)

    def to_csv(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
            writer.writerow(("user", "date", "userscore", "review"))
            writer.writerows(self.data)

    def _get_game_reviews(self, game_url, sleep_time=2):
        reviews = []
        page_count = self._get_page_count(game_url)
        print("Retrieving reviews for {game_url} - Total of {page_count} pages".format(game_url=game_url, page_count=page_count))
        for page in range(page_count):
            print("|- Page {page}/{page_count}".format(page=page+1, page_count=page_count))
            html = self._get_review_page(game_url, page)
            reviews += UserReviewScrapper._parse_page(html)
        return reviews

    @staticmethod
    def _read_index_file(index_path):
        url_list = []
        with open(index_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for _, _, url, _, _, _ in reader:
                url_list.append(url)
        return url_list

    def _build_full_url(self, game_url):
        full_url = urljoin(self.reviews_root, game_url)
        full_url = urljoin(full_url + "/", "user-reviews")
        return full_url

    def _get_page_count(self, game_url):
        html = self._get_review_page(game_url, 0)
        last_page_dom = html.find("li", attrs={"class": "page last_page"})
        if last_page_dom:
            last_page_dom = last_page_dom.find("a")
            return int(last_page_dom.text)
        else:
            return 1

    def _get_review_page(self, game_url, page_number):
        full_url = self._build_full_url(game_url)
        r = requests.get(
            full_url,
            params={"page": page_number, "num_items": 100},
            headers={"User-Agent":  self.user_agent}
        )

        if r.status_code != 200:
            raise RequestException("Request returned abnormal status code {sc}".format(sc=r.status_code))
        else:
            return BeautifulSoup(r.text, features="html.parser")

    @staticmethod
    def _parse_page(html):
        reviews = html.find("div", attrs={"id": "main"}).find_all("li", attrs={"class": ["review", "user_review"]})
        return [UserReviewScrapper._parse_review(review) for review in reviews]

    @staticmethod
    def _parse_review(review):
        text = UserReviewScrapper._parse_review_text(review)
        score = UserReviewScrapper._parse_review_score(review)
        user, date = UserReviewScrapper._parse_review_details(review)
        return user, date, score, text

    @staticmethod
    def _parse_review_text(review):
        body = review.find("div", attrs={"class": "review_body"})
        expanded = body.find("span", attrs={"class": "blurb blurb_expanded"})
        if expanded is not None:
            return expanded.text
        else:
            return body.text

    @staticmethod
    def _parse_review_score(review):
        score_dom = review.find("div", attrs={"class": "metascore_w"})
        return int(score_dom.text)

    @staticmethod
    def _parse_review_details(review):
        detail_dom = review.find("div", attrs={"class": "review_critic"})
        user_dom = detail_dom.find("div", attrs={"class": "name"})
        date_dom = detail_dom.find("div", attrs={"class": "date"})
        review_date = datetime.strptime(date_dom.text, "%b %d, %Y").date()
        user = user_dom.text
        user = re.sub(r'\s+', ' ', user).strip()
        return user, review_date
