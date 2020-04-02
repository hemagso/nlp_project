import requests
import re
import csv

from bs4 import BeautifulSoup
from datetime import datetime
from time import sleep

from .utils import safe_cast
from .exceptions import RequestException


class IndexScrapper(object):
    """ This class scraps the index page of Metacritic game section, retrieving a list of games, their
    scores and release dates.
    """
    def __init__(self, user_agent="Metascraper"):
        """ Create an instance of the IndexScraper

        :param user_agent: User agent that will be sent with the request.
        """
        self.index_root = "http://www.metacritic.com/browse/games/score/metascore/all/all/filtered"
        self.user_agent = user_agent
        self.page_count = self._get_page_count()
        self.last_scrap = None
        self.data = []

    def run(self, start_page=0, end_page=None, sleep_time=2, retry_time=60):
        """ Run the scrapping process

        :param start_page: Which page of the index we should start (Default: First page).
        :param end_page: Which page of the index we should stop (Default: Last page)
        :param sleep_time: How much time, in seconds, should we wait between requests (Default: 2 seconds)

        """
        end_page = self.page_count-1 if end_page is None else end_page
        for page in range(start_page, end_page+1):
            while True:  # TODO: Add a maximum number of attempts
                try:
                    print("Retrieving page {current} ({start_page} to {end_page})".format(
                        current=page,
                        start_page=start_page,
                        end_page=end_page
                    ))
                    html = self._get_index_page(page)
                    self.data += IndexScrapper._parse_page(html, page)
                    sleep(sleep_time)
                    break
                except Exception as e:  # TODO: Make this exception clause more specific
                    print(e)
                    print("Exception raised when processing page {page}".format(page=page))
                    print("Retrying in {retry_time} seconds".format(retry_time=retry_time))
                    sleep(retry_time)

    def to_csv(self, filepath):
        """ Save the results of the scrapping to a CSV file.

        :param filepath: Path to the file where we will save the data
        """
        with open(filepath, "w", encoding="utf-8") as f:
            writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
            writer.writerow(("page_number", "title", "url", "metascore", "userscore", "publish_date"))
            writer.writerows(self.data)

    def _get_page_count(self):
        """ Retrieve the total page count of the index.

        :return: Integer representing the total page count.
        """
        html = self._get_index_page(0)
        last_page_dom = html.find("li", attrs={"class": "page last_page"}).find("a")
        return int(last_page_dom.text)

    def _get_index_page(self, page_number):
        """ Retrieves an specific index page.

        :param page_number: The number of the page we want to retrieve.
        :return: BeautifulSoup document of the retrieved page.
        """
        r = requests.get(self.index_root, params={"page": page_number}, headers={"User-Agent":  self.user_agent})
        if r.status_code != 200:
            raise RequestException("Request returned abnormal status code {sc}".format(sc=r.status_code))
        else:
            return BeautifulSoup(r.text, features="html.parser")

    def __repr__(self):
        return "IndexScrapper with {n} games".format(n=len(self.data))

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def _parse_page(html, page):
        """ Parse a page, returning all game items in it.

        :param html: BeautifulSoup object of the webpage
        :param page: Number of the page (Used only as a field in the dataset generated)
        :return: List with all game items (Represented by tuples).
        """
        rows = html.find("ol", attrs={"class": "list_products"}).find_all("div", attrs={"class": "product_wrap"})
        return [(page,) + IndexScrapper._parse_game(row) for row in rows]

    @staticmethod
    def _parse_game_title(row):
        """ Retrieve the game title from a game item.

        :param row: DOM element representing a game item.
        :return: Tuple (title, url) where title is a string of the game title and url is a string of the
            URL (relative to metacritic top domain) of the details page for the game.
        """
        title_dom = row.find("div", attrs={"class": "product_title"}).find("a", href=True)
        title = title_dom.text
        title = re.sub('\s+', ' ', title).strip()
        url = title_dom["href"]
        return title, url

    @staticmethod
    def _parse_game_score(row):
        """ Retrieve the game title from a game item.

        :param row: DOM element representing a game item.
        :return: Tuple (metascore, userscore) containing the scores for the game.
        """
        metascore_dom = row.find("div", attrs={"class": "metascore_w"})
        metascore = safe_cast(metascore_dom.text, int)
        userscore_dom = row.find("li", attrs={"class": "product_avguserscore"}).find(
            "span", attrs={"class": "textscore"}
        )
        userscore = safe_cast(userscore_dom.text, float)
        return metascore, userscore

    @staticmethod
    def _parse_game_publish_date(row):
        """ Retrieve the game title from a game item.

        :param row: DOM element representing a game item.
        :return: Datetime object containing the publish date for the game
        """
        publish_date_dom = row.find("li", attrs={"class": "release_date"}).find("span", attrs={"class": "data"})
        publish_date = publish_date_dom.text
        publish_date = re.sub(r'\s+', ' ', publish_date).strip()
        publish_date = datetime.strptime(publish_date, "%b %d, %Y").date()
        return publish_date

    @staticmethod
    def _parse_game(row):
        """ Retrieve the game title from a game item.

        :param row: DOM element representing a game item.
        :return: Tuple (title, url, metascore, userscore, publish_date). See other parse methods for details on
            those
        """
        title, url = IndexScrapper._parse_game_title(row)
        metascore, userscore = IndexScrapper._parse_game_score(row)
        publish_date = IndexScrapper._parse_game_publish_date(row)
        return title, url, metascore, userscore, publish_date
