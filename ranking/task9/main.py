import time

from scrapy.crawler import CrawlerProcess

from ranking.task9.spider import ActorSpider, MovieSpider


def parse_actors():
    cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    settings = {
            "FEEDS": {
                f"actors_{cur_time}.jsonl": {"format": "jsonlines"},
            }
        }
    process = CrawlerProcess(settings=settings)

    process.crawl(ActorSpider)
    process.start()


def parse_movies():
    cur_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    settings = {
        "FEEDS": {
            f"movies_{cur_time}.jsonl": {"format": "jsonlines"},
        }
    }
    process = CrawlerProcess(settings=settings)

    process.crawl(MovieSpider)
    process.start()


if __name__ == '__main__':
    parse_actors()
    parse_movies()
