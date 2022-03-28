import scrapy


class MovieItem(scrapy.Item):
    title = scrapy.Field()
    url = scrapy.Field()
    cast = scrapy.Field()


class ActorItem(scrapy.Item):
    bio = scrapy.Field()
    born = scrapy.Field()
    movies = scrapy.Field()
    name = scrapy.Field()
    url = scrapy.Field()


class ActorSpider(scrapy.Spider):
    """Spider parse top-50 popular actors page. It gets name, bio, date of birth, url. Then it goes to actor url
     and parse movies of each actor.

    """

    name = "imdb_actor"
    allowed_domains = ["imdb.com"]
    base_url = 'https://www.imdb.com'
    start_urls = ['https://www.imdb.com/search/name/?gender=male%2Cfemale&ref_=nv_cel_m']

    def parse(self, response):
        table_rows = response.xpath('.//*[@class="lister-item-content"]')
        for row in table_rows:
            bio = row.xpath('p[not(@class)]/text()').extract_first().strip()
            name = row.xpath('h3[@class="lister-item-header"]/a/text()').extract_first().strip()
            url = row.xpath('h3[@class="lister-item-header"]/a/@href').extract_first().strip()
            row_url = self.base_url + url

            yield scrapy.Request(row_url, callback=self._parse_actor_url, meta={'bio': bio, 'name': name, 'url': url})

    def _parse_actor_url(self, response):
        actor = ActorItem()
        actor['bio'] = response.meta['bio']
        actor['name'] = response.meta['name']
        actor['url'] = f"{self.base_url}{response.meta['url']}/"
        born = response.xpath('.//*/div[@id="name-born-info" and @class="txt-block"]/time/@datetime').get()
        if born:
            born = born.strip()
        actor['born'] = born
        actor['movies'] = []
        movies = response.xpath('.//*/div[@class="filmo-category-section"]/*/b/a')
        for row in movies[:15]:
            movie = MovieItem()
            movie['title'] = row.xpath('text()').extract_first().strip()
            movie['url'] = self.base_url + row.xpath('@href').extract_first().strip()
            actor['movies'].append(movie['title'])

        return actor


class MovieSpider(scrapy.Spider):
    """Spider crawl top-50 popular actors page, goes to each actor url, parse movies that actor take part in, then
    goes to each movie url and parse its cast.
    It outputs movie list with its cast.

    """

    name = "imdb_movie"
    allowed_domains = ["imdb.com"]
    base_url = 'https://www.imdb.com'
    start_urls = ['https://www.imdb.com/search/name/?gender=male%2Cfemale&ref_=nv_cel_m']

    def parse(self, response):
        table_rows = response.xpath('.//*[@class="lister-item-content"]')
        for row in table_rows:
            url = row.xpath('h3[@class="lister-item-header"]/a/@href').extract_first().strip()
            row_url = self.base_url + url

            yield scrapy.Request(row_url, callback=self._parse_actor_url)

    def _parse_actor_url(self, response):
        movies = response.xpath('.//*/div[@class="filmo-category-section"]/*/b/a')
        for row in movies[:15]:
            title = row.xpath('text()').extract_first().strip()
            url = self.base_url + row.xpath('@href').extract_first().strip()
            request = scrapy.Request(url, callback=self._parse_movie_url, meta={'url': url, 'title': title})

            yield request

    def _parse_movie_url(self, response):
        movie = MovieItem()
        movie['title'] = response.meta['title']
        movie['url'] = response.meta['url']
        movie['cast'] = []
        actors = response.xpath('.//*/a[@data-testid="title-cast-item__actor"]')
        for row in actors:
            movie['cast'].append(row.xpath('text()').extract_first().strip())

        return movie
