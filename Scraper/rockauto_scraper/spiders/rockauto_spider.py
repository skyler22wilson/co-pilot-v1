import scrapy
from scrapy.spiders import CrawlSpider
import json
import os
from fake_useragent import UserAgent
from urllib.parse import urlparse, urljoin
from datetime import datetime

class CheckpointSystem:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.progress = self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'makes': {}, 'paused': False, 'paused_at': None}

    def save_checkpoint(self, make, year, paused=False, paused_at=None):
        self.progress['makes'][make] = year
        self.progress['paused'] = paused
        self.progress['paused_at'] = paused_at
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f)

    def get_last_processed_year(self, make):
        return self.progress['makes'].get(make, None)

    def is_paused(self):
        return self.progress['paused']

    def get_paused_at(self):
        return self.progress['paused_at']

class RockAutoURLSpider(CrawlSpider):
    name = 'rockauto_urls'
    allowed_domains = ['rockauto.com']
    TARGET_MAKES = ['toyota']  # Add more makes if needed

    def __init__(self, checkpoint_file='scraping_checkpoint.json', *args, **kwargs):
        super(RockAutoURLSpider, self).__init__(*args, **kwargs)
        self.ua = UserAgent()
        self.base_url = "https://www.rockauto.com/en/catalog/"
        self.checkpoint = CheckpointSystem(checkpoint_file)

    def start_requests(self):
        make_urls = self.get_make_urls()
        for url in make_urls:
            self.logger.info(f"Starting request for URL: {url}")
            yield scrapy.Request(url, headers={'User-Agent': self.ua.random}, callback=self.parse_make)

    def get_make_urls(self):
        return [f"{self.base_url}{make.lower().replace(' ', '+')}" for make in self.TARGET_MAKES]

    def parse_make(self, response):
        year_urls = self.get_year_urls(response.url)
        for url in year_urls:
            yield scrapy.Request(url, headers={'User-Agent': self.ua.random}, callback=self.parse_year)

    def get_year_urls(self, make_url):
        current_year = datetime.now().year
        return [f"{make_url},{year}" for year in range(current_year, 1999, -1)]

    def parse_year(self, response):
        model_urls = self.get_model_urls(response)
        for url in model_urls:
            yield scrapy.Request(url, headers={'User-Agent': self.ua.random}, callback=self.parse_model)

    def get_model_urls(self, response):
        model_links = response.xpath("//a[@class='navlabellink']/@href").extract()
        base_path = urlparse(response.url).path
        
        return [urljoin(response.url, link) for link in model_links 
                if len(urlparse(link).path.split(',')) == 3 
                and urlparse(link).path.split(',')[2] not in base_path]

    def parse_model(self, response):
        engine_urls = self.get_engine_urls(response)
        for engine in engine_urls:
            yield scrapy.Request(engine['url'], headers={'User-Agent': self.ua.random}, callback=self.parse_engine, meta={'engine_name': engine['name']})

    def get_engine_urls(self, response):
        engine_tds = response.xpath("//td[@class='nlabel']")
        base_path = urlparse(response.url).path
        
        return [{'name': link.xpath("text()").get().strip(), 'url': urljoin(response.url, link.xpath("@href").get())}
                for td in engine_tds
                for link in td.xpath(".//a[@class='navlabellink']")
                if len(urlparse(link.xpath("@href").get()).path.split(',')) > 3
                and urlparse(link.xpath("@href").get()).path.split(',')[3] not in base_path]

    def parse_engine(self, response):
        category_urls = self.get_category_urls(response)
        for category in category_urls:
            yield scrapy.Request(category['url'], headers={'User-Agent': self.ua.random}, callback=self.parse_category, 
                                 meta={'engine_name': response.meta['engine_name'], 'category_name': category['name']})

    def get_category_urls(self, response):
        category_links = response.xpath("//a[@class='navlabellink']/@href").extract()
        base_path = urlparse(response.url).path
        
        return [{'name': response.xpath(f"//a[@href='{link}']/text()").get().strip(), 'url': urljoin(response.url, link)}
                for link in category_links
                if len(urlparse(link).path.split(',')) == 6
                and urlparse(link).path.split(',')[5] not in base_path]

    def parse_category(self, response):
        subcategory_urls = self.get_subcategory_urls(response)
        for subcategory in subcategory_urls:
            yield scrapy.Request(subcategory['url'], headers={'User-Agent': self.ua.random}, callback=self.parse_subcategory,
                                 meta={'engine_name': response.meta['engine_name'], 
                                       'category_name': response.meta['category_name'],
                                       'subcategory_name': subcategory['name']})

    def get_subcategory_urls(self, response):
        subcategory_tds = response.xpath("//td[@class='nlabel']")
        base_path = urlparse(response.url).path
        
        return [{'name': td.xpath("text()").get().strip(), 'url': urljoin(response.url, link.xpath("@href").get())}
                for td in subcategory_tds
                for link in td.xpath(".//a[@class='navlabellink']")
                if len(urlparse(link.xpath("@href").get()).path.split(',')) == 8
                and urlparse(link.xpath("@href").get()).path.split(',')[7] not in base_path]

    def parse_subcategory(self, response):
        if self.is_banned(response):
            self.handle_ban()
            return

        fitment_data = self.scrape_fitment_data(response)
        for part in fitment_data:
            yield part

    def extract_info_from_url(self, url):
        parts = urlparse(url).path.split(',')
        info = {
            'make': parts[0].split('/')[-1] if len(parts) > 0 else None,
            'year': parts[1] if len(parts) > 1 else None,
            'model': parts[2] if len(parts) > 2 else None,
            'engine': parts[3] if len(parts) > 3 else None,
            'category': parts[5].replace('+', ' ').replace('&', 'and') if len(parts) > 5 else None,
            'subcategory': parts[6].replace('+', ' ').replace('&', 'and') if len(parts) > 6 else None
        }
        return {k: v for k, v in info.items() if v is not None}

    def scrape_fitment_data(self, response):
        url_info = self.extract_info_from_url(response.url)
        tbody_elements = response.xpath("//tbody[re:test(@id, 'listingcontainer\\[\\d+\\]')]")
        
        all_parts = []
        for tbody in tbody_elements:
            part_info = url_info.copy()
            part_number = tbody.xpath(".//span[@class='listing-final-partnumber as-link-if-js']/text()").get()
            description = tbody.xpath(".//span[@class='listing-footnote-text']/text()").get()
            
            if part_number:
                part_info['part_number'] = part_number.strip()
            if description:
                part_info['description'] = description.strip()
            
            all_parts.append(part_info)
        
        return all_parts

    def is_banned(self, response):
        if response.status in [403, 429]:
            self.logger.warning(f"Request was banned with status code: {response.status}")
            return True
        banned_phrases = ["Access denied", "You have been blocked", "Your IP has been blocked", "403 Forbidden", "429 Too Many Requests"]
        for phrase in banned_phrases:
            if phrase in response.text:
                self.logger.warning(f"Banned phrase detected in response: '{phrase}'")
                return True
        return False

    def handle_ban(self):
        self.logger.warning("Ban detected. Pausing the scraper...")
        self.checkpoint.save_checkpoint(paused=True, paused_at=datetime.now().isoformat())
        self.crawler.engine.pause()

    def process_exception(self, response, exception, spider):
        if isinstance(exception, scrapy.exceptions.IgnoreRequest):
            self.logger.warning(f"Request ignored: {response.url}")
        elif isinstance(exception, scrapy.exceptions.CloseSpider):
            self.logger.error(f"Spider closed: {exception}")
            self.checkpoint.save_checkpoint(paused=True, paused_at=datetime.now().isoformat())
        else:
            self.logger.error(f"Unhandled exception: {exception}")
        return None
