import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import csv
import json
import os
from fake_useragent import UserAgent
from urllib.parse import urlparse, urljoin

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

    def __init__(self, makes='toyota', start_year=2000, end_year=2024, checkpoint_file='scraping_checkpoint.json', *args, **kwargs):
        super(RockAutoURLSpider, self).__init__(*args, **kwargs)
        self.makes = makes.split(',')
        self.start_year = int(start_year)
        self.end_year = int(end_year)
        self.checkpoint = CheckpointSystem(checkpoint_file)
        self.ua = UserAgent()
        
        self.start_urls = [f'https://www.rockauto.com/en/catalog/{make}' for make in self.makes]

        make_pattern = '|'.join(self.makes)
        year_pattern = '|'.join(str(year) for year in range(self.start_year, self.end_year + 1))
        
        self.rules = (
            Rule(LinkExtractor(allow=rf'/en/catalog/({make_pattern}),({year_pattern})$'), callback='parse_year', follow=True),
            Rule(LinkExtractor(allow=rf'/en/catalog/({make_pattern}),({year_pattern}),[^,]+$'), callback='parse_model', follow=True),
            Rule(LinkExtractor(allow=rf'/en/catalog/({make_pattern}),({year_pattern}),[^,]+,[^,]+$'), callback='parse_engine', follow=True),
            Rule(LinkExtractor(allow=rf'/en/catalog/({make_pattern}),({year_pattern}),[^,]+,[^,]+,[^,]+$'), callback='parse_category', follow=True),
            Rule(LinkExtractor(allow=rf'/en/catalog/({make_pattern}),({year_pattern}),[^,]+,[^,]+,[^,]+,[^,]+$'), callback='parse_subcategory', follow=True),
        )

        self.url_file = open('rockauto_urls.csv', 'w', newline='')
        self.url_writer = csv.writer(self.url_file)
        self.url_writer.writerow(['URL', 'Depth', 'Make', 'Year'])

    def parse_year(self, response):
        self.logger.info(f"Processing Year Page: {response.url}")
        year_urls = self.get_year_urls(response.url)
        for url in year_urls:
            yield scrapy.Request(url, headers={'User-Agent': self.ua.random}, callback=self.parse_model)

    def parse_model(self, response):
        self.logger.info(f"Processing Model Page: {response.url}")
        model_urls = self.get_model_urls(response)
        for url in model_urls:
            yield scrapy.Request(url, headers={'User-Agent': self.ua.random}, callback=self.parse_engine)

    def parse_engine(self, response):
        self.logger.info(f"Processing Engine Page: {response.url}")
        engine_urls = self.get_engine_urls(response)
        for url in engine_urls:
            yield scrapy.Request(url['url'], headers={'User-Agent': self.ua.random}, callback=self.parse_category)

    def parse_category(self, response):
        self.logger.info(f"Processing Category Page: {response.url}")
        category_urls = self.get_category_urls(response)
        for url in category_urls:
            yield scrapy.Request(url['url'], headers={'User-Agent': self.ua.random}, callback=self.parse_subcategory)

    def parse_subcategory(self, response):
        self.logger.info(f"Processing Subcategory Page: {response.url}")
        fitment_data = self.scrape_fitment_data(response)
        if fitment_data:
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

    def get_model_urls(self, response):
        model_links = response.xpath("//a[@class='navlabellink']/@href").extract()
        base_path = urlparse(response.url).path
        
        return [urljoin(response.url, link) for link in model_links 
                if len(urlparse(link).path.split(',')) == 3 
                and urlparse(link).path.split(',')[2] not in base_path]

    def get_engine_urls(self, response):
        engine_tds = response.xpath("//td[@class='nlabel']")
        base_path = urlparse(response.url).path
        
        return [{'name': link.xpath("text()").get().strip(), 'url': urljoin(response.url, link.xpath("@href").get())}
                for td in engine_tds
                for link in td.xpath(".//a[@class='navlabellink']")
                if len(urlparse(link.xpath("@href").get()).path.split(',')) > 3
                and urlparse(link.xpath("@href").get()).path.split(',')[3] not in base_path]

    def get_category_urls(self, response):
        category_links = response.xpath("//a[@class='navlabellink']/@href").extract()
        base_path = urlparse(response.url).path
        
        return [{'name': response.xpath(f"//a[@href='{link}']/text()").get().strip(), 'url': urljoin(response.url, link)}
                for link in category_links
                if len(urlparse(link).path.split(',')) == 6
                and urlparse(link).path.split(',')[5] not in base_path]

    def get_subcategory_urls(self, response):
        subcategory_tds = response.xpath("//td[@class='nlabel']")
        base_path = urlparse(response.url).path
        
        return [{'name': td.xpath("text()").get().strip(), 'url': urljoin(response.url, link.xpath("@href").get())}
                for td in subcategory_tds
                for link in td.xpath(".//a[@class='navlabellink']")
                if len(urlparse(link.xpath("@href").get()).path.split(',')) == 8
                and urlparse(link.xpath("@href").get()).path.split(',')[7] not in base_path]
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, headers={'User-Agent': self.ua.random})

    def parse_url(self, response):
        url = response.url
        parts = url.split(',')
        depth = len(parts) - 1
        make = parts[0].split('/')[-1]
        year = parts[1] if len(parts) > 1 else ''

        if make in self.makes and self.start_year <= int(year) <= self.end_year:
            self.url_writer.writerow([url, depth, make, year])
            self.logger.info(f'Extracted URL: {url} (Depth: {depth}, Make: {make}, Year: {year})')
            self.checkpoint.save_checkpoint(make, int(year))
            return {'url': url, 'depth': depth, 'make': make, 'year': year}

    def closed(self, reason):
        self.url_file.close()
