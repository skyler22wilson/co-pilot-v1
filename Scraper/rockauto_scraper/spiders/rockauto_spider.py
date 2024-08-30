import scrapy
from scrapy.spiders import CrawlSpider
from urllib.parse import urlparse
from datetime import datetime
import re
import json
import os
import pandas as pd
from tqdm import tqdm

class CheckpointSystem:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.progress = self.load_checkpoint()

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'makes': {}}

    def save_checkpoint(self, make, year, model):
        if make not in self.progress['makes']:
            self.progress['makes'][make] = {}
        if year not in self.progress['makes'][make]:
            self.progress['makes'][make][year] = []
        if model not in self.progress['makes'][make][year]:
            self.progress['makes'][make][year].append(model)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.progress, f)

    def get_processed_models(self, make, year):
        return self.progress['makes'].get(make, {}).get(str(year), [])

class RockAutoSpider(CrawlSpider):
    name = 'rockauto_urls'
    allowed_domains = ['rockauto.com']
    TARGET_MAKES = ['toyota']
    base_url = "https://www.rockauto.com/en/catalog/"

    def __init__(self, *args, **kwargs):
        super(RockAutoSpider, self).__init__(*args, **kwargs)
        self.checkpoint = CheckpointSystem('/Users/skylerwilson/Desktop/PartsWise/Data/fitment_data/scraping_checkpoint.json')
        self.tbody_pattern = re.compile(r"listingcontainer\[\d+\]")

    def get_make_urls(self):
        return [f"{self.base_url}{make.lower().replace(' ', '+')}" for make in self.TARGET_MAKES]

    def start_requests(self):
        make_urls = self.get_make_urls()
        for make_url in make_urls:
            make_name = urlparse(make_url).path.split('/')[-1]
            current_year = datetime.now().year
            years = range(current_year, 2009, -1)  # From current year to 2010

            self.logger.info(f"Starting scraping for {make_name.capitalize()} from year {current_year} to 2010")

            for year in tqdm(years, desc=f"Years for {make_name.capitalize()}"):
                year_url = f"{make_url},{year}"
                yield scrapy.Request(year_url, callback=self.parse_year, meta={'make': make_name, 'year': year})

    def parse_year(self, response):
        make = response.meta['make']
        year = response.meta['year']

        self.logger.info(f"Parsing year page for {make} {year}: {response.url}")

        model_links = response.css("a.navlabellink")
        self.logger.info(f"Found {len(model_links)} potential model links")

        model_urls = []
        for i, link in enumerate(model_links):
            href = link.attrib.get('href')
            text = link.xpath("text()").get()

            if href and text:
                parsed_href = urlparse(href)
                href_parts = parsed_href.path.strip('/').split('/')

                if len(href_parts) == 3 and href_parts[0] == 'en' and href_parts[1] == 'catalog':
                    parts = href_parts[2].split(',')
                    if len(parts) == 3 and parts[0] == make and parts[1] == str(year):
                        model_name = text.strip()
                        model_url = response.urljoin(href)
                        model_urls.append((model_name, model_url))

        self.logger.info(f"Found {len(model_urls)} models for {make} {year}")
        for i, (model_name, model_url) in enumerate(model_urls):
            self.logger.debug(f"Yielding request for {make} {year} {model_name}: {model_url}")
            yield scrapy.Request(model_url, callback=self.parse_model, 
                                meta={'make': make, 'year': year, 'model': model_name})

        if not model_urls:
            self.logger.warning(f"No models found for {make} {year}. This might be an error or the end of available data.")

    def parse_model(self, response):
        make = response.meta['make']
        year = response.meta['year']
        model = response.meta['model']
        
        self.logger.info(f"Parsing model page for {make} {year} {model}: {response.url}")
        
        # Updated CSS selector to find engine links
        engine_links = response.css("a.navlabellink")
        self.logger.info(f"Found {len(engine_links)} potential engine links")

        engine_urls = []
        for i, link in enumerate(engine_links):
            href = link.attrib.get('href')
            text = link.xpath("text()").get()
            self.logger.debug(f"Link {i}: href='{href}', text='{text}'")
            
            if href and text:
                parsed_href = urlparse(href)
                href_parts = parsed_href.path.strip('/').split('/')
                self.logger.debug(f"Link {i}: parsed path parts = {href_parts}")
                
                if len(href_parts) == 3 and href_parts[0] == 'en' and href_parts[1] == 'catalog':
                    parts = href_parts[2].split(',')
                    if len(parts) == 5 and parts[0] == make and parts[1] == str(year) and parts[2] == model.lower():
                        engine_name = text.strip()
                        engine_url = response.urljoin(href)
                        engine_urls.append((engine_name, engine_url))
                        self.logger.debug(f"Link {i}: Added as engine: {engine_name} - {engine_url}")
                    else:
                        self.logger.debug(f"Link {i}: Skipped (not a valid engine link)")
                else:
                    self.logger.debug(f"Link {i}: Skipped (not a catalog link)")

        self.logger.info(f"Found {len(engine_urls)} engines for {make} {year} {model}")
        for i, (engine_name, engine_url) in enumerate(engine_urls):
            self.logger.debug(f"Engine {i}: {engine_name} - {engine_url}")
            yield scrapy.Request(engine_url, callback=self.parse_engine, 
                                meta={'make': make, 'year': year, 'model': model, 'engine': engine_name})

        if not engine_urls:
            self.logger.warning(f"No engines found for {make} {year} {model}. This might be an error or the end of available data.")

    def parse_engine(self, response):
        make = response.meta['make']
        year = response.meta['year']
        model = response.meta['model']
        engine = response.meta['engine']
        
        self.logger.info(f"Parsing engine page for {make} {year} {model} {engine}: {response.url}")
        
        category_links = response.css("a.navlabellink")
        self.logger.info(f"Found {len(category_links)} potential category links")

        base_path = urlparse(response.url).path
        category_urls = []

        for link in category_links:
            href = link.attrib.get('href')
            text = link.css("::text").get().strip()
            
            if href and text:
                parsed_href = urlparse(href)
                href_parts = parsed_href.path.strip('/').split(',')
                
                if len(href_parts) == 6 and href_parts[5] not in base_path:
                    category_url = response.urljoin(href)
                    engine_id = href_parts[4]
                    category_name_from_url = href_parts[5].replace('+', ' ')
                    category_urls.append((engine_id, text, category_url, category_name_from_url))

        self.logger.info(f"Found {len(category_urls)} categories for {make} {year} {model} {engine}")
        for engine_id, category_name_from_link, category_url, category_name_from_url in category_urls:
            self.logger.debug(f"Category: {category_name_from_link} (URL: {category_name_from_url}) - {category_url}")
            yield scrapy.Request(
                category_url, 
                callback=self.parse_category, 
                meta={
                    'make': make, 
                    'year': year, 
                    'model': model, 
                    'engine': engine, 
                    'engine_id': engine_id, 
                    'category': category_name_from_link,
                    'category_url': category_name_from_url
                }
            )
        if not category_urls:
            self.logger.warning(f"No categories found for {make} {year} {model} {engine}. This might be an error or the end of available data.")

    def parse_category(self, response):
        make = response.meta['make']
        year = response.meta['year']
        model = response.meta['model']
        engine = response.meta['engine']
        category = response.meta['category']
        category_url_name = response.meta['category_url']
        engine_id = response.meta['engine_id']
        
        self.logger.info(f"Parsing category page for {make} {year} {model} {engine}")
        self.logger.info(f"Category: {category} (URL: {category_url_name})")
        
        subcategory_links = response.css("td.nlabel a.navlabellink")
        self.logger.info(f"Found {len(subcategory_links)} potential subcategory links")

        base_path = urlparse(response.url).path
        subcategory_urls = []

        for link in subcategory_links:
            href = link.attrib.get('href')
            text = link.css("::text").get().strip()
            
            if href and text:
                parsed_href = urlparse(href)
                href_parts = parsed_href.path.strip('/').split(',')
                
                if len(href_parts) == 8 and href_parts[7] not in base_path:
                    subcategory_url = response.urljoin(href)
                    subcategory_urls.append((text, subcategory_url, href_parts[7]))

        self.logger.info(f"Found {len(subcategory_urls)} subcategories for {make} {year} {model} {engine} {category}")
        for subcategory_name, subcategory_url, subcategory_id in subcategory_urls:
            self.logger.debug(f"Subcategory: {subcategory_name} - {subcategory_url}")
            yield scrapy.Request(subcategory_url, callback=self.parse_subcategory, 
                     meta={'make': make, 'year': year, 'model': model, 'engine': engine, 
                           'category': category, 'subcategory': subcategory_name, 
                           'engine_id': engine_id, 'subcategory_id': subcategory_id})

        if not subcategory_urls:
            self.logger.warning(f"No subcategories found for {make} {year} {model} {engine} {category}. This might be an error or the end of available data.")

    def parse_subcategory(self, response):
        make = response.meta['make']
        year = response.meta['year']
        model = response.meta['model']
        engine = response.meta['engine']
        engine_id = response.meta['engine_id']
        category = response.meta['category']
        subcategory = response.meta['subcategory']
        subcategory_id = response.meta.get('subcategory_id', '')

        self.logger.info(f"Parsing subcategory page for {make} {year} {model} {engine} {category} {subcategory}: {response.url}")

        # Find all rows in the table
        rows = response.css('tbody[id^="listingcontainer"]')
        
        for row in rows:
            part_number = row.css('span.listing-final-partnumber::text').get()
            description = row.css('span.span-link-underline-remover::text').get()
            
            if part_number and description:
                part_info = {
                    'make': make,
                    'year': year,
                    'model': model,
                    'engine': engine,
                    'engine_id': engine_id,
                    'category': category,
                    'subcategory': subcategory,
                    'subcategory_id': subcategory_id,
                    'url': response.url,
                    'part_number': part_number.strip(),
                    'description': description.strip(),
                }
                
                # Clean up the data
                part_info = {k: v.strip() if isinstance(v, str) else v for k, v in part_info.items()}
                part_info = {k: v if v else None for k, v in part_info.items()}  
                
                yield part_info

        self.logger.info(f"Scraped {len(rows)} parts for subcategory {subcategory}")
        self.checkpoint.save_checkpoint(make, year)

        # Check for pagination
        next_page = response.css('a.navpageurlnext::attr(href)').get()
        if next_page:
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse_subcategory, meta=response.meta)

    def extract_info_from_url(self, url):
        parts = urlparse(url).path.split(',')
        info = {
            'model': parts[2] if len(parts) > 2 else None,
        }
        return {k: v for k, v in info.items() if v is not None}
    

