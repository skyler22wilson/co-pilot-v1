import os
import pandas as pd
from scrapy.exceptions import DropItem

class RockAutoScraperPipeline:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.current_make = None
        self.current_year = None
        self.model_buffers = {}
        self.items_count = 0

    @classmethod
    def from_crawler(cls, crawler):
        output_dir = crawler.settings.get('OUTPUT_DIR', '/Users/skylerwilson/Desktop/PartsWise/Data/fitment_data/')
        return cls(output_dir)

    def process_item(self, item, spider):
        if not item:
            raise DropItem("Empty item found")

        make = item['make']
        year = item['year']
        model = item['model']

        if make != self.current_make or year != self.current_year:
            self.save_all_buffers(spider)
            self.current_make = make
            self.current_year = year
            self.model_buffers = {}

        if model not in self.model_buffers:
            self.model_buffers[model] = []

        # Ensure all required fields are present
        required_fields = ['make', 'year', 'model', 'engine', 'category', 'subcategory', 'part_number', 'description']
        for field in required_fields:
            if field not in item:
                item[field] = None  # Set to None if the field is missing

        self.model_buffers[model].append(item)
        self.items_count += 1

        if len(self.model_buffers[model]) >= 500:  # Save every 500 items per model
            self.save_model_buffer(spider, model)

        return item

    def save_model_buffer(self, spider, model):
        if not self.model_buffers[model]:
            return

        df = pd.DataFrame(self.model_buffers[model])
        filename = f"rockauto_fitment_data_{self.current_make}_{self.current_year}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure all required columns are present
        required_columns = ['make', 'year', 'model', 'engine', 'category', 'subcategory', 'part_number', 'description']
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        mode = 'a' if os.path.exists(filepath) else 'w'
        header = not os.path.exists(filepath)
        
        # Append to CSV, including all required columns
        df.to_csv(filepath, mode=mode, index=False, columns=required_columns, header=header)
        
        spider.logger.info(f"Saved {len(df)} items for {self.current_make} {self.current_year} - {model}")
        
        if hasattr(spider, 'checkpoint'):
            spider.checkpoint.save_checkpoint(self.current_make, self.current_year, model)
        self.model_buffers[model] = []

    def save_all_buffers(self, spider):
        for model in self.model_buffers:
            self.save_model_buffer(spider, model)

    def close_spider(self, spider):
        self.save_all_buffers(spider)  # Save any remaining items
        spider.logger.info(f"Total items scraped: {self.items_count}")

