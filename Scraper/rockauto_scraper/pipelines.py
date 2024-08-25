import os
import pandas as pd
from scrapy.exceptions import DropItem

class RockAutoScraperPipeline:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.current_make = None
        self.current_year = None
        self.items_buffer = []
        self.items_count = 0

    @classmethod
    def from_crawler(cls, crawler):
        output_dir = crawler.settings.get('OUTPUT_DIR', '/Users/skylerwilson/Desktop/PartsWise/Data/fitment_data/')
        return cls(output_dir)

    def process_item(self, item, spider):
        if not item:
            raise DropItem("Empty item found")

        if item['make'] != self.current_make or item['year'] != self.current_year:
            self.save_buffer(spider)
            self.current_make = item['make']
            self.current_year = item['year']

        # Ensure all required fields are present
        required_fields = ['make', 'year', 'model', 'engine', 'category', 'subcategory', 'part_number', 'description']
        for field in required_fields:
            if field not in item:
                item[field] = None  # Set to None if the field is missing

        self.items_buffer.append(item)
        self.items_count += 1

        if len(self.items_buffer) >= 1000:  # Save every 1000 items
            self.save_buffer(spider)

        return item

    def save_buffer(self, spider):
        if not self.items_buffer:
            return

        df = pd.DataFrame(self.items_buffer)
        filename = f"rockauto_fitment_data_{self.current_make}_{self.current_year}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure all required columns are present
        required_columns = ['make', 'year', 'model', 'engine', 'category', 'subcategory', 'part_number', 'description']
        for column in required_columns:
            if column not in df.columns:
                df[column] = None

        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            # Ensure existing DataFrame has all required columns
            for column in required_columns:
                if column not in existing_df.columns:
                    existing_df[column] = None
            df = pd.concat([existing_df, df], ignore_index=True)
        
        # Remove duplicates based on all columns
        df.drop_duplicates(subset=required_columns, keep='last', inplace=True)
        
        # Save to CSV, including all required columns
        df.to_csv(filepath, index=False, columns=required_columns)
        
        spider.logger.info(f"Saved {len(df)} items for {self.current_make} {self.current_year}")
        
        if hasattr(spider, 'checkpoint'):
            spider.checkpoint.save_checkpoint(self.current_make, self.current_year)
        self.items_buffer = []

    def close_spider(self, spider):
        self.save_buffer(spider)  # Save any remaining items
        spider.logger.info(f"Total items scraped: {self.items_count}")

