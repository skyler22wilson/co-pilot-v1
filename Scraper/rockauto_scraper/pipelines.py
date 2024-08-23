# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter


import csv

class RockAutoPipeline:
    def __init__(self):
        self.file = open('rockauto_data.csv', 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=['make', 'year', 'model', 'engine', 'category', 'subcategory', 'part_number', 'description'])
        self.writer.writeheader()

    def process_item(self, item, spider):
        self.writer.writerow(item)
        return item

    def close_spider(self, spider):
        self.file.close()

