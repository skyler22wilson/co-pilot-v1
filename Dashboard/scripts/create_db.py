from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import relationship
import os
import logging

Base = declarative_base()

class Parts(Base):
    __tablename__ = 'parts'
    part_number = Column(String, primary_key=True, index=True)
    description = Column(String)
    supplier_name = Column(String, index=True)
    quantity = Column(Integer)
    price = Column(Float)
    cost_per_unit = Column(Integer)
    months_no_sale = Column(Integer)
    quantity_ordered_ytd = Column(Integer)
    special_orders_ytd = Column(Integer)
    negative_on_hand = Column(Integer)
    roi = Column(Float, index=True)
    annual_days_supply = Column(Float)
    three_month_days_supply = Column(Float)
    one_month_days_supply = Column(Float)
    annual_turnover = Column(Float)
    three_month_turnover = Column(Float)
    one_month_turnover = Column(Float)
    sell_through_rate = Column(Float)
    days_of_inventory_outstanding = Column(Integer)
    order_to_sales_ratio = Column(Float)
    safety_stock = Column(Integer)
    reorder_point = Column(Integer)
    demand = Column(Float)
    inventory_category = Column(String, index=True)
    obsolescence_risk = Column(Float, index=True)
    sales_id = Column(String, ForeignKey('sales.id', ondelete='CASCADE', onupdate='CASCADE'))

    sales = relationship('Sales', back_populates='parts')

    __table_args__ = (
        Index('idx_supplier_name_part_number', 'supplier_name', 'part_number'),
        Index('idx_part_number_roi', 'part_number', 'roi'),
        Index('idx_supplier_name_inventory_category', 'supplier_name', 'inventory_category'),
        Index('idx_supplier_name_obsolescence_risk', 'supplier_name', 'obsolescence_risk')
    )

    @hybrid_property
    def gross_profit(self):
        total_quantity_sold = sum([sale.quantity_sold for sale in self.sales])
        return total_quantity_sold * (self.price - self.cost_per_unit)

    @hybrid_property
    def cogs(self):
        total_quantity_sold = sum([sale.quantity_sold for sale in self.sales])
        return total_quantity_sold * self.cost_per_unit

    @hybrid_property
    def cost(self):
        return self.price - self.cogs

    @hybrid_property
    def margin_percentage(self):
        total_revenue = self.price * sum([sale.quantity_sold for sale in self.sales])
        return (self.gross_profit / total_revenue) * 100

    @hybrid_property
    def sales_revenue(self):
        total_revenue = self.price * sum([sale.quantity_sold for sale in self.sales])
        return total_revenue

class Sales(Base):
    __tablename__ = 'sales'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String, ForeignKey('parts.part_number', ondelete='CASCADE', onupdate='CASCADE'))
    month = Column(String)
    year = Column(Integer)
    quantity_sold = Column(Integer)
    __table_args__ = (UniqueConstraint('part_number', 'month', 'year', name='unique_sales'),
                      Index('idx_month_year_sales', 'month', 'year'))

    parts = relationship('Parts', back_populates='sales')


# Function to create the schema using SQLAlchemy
def generate_db_file_path(user_identifier='island_moto'):
    base_directory = 'data/databases/'
    db_file_name = f'partswise_{user_identifier}.db'
    db_file_path = os.path.join(base_directory, db_file_name)

    # Ensure the directory exists before returning the file path
    directory = os.path.dirname(db_file_path)
    print(f"Directory to create: {directory}")
    if not os.path.exists(directory):
        logging.info(f"Creating directory for database at {directory}")
        os.makedirs(directory, exist_ok=True)
    return db_file_path

def create_database_schema(db_file_path):
    logging.info(f"Creating database schema at {db_file_path}")
    engine = create_engine(f'sqlite:///{db_file_path}')
    Base.metadata.create_all(engine)
    logging.info("Database schema created successfully.")