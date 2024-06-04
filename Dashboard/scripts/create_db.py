from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, UniqueConstraint, DECIMAL, Index
from sqlalchemy.ext.declarative import declarative_base
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
    margin = Column(Float)
    months_no_sale = Column(Integer)
    quantity_ordered_ytd = Column(Integer)
    special_orders_ytd = Column(Integer)
    negative_on_hand = Column(Integer)
    cost_per_unit = Column(Integer)
    roi = Column(Float, index=True)
    annual_days_supply = Column(Float)
    three_month_days_supply = Column(Float)
    one_month_days_supply = Column(Float)
    annual_turnover = Column(Float)
    three_month_turnover = Column(Float)
    one_month_turnover = Column(Float)
    sales_to_stock_ratio = Column(Float)
    order_to_sales_ratio = Column(Float)
    safety_stock = Column(Integer)
    reorder_point = Column(Integer)
    demand = Column(Float)
    inventory_category = Column(String, index=True)
    obsolescence_risk = Column(Float, index=True)
    supplier_id = Column(String, ForeignKey('supplier_parts_summary.supplier_id', ondelete='CASCADE', onupdate='CASCADE'))

    sales = relationship('Sales', back_populates='parts')
    supplier_parts_summary = relationship('SupplierPartsSummary', back_populates='parts')

    __table_args__ = (
        Index('idx_supplier_name_part_number', 'supplier_name', 'part_number'),
        Index('idx_part_number_roi', 'part_number', 'roi'),
        Index('idx_supplier_name_inventory_category', 'supplier_name', 'inventory_category'),
        Index('idx_supplier_name_obsolescence_risk', 'supplier_name', 'obsolescence_risk')
    )

    @property
    def gross_profit(self):
        total_quantity_sold = sum([sale.quantity_sold for sale in self.sales])
        return total_quantity_sold * (self.price - self.cost_per_unit)

    @property
    def cogs(self):
        total_quantity_sold = sum([sale.quantity_sold for sale in self.sales])
        return total_quantity_sold * self.cost_per_unit

    @property
    def cost(self):
        return self.price - self.cogs

    @property
    def margin_percentage(self):
        total_revenue = self.price * sum([sale.quantity_sold for sale in self.sales])
        return (self.gross_profit / total_revenue) * 100

    @property
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


class SupplierPartsSummary(Base):
    __tablename__ = 'supplier_parts_summary'
    supplier_id = Column(String, primary_key=True)
    supplier_name = Column(String, index=True)  # Direct index on supplier_name
    total_quantity = Column(Integer)
    total_negative_on_hand = Column(Integer)
    total_cost = Column(DECIMAL)
    average_margin = Column(DECIMAL)
    total_sales_revenue = Column(DECIMAL)
    total_cogs = Column(DECIMAL)
    total_gross_profit = Column(DECIMAL)
    average_turnover = Column(DECIMAL)
    average_days_supply = Column(DECIMAL)
    average_months_no_sale = Column(Integer)
    average_obsolescence_risk = Column(DECIMAL)
    average_demand = Column(DECIMAL)
    

    # Composite indexes
    __table_args__ = (
        Index('idx_supplier_name_total_quantity', 'supplier_name', 'total_quantity'),
        Index('idx_supplier_name_total_negative_on_hand', 'supplier_name', 'total_negative_on_hand'),
        Index('idx_supplier_name_total_cost', 'supplier_name', 'total_cost'),
        Index('idx_supplier_name_total_sales_revenue', 'supplier_name', 'total_sales_revenue'),
        Index('idx_supplier_name_total_gross_profit', 'supplier_name', 'total_gross_profit'),
        Index('idx_supplier_name_average_turnover', 'supplier_name', 'average_turnover'),
        Index('idx_supplier_name_average_obsolescence_risk', 'supplier_name', 'average_obsolescence_risk')
    )

    parts = relationship('Parts', back_populates='supplier_parts_summary')
    sales_summary = relationship('SupplierSalesSummary', back_populates='supplier')


class SupplierSalesSummary(Base):
    __tablename__ = 'supplier_sales_summary'
    id = Column(Integer, primary_key=True, autoincrement=True)
    supplier_name = Column(String, index=True)
    year = Column(Integer)
    month = Column(String)
    quantity_sold = Column(Integer)
    supplier_id = Column(String, ForeignKey('supplier_parts_summary.supplier_id', ondelete='CASCADE', onupdate='CASCADE'))
    __table_args__ = (
        UniqueConstraint('supplier_id', 'month', 'year', name='unique_supplier_sales'),
        Index('idx_month_year_summary', 'month', 'year')
    )

    supplier = relationship('SupplierPartsSummary', back_populates='sales_summary')

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