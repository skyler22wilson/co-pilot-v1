from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, UniqueConstraint, Index, Date, DateTime, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
import os
import logging
from datetime import datetime
import enum

Base = declarative_base()

class Parts(Base):
    __tablename__ = 'parts'
    part_number = Column(String(15), primary_key=True, index=True)
    description = Column(String)
    supplier_name = Column(String, index=True)
    price = Column(Float(precision=2))
    cost_per_unit = Column(Float(precision=2))
    # Relationships
    inventory = relationship("Inventory", back_populates="part", uselist=False)
    part_metrics = relationship("PartMetrics", back_populates="part", uselist=False)
    temporal_metrics = relationship("TemporalMetrics", back_populates="part", uselist=False)
    supplier_orders = relationship("SupplierOrders", back_populates="part")
    sales = relationship("Sales", back_populates="part")

    __table_args__ = (
        Index('idx_parts_supplier_price', 'supplier_name', 'price'),
        Index('idx_parts_cost', 'cost_per_unit')
    )

    @hybrid_property
    def margin(self):
        return self.price - self.cost_per_unit

    @hybrid_property
    def margin_percentage(self):
        return (self.margin / self.price) * 100 if self.price > 0 else 0

    def __repr__(self):
        return f"<Part(part_number='{self.part_number}', description='{self.description}')>"

class PartStatus(enum.Enum):
    essential = 'essential'
    active = 'active'
    idle = 'idle'
    obsolete = 'obsolete'

class Inventory(Base):
    __tablename__ = 'inventory'
    id = Column(Integer, primary_key=True, autoincrement=True, ondelete='CASCADE', onupdate='CASCADE')
    part_number = Column(String(15), ForeignKey('parts.part_number'), index=True)
    quantity = Column(Integer)
    safety_stock = Column(Integer)
    reorder_point = Column(Integer)
    part_status = Column(Enum(PartStatus))
    negative_on_hand = Column(Integer)
    last_updated = Column(DateTime, default=datetime.now(datetime.UTC))
    # Relationship
    part = relationship("Parts", back_populates="inventory")

    __table_args__ = (
        Index('idx_inventory_status_quantity', 'part_status', 'quantity'),
    )

    @hybrid_property
    def is_low_stock(self):
        return self.quantity < self.safety_stock
    
    @hybrid_property
    def reorder(self):
        return self.quantity < self.reorder_point
    
    #need to add hybrid properties for things like stockouts 

    def __repr__(self):
        return f"<Inventory(part_number='{self.part_number}', quantity={self.quantity})>"

class PartMetrics(Base):
    __tablename__ = 'part_metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number'), index=True, ondelete='CASCADE', onupdate='CASCADE')
    roi = Column(Float(precision=2))
    demand = Column(Float(precision=2))
    obsolescence_risk = Column(Float(precision=2))
    days_of_inventory_outstanding = Column(Integer)
    sell_through_rate = Column(Float(precision=2))
    order_to_sales_ratio = Column(Float(precision=2))

    part = relationship("Parts", back_populates="part_metrics")

    __table_args__ = (
        Index('idx_metrics_risk_demand', 'obsolescence_risk', 'demand'),
    )

    def __repr__(self):
        return f"<PartMetrics(part_number='{self.part_number}', roi={self.roi})>"

class TemporalMetrics(Base):
    __tablename__ = 'temporal_metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number'), index=True)
    months_no_sale = Column(Integer)
    sales_volatility = Column(Float(precision=4))
    sales_trend = Column(Float(precision=4))
    recent_sales_trend = Column(Float(precision=4))
    twelve_m_days_supply = Column(Float(precision=2))
    three_m_days_supply = Column(Float(precision=2))
    one_m_days_supply = Column(Float(precision=2))
    three_m_turnover = Column(Float(precision=4))
    turnover = Column(Float(precision=4))
    last_updated = Column(DateTime, default=datetime.now(datetime.UTC), onupdate=datetime.now(datetime.UTC))
    # Relationship
    part = relationship("Parts", back_populates="temporal_metrics")

    __table_args__ = (
        Index('idx_temporal_part_turnover', 'part_number', 'turnover'),
    )

    def __repr__(self):
        return f"<TemporalMetrics(part_number='{self.part_number}', turnover={self.turnover})>"

class SupplierOrders(Base):
    __tablename__ = 'supplier_orders'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number'), index=True)
    quantity_ordered_ytd = Column(Integer)
    special_orders_ytd = Column(Boolean)
    # Relationship
    part = relationship("Parts", back_populates="supplier_orders")

    __table_args__ = (
        Index('idx_supplier_orders_ytd', 'part_number', 'quantity_ordered_ytd'),
    )

    def __repr__(self):
        return f"<SupplierOrders(part_number='{self.part_number}', quantity_ordered_ytd={self.quantity_ordered_ytd})>"

class Sales(Base):
    __tablename__ = 'sales'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number', ondelete='CASCADE', onupdate='CASCADE'))
    month = Column(String(10))
    year = Column(Integer)
    quantity_sold = Column(Integer)
    # Relationship
    part = relationship("Parts", back_populates="sales")

    __table_args__ = (
        UniqueConstraint('part_number', 'month', 'year', name='unique_sales'),
        Index('idx_sales_part_date', 'part_number', 'year', 'month'),
        Index('idx_sales_quantity', 'quantity_sold')
    )

    def __repr__(self):
        return f"<Sales(part_number='{self.part_number}', year={self.year}, month='{self.month}', quantity_sold={self.quantity_sold})>"

class FieldMetadata(Base):
    __tablename__ = 'field_metadata'
    field_name = Column(String, primary_key=True)
    table_name = Column(String)
    description = Column(String)
    calculation = Column(String)


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