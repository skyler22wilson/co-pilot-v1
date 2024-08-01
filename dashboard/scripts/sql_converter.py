from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Date, Boolean, ForeignKey, Index, UniqueConstraint, Enum, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
from sqlalchemy import text
import enum
import json
import sqlite3
import os

Base = declarative_base()

class PartStatus(enum.Enum):
    active = 'active'
    obsolete = 'obsolete'
    pending = 'pending'

class Parts(Base):
    __tablename__ = 'parts'
    part_number = Column(String(15), primary_key=True)
    description = Column(String(255))
    supplier_name = Column(String(50), index=True)
    price = Column(Float(precision=2))
    cost_per_unit = Column(Float(precision=2))
    __table_args__ = (
        Index('idx_parts_supplier_price', 'supplier_name', 'price'),
        Index('idx_parts_cost', 'cost_per_unit'),
        CheckConstraint('price > 0')
    )
    sales = relationship('Sales', back_populates='parts')

class Inventory(Base):
    __tablename__ = 'inventory'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number', ondelete='CASCADE'), index=True)
    quantity = Column(Integer)
    safety_stock = Column(Integer)
    reorder_point = Column(Integer)
    part_status = Column(Enum(PartStatus))
    negative_on_hand = Column(Integer)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        Index('idx_inventory_status_quantity', 'part_status', 'quantity'),
        Index('idx_inventory_reorder', 'part_number', 'quantity', 'reorder_point'),
        CheckConstraint('quantity >= 0'),
        CheckConstraint('safety_stock >= 0'),
        CheckConstraint('reorder_point >= 0')
    )

class PartMetrics(Base):
    __tablename__ = 'part_metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number', ondelete='CASCADE'), index=True)
    date = Column(Date, index=True)
    roi = Column(Float(precision=4))
    demand = Column(Float(precision=2))
    obsolescence_risk = Column(Float(precision=4))
    sales_volatility = Column(Float(precision=4))
    sales_trend = Column(Float(precision=4))
    recent_sales_trend = Column(Float(precision=4))
    days_of_inventory_outstanding = Column(Integer)
    sell_through_rate = Column(Float(precision=4))
    order_to_sales_ratio = Column(Float(precision=4))
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        Index('idx_metrics_risk_demand', 'obsolescence_risk', 'demand'),
        Index('idx_metrics_part_date', 'part_number', 'date'),
        CheckConstraint('roi >= 0'),
        CheckConstraint('obsolescence_risk >= 0 AND obsolescence_risk <= 1'),
        CheckConstraint('sell_through_rate >= 0 AND sell_through_rate <= 1')
    )

class TemporalMetrics(Base):
    __tablename__ = 'temporal_metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number', ondelete='CASCADE'), index=True)
    months_no_sale = Column(Integer)
    twelve_m_days_supply = Column(Float(precision=2))
    three_m_days_supply = Column(Float(precision=2))
    one_m_days_supply = Column(Float(precision=2))
    three_m_turnover = Column(Float(precision=4))
    turnover = Column(Float(precision=4))
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        Index('idx_temporal_part_turnover', 'part_number', 'turnover'),
        Index('idx_temporal_days_supply', 'part_number', 'twelve_m_days_supply', 'three_m_days_supply', 'one_m_days_supply'),
        CheckConstraint('months_no_sale >= 0'),
        CheckConstraint('twelve_m_days_supply >= 0'),
        CheckConstraint('three_m_days_supply >= 0'),
        CheckConstraint('one_m_days_supply >= 0')
    )

class SupplierOrders(Base):
    __tablename__ = 'supplier_orders'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number', ondelete='CASCADE'), index=True)
    quantity_ordered_ytd = Column(Integer)
    special_orders_ytd = Column(Boolean, default=False)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        Index('idx_supplier_orders_ytd', 'part_number', 'quantity_ordered_ytd'),
        CheckConstraint('quantity_ordered_ytd >= 0')
    )

class Sales(Base):
    __tablename__ = 'sales'
    id = Column(Integer, primary_key=True, autoincrement=True)
    part_number = Column(String(15), ForeignKey('parts.part_number', ondelete='CASCADE'))
    month = Column(String(3)) #use full names
    year = Column(Integer)
    quantity_sold = Column(Integer)
    last_updated = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    __table_args__ = (
        UniqueConstraint('part_number', 'month', 'year', name='unique_sales'),
        Index('idx_sales_part_date', 'part_number', 'year', 'month'),
        Index('idx_sales_quantity', 'quantity_sold'),
        CheckConstraint('quantity_sold >= 0'),
        CheckConstraint('year >= 2000 AND year <= 2100')  # Adjust year range as needed
    )
    parts = relationship('Parts', back_populates='sales')


def create_in_memory_db(data_path):
    engine = create_engine('sqlite:///:memory:', echo=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    with open(data_path, 'r') as f:
        sql_ready_data = json.load(f)

    table_classes = {
        'parts': Parts,
        'inventory': Inventory,
        'part_metrics': PartMetrics,
        'temporal_metrics': TemporalMetrics,
        'supplier_orders': SupplierOrders,
        'sales': Sales
    }

    for table_name, columns in sql_ready_data.items():
        if table_name in table_classes:
            # Restructure the data
            data = []
            for i in range(len(next(iter(columns.values())))):  # Assume all columns have the same length
                row = {}
                for column, values in columns.items():
                    row[column] = values[i]
                data.append(row)
            
            try:
                session.bulk_insert_mappings(table_classes[table_name], data)
                print(f"Inserted {len(data)} rows into {table_name}")
            except Exception as e:
                print(f"Error inserting data into {table_name}: {str(e)}")
                # Optionally, you can choose to continue with other tables or raise the exception
                raise e

    session.commit()
    return engine, session


def save_db_to_file(engine, file_path):
    in_memory_con = engine.raw_connection()
    file_con = sqlite3.connect(file_path)
    in_memory_con.backup(file_con)
    file_con.close()

def main(json_file_path):
    engine, session = create_in_memory_db(json_file_path)
    
    # Define the output file path
    output_file = '/Users/skylerwilson/Desktop/PartsWise/co-pilot-v1/data/databases/partswise_database.db'
    
    # Save the in-memory database to a file
    save_db_to_file(engine, output_file)
    
    session.close()
    
    # Verify that the file was created
    if os.path.exists(output_file):
        print(f"Database saved successfully to {output_file}")
        return output_file
    else:
        print("Failed to save database")
        return None