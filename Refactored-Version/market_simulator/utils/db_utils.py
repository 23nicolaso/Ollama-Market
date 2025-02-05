from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from zoneinfo import ZoneInfo
from datetime import datetime
import queue
from threading import Thread, Lock

Base = declarative_base()

class PriceHistory(Base):
    __tablename__ = 'price_history'
    
    id = Column(Integer, primary_key=True)
    asset = Column(String)
    price = Column(Float)
    timestamp = Column(DateTime, default=lambda: datetime.now(ZoneInfo('UTC')))

class DatabaseManager:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self.engine = create_engine('sqlite:///market_history.db')
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        self.write_queue = queue.Queue()
        self.is_running = True
        
        # Start background writer thread
        self.writer_thread = Thread(target=self._process_write_queue, daemon=True)
        self.writer_thread.start()
    
    def init_db(self):
        Base.metadata.create_all(self.engine)
    
    def _process_write_queue(self):
        """Background thread to process database writes"""
        session = self.Session()
        batch = []
        
        while self.is_running:
            try:
                # Collect items for 1 second or until batch size reached
                try:
                    while len(batch) < 1000:  # Max batch size
                        item = self.write_queue.get(timeout=1.0)
                        batch.append(item)
                except queue.Empty:
                    pass
                
                # Process batch if we have any items
                if batch:
                    session.bulk_save_objects(batch)
                    session.commit()
                    batch = []
                    
            except Exception as e:
                print(f"Error in database writer: {e}")
                session.rollback()
                batch = []
    
    def queue_price_update(self, asset, price, timestamp=None):
        """Queue a price update for batch processing"""
        if timestamp is None:
            timestamp = datetime.now(ZoneInfo('UTC'))
        record = PriceHistory(asset=asset, price=price, timestamp=timestamp)
        self.write_queue.put(record)
    
    def get_price_history(self, asset, start_time=None, end_time=None):
        """Get price history from database"""
        session = self.Session()
        try:
            query = session.query(PriceHistory).filter(PriceHistory.asset == asset)
            if start_time:
                query = query.filter(PriceHistory.timestamp >= start_time)
            if end_time:
                query = query.filter(PriceHistory.timestamp <= end_time)
            return [(record.price, record.timestamp) for record in query.all()]
        finally:
            self.Session.remove()  # Changed from session.remove() to self.Session.remove()
    
    def shutdown(self):
        """Gracefully shutdown the database manager"""
        self.is_running = False
        self.writer_thread.join()
        self.Session.remove()
    
    def wipe_db(self):
        """Wipes all data from the database"""
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

# Global instance
db_manager = DatabaseManager()

# Convenience functions
def init_db():
    db_manager.init_db()

def store_price(asset, price, timestamp=None):
    db_manager.queue_price_update(asset, price, timestamp)

def get_price_history(asset, start_time=None, end_time=None):
    return db_manager.get_price_history(asset, start_time, end_time)

def shutdown_db():
    db_manager.shutdown()

def wipe_db():
    """Convenience function to wipe the database"""
    db_manager.wipe_db()