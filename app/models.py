from sqlalchemy import Column, Integer, String, Float, DateTime
from app.db import Base

class ObservationModel(Base):
    __tablename__ = 'observations'
    id = Column(Integer, primary_key=True, index=True)
    series_id = Column(String, index=True)
    observed_at = Column(DateTime)
    value = Column(Float)
