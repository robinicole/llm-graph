from __future__ import annotations

import os

from sqlalchemy import (
    Column,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = os.environ['PSQL_URL']

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):  # type: ignore
    __tablename__ = 'users'

    '''username: string;
    password: string;
    num_trials: number;
    coverlettertrials: number;
    bookgraphtrials: number;'''
    username = Column(String, unique=True, index=True, primary_key=True)
    num_trials = Column(Integer)
    coverlettertrials = Column(Integer)
    bookgraphtrials = Column(Integer)


def get_db():  # type: ignore
    """
    Get a database session.

    Returns:
        Generator[SessionLocal, None, None]: A generator that yields a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
