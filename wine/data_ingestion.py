import pandas as pd
from sqlalchemy import create_engine, text

db_path = "sqlite:///wines.db"
engine = create_engine(db_path)

df = pd.read_csv('wine-ratings.csv')
df.to_sql('wines', engine, if_exists='replace', index=False)