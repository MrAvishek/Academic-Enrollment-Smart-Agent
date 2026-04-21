from app.db.connection import get_db

con = get_db()
print(" DB Connected")
con.close()