from .connection import get_connection
from datetime import datetime, timedelta

class AttendanceRepository:
    def log_presence(self, student_id, confidence):
        conn = get_connection()
        cursor = conn.cursor()
        
        # AGENTIC CHECK: Has this student been logged in the last 30 minutes?
        time_threshold = datetime.now() - timedelta(minutes=30)
        check_query = "SELECT id FROM attendance_logs WHERE student_id = %s AND check_in_time > %s"
        cursor.execute(check_query, (student_id, time_threshold))
        
        if cursor.fetchone() is None:
            # If not logged recently, create a new entry
            insert_query = "INSERT INTO attendance_logs (student_id, confidence_score) VALUES (%s, %s)"
            cursor.execute(insert_query, (student_id, confidence))
            conn.commit()
            print(f"✔️ Database updated for {student_id}")
            
        cursor.close()
        conn.close()