from .connection import get_connection

class StudentRepository:
    def add_student(self, student_id, name):
        conn = get_connection()
        cursor = conn.cursor()
        query = "INSERT IGNORE INTO students (student_id, full_name) VALUES (%s, %s)"
        cursor.execute(query, (student_id, name))
        conn.commit()
        cursor.close()
        conn.close()

    def get_all_students(self):
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students")
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        return students