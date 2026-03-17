"""HELB Smart System — Database Layer"""
import sqlite3
import pandas as pd
from datetime import datetime
from config import Config

class HELBDatabase:
    def __init__(self):
        self.path = Config.DATABASE_PATH
        self._create_tables()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _create_tables(self):
        conn = self._conn()
        c = conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                id_number TEXT UNIQUE NOT NULL,
                email TEXT,
                phone TEXT,
                institution TEXT,
                course TEXT,
                year_of_study INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                household_income REAL,
                num_dependents INTEGER,
                siblings_in_uni INTEGER,
                orphan_status INTEGER DEFAULT 0,
                disability_status INTEGER DEFAULT 0,
                school_type TEXT,
                region TEXT,
                sponsor_type TEXT,
                annual_fees REAL,
                parental_employment TEXT,
                raw_score REAL,
                adjusted_score REAL,
                assigned_band INTEGER,
                explanation TEXT,
                status TEXT DEFAULT 'Pending',
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            CREATE TABLE IF NOT EXISTS appeals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                application_id INTEGER NOT NULL,
                reason TEXT,
                supporting_docs TEXT,
                status TEXT DEFAULT 'Under Review',
                reviewer_notes TEXT,
                submitted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (application_id) REFERENCES applications(id)
            );
        """)
        conn.commit()
        conn.close()

    def insert_student(self, data):
        conn = self._conn()
        c = conn.cursor()
        c.execute("""INSERT OR IGNORE INTO students
            (full_name, id_number, email, phone, institution, course, year_of_study)
            VALUES (?,?,?,?,?,?,?)""",
            (data['full_name'], data['id_number'], data.get('email',''),
             data.get('phone',''), data.get('institution',''), data.get('course',''),
             data.get('year_of_study', 1)))
        conn.commit()
        sid = c.execute("SELECT id FROM students WHERE id_number=?", (data['id_number'],)).fetchone()[0]
        conn.close()
        return sid

    def insert_application(self, data):
        conn = self._conn()
        c = conn.cursor()
        c.execute("""INSERT INTO applications
            (student_id, household_income, num_dependents, siblings_in_uni,
             orphan_status, disability_status, school_type, region, sponsor_type,
             annual_fees, parental_employment, raw_score, adjusted_score,
             assigned_band, explanation, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (data['student_id'], data['household_income'], data['num_dependents'],
             data['siblings_in_uni'], data['orphan_status'], data['disability_status'],
             data['school_type'], data['region'], data['sponsor_type'],
             data['annual_fees'], data['parental_employment'],
             data['raw_score'], data['adjusted_score'], data['assigned_band'],
             data['explanation'], 'Processed'))
        conn.commit()
        aid = c.lastrowid
        conn.close()
        return aid

    def get_application(self, app_id):
        conn = self._conn()
        df = pd.read_sql_query(
            "SELECT a.*, s.full_name, s.id_number, s.institution FROM applications a "
            "JOIN students s ON a.student_id=s.id WHERE a.id=?", conn, params=(app_id,))
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else None

    def get_all_applications(self):
        conn = self._conn()
        df = pd.read_sql_query(
            "SELECT a.*, s.full_name, s.id_number FROM applications a "
            "JOIN students s ON a.student_id=s.id ORDER BY a.submitted_at DESC", conn)
        conn.close()
        return df

    def insert_appeal(self, data):
        conn = self._conn()
        c = conn.cursor()
        c.execute("INSERT INTO appeals (application_id, reason, supporting_docs) VALUES (?,?,?)",
                  (data['application_id'], data['reason'], data.get('supporting_docs','')))
        conn.commit()
        aid = c.lastrowid
        conn.close()
        return aid

    def get_stats(self):
        conn = self._conn()
        stats = {}
        stats['total_applications'] = pd.read_sql_query("SELECT COUNT(*) as n FROM applications", conn).iloc[0]['n']
        stats['total_students']     = pd.read_sql_query("SELECT COUNT(*) as n FROM students", conn).iloc[0]['n']
        stats['total_appeals']      = pd.read_sql_query("SELECT COUNT(*) as n FROM appeals", conn).iloc[0]['n']
        band_dist = pd.read_sql_query(
            "SELECT assigned_band, COUNT(*) as count FROM applications GROUP BY assigned_band", conn)
        stats['band_distribution'] = band_dist.set_index('assigned_band')['count'].to_dict()
        conn.close()
        return stats
