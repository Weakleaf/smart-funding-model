"""HELB Smart System — Configuration"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    DATABASE_PATH = os.path.join(BASE_DIR, 'helb_smart.db')
    MODEL_PATH    = os.path.join(BASE_DIR, 'models', 'helb_model.pkl')
    SECRET_KEY    = 'helb-smart-system-2026'
    DEBUG         = True
    API_HOST      = '127.0.0.1'
    API_PORT      = 5000

    BANDS = {
        1: {'label': 'Band 1', 'scholarship': 80, 'loan': 15, 'household':  5, 'color': '#1a7a3c'},
        2: {'label': 'Band 2', 'scholarship': 60, 'loan': 25, 'household': 15, 'color': '#2e9e52'},
        3: {'label': 'Band 3', 'scholarship': 40, 'loan': 35, 'household': 25, 'color': '#f0a500'},
        4: {'label': 'Band 4', 'scholarship': 20, 'loan': 40, 'household': 40, 'color': '#d45f00'},
        5: {'label': 'Band 5', 'scholarship':  0, 'loan': 30, 'household': 70, 'color': '#c0392b'},
    }

    SCHOOL_TYPES  = ['National', 'Extra-County', 'County', 'Sub-County', 'Private']
    REGIONS       = ['Nairobi', 'Central', 'Coast', 'Eastern', 'North Eastern',
                     'Nyanza', 'Rift Valley', 'Western']
    SPONSOR_TYPES = ['Government', 'Self-Sponsored']

    REGION_WEIGHTS = {
        'North Eastern': 1.4, 'Eastern': 1.2, 'Rift Valley': 1.1,
        'Coast': 1.1, 'Western': 1.0, 'Nyanza': 1.0,
        'Central': 0.9, 'Nairobi': 0.8
    }

    @staticmethod
    def create_dirs():
        os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
