"""HELB Smart System — Flask API"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from config import Config
from database import HELBDatabase
from model import HELBAllocationModel

app  = Flask(__name__)
CORS(app)

db    = HELBDatabase()
model = HELBAllocationModel()

# Load or train model on startup
if not model.load():
    print("No saved model found — training now...")
    model.train()


@app.route('/')
def home():
    return jsonify({'system': 'HELB Smart Allocation System', 'version': '1.0', 'status': 'running'})


@app.route('/api/apply', methods=['POST'])
def apply():
    """Submit a new funding application."""
    try:
        d = request.get_json()

        # Save student
        sid = db.insert_student({
            'full_name':     d['full_name'],
            'id_number':     d['id_number'],
            'email':         d.get('email', ''),
            'phone':         d.get('phone', ''),
            'institution':   d.get('institution', ''),
            'course':        d.get('course', ''),
            'year_of_study': d.get('year_of_study', 1),
        })

        # Run allocation model
        applicant = {
            'household_income':   float(d['household_income']),
            'num_dependents':     int(d['num_dependents']),
            'siblings_in_uni':    int(d.get('siblings_in_uni', 0)),
            'orphan_status':      int(d.get('orphan_status', 0)),
            'disability_status':  int(d.get('disability_status', 0)),
            'school_type':        d.get('school_type', 'County'),
            'region':             d.get('region', 'Nairobi'),
            'sponsor_type':       d.get('sponsor_type', 'Government'),
            'annual_fees':        float(d.get('annual_fees', 80000)),
            'parental_employment':d.get('parental_employment', 'Informal'),
        }

        result = model.predict_band(applicant)

        # Save application
        app_id = db.insert_application({
            'student_id':        sid,
            'raw_score':         result['raw_score'],
            'adjusted_score':    result['adjusted_score'],
            'assigned_band':     result['band'],
            'explanation':       result['explanation'],
            **applicant
        })

        return jsonify({
            'success':        True,
            'application_id': app_id,
            'band':           result['band'],
            'band_info':      result['band_info'],
            'probabilities':  result['probabilities'],
            'explanation':    result['explanation'],
            'raw_score':      result['raw_score'],
            'adjusted_score': result['adjusted_score'],
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/result/<int:app_id>', methods=['GET'])
def get_result(app_id):
    """Fetch result for a specific application."""
    try:
        rec = db.get_application(app_id)
        if not rec:
            return jsonify({'error': 'Application not found'}), 404
        return jsonify({'success': True, 'data': rec})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/appeal', methods=['POST'])
def submit_appeal():
    """Submit an appeal for band review."""
    try:
        d = request.get_json()
        aid = db.insert_appeal({
            'application_id':  d['application_id'],
            'reason':          d['reason'],
            'supporting_docs': d.get('supporting_docs', ''),
        })
        return jsonify({'success': True, 'appeal_id': aid,
                        'message': 'Appeal submitted successfully. You will be notified within 14 working days.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Admin statistics."""
    try:
        return jsonify({'success': True, 'data': db.get_stats()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/applications', methods=['GET'])
def all_applications():
    """Admin: all applications."""
    try:
        df = db.get_all_applications()
        return jsonify({'success': True, 'count': len(df), 'data': df.to_dict('records')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model.is_trained,
                    'timestamp': datetime.now().isoformat()})


if __name__ == '__main__':
    print("=" * 55)
    print("  HELB Smart Allocation System — API Server")
    print(f"  Running on http://{Config.API_HOST}:{Config.API_PORT}")
    print("=" * 55)
    app.run(host=Config.API_HOST, port=Config.API_PORT, debug=Config.DEBUG)
