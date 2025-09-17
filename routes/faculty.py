# ===== routes/faculty.py - Faculty CRUD Routes =====
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from models.database import db, Faculty, Department
from sqlalchemy import func
from datetime import datetime
import logging

faculty_bp = Blueprint('faculty', __name__)
logger = logging.getLogger(__name__)

# === API ROUTES ===

@faculty_bp.route('/api/faculties', methods=['GET'])
def api_list_faculties():
    """API: Liste tüm fakülteleri (department sayıları ile)"""
    try:
        faculties = Faculty.query.filter_by(is_active=True).all()
        
        # Her fakülte için department sayısını ekle
        result = []
        for faculty in faculties:
            faculty_data = faculty.to_dict()
            faculty_data['active_department_count'] = Department.query.filter_by(
                faculty_id=faculty.id, 
                is_active=True
            ).count()
            result.append(faculty_data)
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(result)
        })
    
    except Exception as e:
        logger.error(f"Error listing faculties: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch faculties',
            'message': str(e)
        }), 500

@faculty_bp.route('/api/faculties/<int:faculty_id>', methods=['GET'])
def api_get_faculty(faculty_id):
    """API: Belirli bir fakülteyi getir"""
    try:
        faculty = Faculty.query.get(faculty_id)
        if not faculty:
            return jsonify({
                'success': False,
                'error': 'Faculty not found'
            }), 404
        
        # Fakülte bilgilerini ve bölümlerini getir
        faculty_data = faculty.to_dict()
        faculty_data['departments'] = [dept.to_dict() for dept in faculty.departments if dept.is_active]
        
        return jsonify({
            'success': True,
            'data': faculty_data
        })
    
    except Exception as e:
        logger.error(f"Error fetching faculty {faculty_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch faculty',
            'message': str(e)
        }), 500

@faculty_bp.route('/api/faculties', methods=['POST'])
def api_create_faculty():
    """API: Yeni fakülte oluştur"""
    try:
        data = request.get_json()
        
        # Validation
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        required_fields = ['name', 'code']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False,
                    'error': f'Field {field} is required'
                }), 400
        
        # Unique kontrolü
        existing_name = Faculty.query.filter_by(name=data['name']).first()
        if existing_name:
            return jsonify({
                'success': False,
                'error': 'Faculty name already exists'
            }), 409
        
        existing_code = Faculty.query.filter_by(code=data['code'].upper()).first()
        if existing_code:
            return jsonify({
                'success': False,
                'error': 'Faculty code already exists'
            }), 409
        
        # Yeni fakülte oluştur
        faculty = Faculty(
            name=data['name'].strip(),
            code=data['code'].strip().upper(),
            building=data.get('building', '').strip() if data.get('building') else None
        )
        
        db.session.add(faculty)
        db.session.commit()
        
        logger.info(f"Faculty created: {faculty.name} ({faculty.code})")
        
        return jsonify({
            'success': True,
            'message': 'Faculty created successfully',
            'data': faculty.to_dict()
        }), 201
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating faculty: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to create faculty',
            'message': str(e)
        }), 500

@faculty_bp.route('/api/faculties/<int:faculty_id>', methods=['PUT'])
def api_update_faculty(faculty_id):
    """API: Fakülteyi güncelle"""
    try:
        faculty = Faculty.query.get(faculty_id)
        if not faculty:
            return jsonify({
                'success': False,
                'error': 'Faculty not found'
            }), 404
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Name güncellemesi ve unique kontrolü
        if 'name' in data and data['name'].strip() != faculty.name:
            existing_name = Faculty.query.filter(
                Faculty.name == data['name'].strip(),
                Faculty.id != faculty_id
            ).first()
            if existing_name:
                return jsonify({
                    'success': False,
                    'error': 'Faculty name already exists'
                }), 409
            faculty.name = data['name'].strip()
        
        # Code güncellemesi ve unique kontrolü
        if 'code' in data and data['code'].strip().upper() != faculty.code:
            existing_code = Faculty.query.filter(
                Faculty.code == data['code'].strip().upper(),
                Faculty.id != faculty_id
            ).first()
            if existing_code:
                return jsonify({
                    'success': False,
                    'error': 'Faculty code already exists'
                }), 409
            faculty.code = data['code'].strip().upper()
        
        # Building güncellemesi
        if 'building' in data:
            faculty.building = data['building'].strip() if data['building'] else None
        
        # is_active güncellemesi
        if 'is_active' in data:
            faculty.is_active = bool(data['is_active'])
        
        faculty.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Faculty updated: {faculty.name} ({faculty.code})")
        
        return jsonify({
            'success': True,
            'message': 'Faculty updated successfully',
            'data': faculty.to_dict()
        })
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating faculty {faculty_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to update faculty',
            'message': str(e)
        }), 500

@faculty_bp.route('/api/faculties/<int:faculty_id>', methods=['DELETE'])
def api_delete_faculty(faculty_id):
    """API: Fakülteyi sil (soft delete)"""
    try:
        faculty = Faculty.query.get(faculty_id)
        if not faculty:
            return jsonify({
                'success': False,
                'error': 'Faculty not found'
            }), 404
        
        # Bölüm kontrolü - aktif bölüm varsa silinemez
        active_departments = Department.query.filter_by(
            faculty_id=faculty_id, 
            is_active=True
        ).count()
        
        if active_departments > 0:
            return jsonify({
                'success': False,
                'error': f'Cannot delete faculty with {active_departments} active departments'
            }), 409
        
        # Soft delete
        faculty.is_active = False
        faculty.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info(f"Faculty soft deleted: {faculty.name} ({faculty.code})")
        
        return jsonify({
            'success': True,
            'message': 'Faculty deleted successfully'
        })
    
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting faculty {faculty_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to delete faculty',
            'message': str(e)
        }), 500

@faculty_bp.route('/api/faculties/<int:faculty_id>/departments', methods=['GET'])
def api_get_faculty_departments(faculty_id):
    """API: Fakülteye ait bölümleri getir"""
    try:
        faculty = Faculty.query.get(faculty_id)
        if not faculty:
            return jsonify({
                'success': False,
                'error': 'Faculty not found'
            }), 404
        
        departments = Department.query.filter_by(
            faculty_id=faculty_id, 
            is_active=True
        ).order_by(Department.name).all()
        
        return jsonify({
            'success': True,
            'data': [dept.to_dict() for dept in departments],
            'count': len(departments),
            'faculty': {
                'id': faculty.id,
                'name': faculty.name,
                'code': faculty.code
            }
        })
    
    except Exception as e:
        logger.error(f"Error fetching departments for faculty {faculty_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch departments',
            'message': str(e)
        }), 500

# === WEB UI ROUTES ===

@faculty_bp.route('/')
def index():
    """Fakülte yönetimi ana sayfası"""
    try:
        faculties = Faculty.query.filter_by(is_active=True).order_by(Faculty.name).all()
        
        # Her fakülte için istatistikleri hesapla
        faculty_stats = []
        for faculty in faculties:
            dept_count = Department.query.filter_by(faculty_id=faculty.id, is_active=True).count()
            faculty_stats.append({
                'faculty': faculty,
                'department_count': dept_count
            })
        
        return render_template('faculty/index.html', faculty_stats=faculty_stats)
    
    except Exception as e:
        logger.error(f"Error loading faculty index: {e}")
        flash('Error loading faculties', 'error')
        return render_template('faculty/index.html', faculty_stats=[])

@faculty_bp.route('/create')
def create():
    """Yeni fakülte oluşturma sayfası"""
    return render_template('faculty/create.html')

@faculty_bp.route('/edit/<int:faculty_id>')
def edit(faculty_id):
    """Fakülte düzenleme sayfası"""
    try:
        faculty = Faculty.query.get_or_404(faculty_id)
        departments = Department.query.filter_by(faculty_id=faculty_id).all()
        
        return render_template('faculty/edit.html', faculty=faculty, departments=departments)
    
    except Exception as e:
        logger.error(f"Error loading faculty edit page: {e}")
        flash('Faculty not found', 'error')
        return redirect(url_for('faculty.index'))

@faculty_bp.route('/view/<int:faculty_id>')
def view(faculty_id):
    """Fakülte detay görüntüleme sayfası"""
    try:
        faculty = Faculty.query.get_or_404(faculty_id)
        departments = Department.query.filter_by(
            faculty_id=faculty_id, 
            is_active=True
        ).order_by(Department.name).all()
        
        # İstatistikler
        total_lessons = 0
        total_instructors = 0
        
        for dept in departments:
            total_lessons += len([l for l in dept.lessons if l.is_active])
            total_instructors += len([i for i in dept.instructors if i.is_active])
        
        stats = {
            'total_departments': len(departments),
            'total_lessons': total_lessons,
            'total_instructors': total_instructors
        }
        
        return render_template('faculty/view.html', faculty=faculty, departments=departments, stats=stats)
    
    except Exception as e:
        logger.error(f"Error loading faculty view page: {e}")
        flash('Faculty not found', 'error')
        return redirect(url_for('faculty.index'))

# === UTILITY ROUTES ===

@faculty_bp.route('/api/faculties/validate', methods=['POST'])
def api_validate_faculty():
    """API: Fakülte verilerini validate et"""
    try:
        data = request.get_json()
        errors = []
        
        if not data.get('name'):
            errors.append('Faculty name is required')
        elif len(data['name'].strip()) < 3:
            errors.append('Faculty name must be at least 3 characters')
        
        if not data.get('code'):
            errors.append('Faculty code is required')
        elif len(data['code'].strip()) < 2:
            errors.append('Faculty code must be at least 2 characters')
        
        # Unique kontrolü
        faculty_id = data.get('id')  # Update için
        if data.get('name'):
            query = Faculty.query.filter_by(name=data['name'].strip())
            if faculty_id:
                query = query.filter(Faculty.id != faculty_id)
            if query.first():
                errors.append('Faculty name already exists')
        
        if data.get('code'):
            query = Faculty.query.filter_by(code=data['code'].strip().upper())
            if faculty_id:
                query = query.filter(Faculty.id != faculty_id)
            if query.first():
                errors.append('Faculty code already exists')
        
        return jsonify({
            'success': len(errors) == 0,
            'errors': errors
        })
    
    except Exception as e:
        logger.error(f"Error validating faculty: {e}")
        return jsonify({
            'success': False,
            'errors': ['Validation failed'],
            'message': str(e)
        }), 500