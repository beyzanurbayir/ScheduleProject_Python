# ===== routes/data_entry.py - Güncellenmiş =====
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from models.database import Faculty, db, Department, Lesson, Instructor, Classroom, InstructorLesson, ClassroomAvailability
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, distinct
import json
from datetime import datetime

data_entry_bp = Blueprint('data_entry', __name__)

@data_entry_bp.route('/')
def index():
    departments = Department.query.all()
    return render_template('data_entry/index.html', departments=departments)

@data_entry_bp.route('/departments')
def departments():
    # YENİ: Faculty filtreleme desteği
    faculty_id = request.args.get('faculty_id', type=int)
    
    # Base query
    query = Department.query
    
    # Faculty filter
    if faculty_id:
        query = query.filter_by(faculty_id=faculty_id)
    
    departments = query.order_by(Department.name).all()
    faculties = Faculty.query.filter_by(is_active=True).order_by(Faculty.name).all()  # YENİ: Faculty listesi
    
    # Department statistics by faculty
    department_stats = []
    for dept in departments:
        lesson_count = Lesson.query.filter_by(department_id=dept.id, is_active=True).count()
        instructor_count = Instructor.query.filter_by(department_id=dept.id, is_active=True).count()
        
        department_stats.append({
            'department': dept,
            'lesson_count': lesson_count,
            'instructor_count': instructor_count
        })
    
    return render_template('data_entry/departments.html', 
                         department_stats=department_stats,  # YENİ: stats yerine departments
                         faculties=faculties,  # YENİ: Faculty listesi
                         selected_faculty=faculty_id)  # YENİ: Seçili faculty

@data_entry_bp.route('/departments/create', methods=['GET', 'POST'])
def create_department():
    if request.method == 'POST':
        try:
            name = request.form['name']
            code = request.form.get('code', '').upper()
            faculty_id = int(request.form['faculty_id'])  # YENİ: Faculty seçimi zorunlu
            num_grades = int(request.form['num_grades'])
            
            # Faculty kontrolü
            faculty = Faculty.query.get(faculty_id)
            if not faculty or not faculty.is_active:
                flash('Selected faculty not found or inactive!', 'error')
                raise ValueError("Invalid faculty")
            
            # Auto-generate code if not provided
            if not code:
                words = name.split()
                if len(words) >= 2:
                    code = ''.join([word[0].upper() for word in words[:2]])
                else:
                    code = name[:3].upper()
            
            # YENİ: Code uniqueness check within faculty
            existing_dept = Department.query.filter_by(faculty_id=faculty_id, code=code).first()
            if existing_dept:
                flash(f'Department code "{code}" already exists in {faculty.name}!', 'error')
                raise ValueError("Duplicate code")
            
            department = Department(
                name=name,
                code=code,
                faculty_id=faculty_id,  # YENİ: Faculty assignment
                num_grades=num_grades,
                head_of_department=request.form.get('head_of_department'),
                building=request.form.get('building'),
                floor=int(request.form['floor']) if request.form.get('floor') else None,
                phone=request.form.get('phone'),
                email=request.form.get('email')
            )
            
            db.session.add(department)
            db.session.commit()
            
            flash(f'Department "{name}" created successfully in {faculty.name}!', 'success')
            return redirect(url_for('data_entry.departments'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error creating department: {str(e)}', 'error')
    
    # YENİ: Active faculties for selection
    faculties = Faculty.query.filter_by(is_active=True).order_by(Faculty.name).all()
    
    if not faculties:
        flash('No active faculties found. Please create a faculty first!', 'warning')
        return redirect(url_for('faculty.index'))  # Redirect to faculty management
    
    return render_template('data_entry/create_department.html', faculties=faculties)

@data_entry_bp.route('/departments/<int:dept_id>/edit', methods=['GET', 'POST'])
def edit_department(dept_id):
    department = Department.query.get_or_404(dept_id)
    
    if request.method == 'POST':
        try:
            department.name = request.form['name']
            new_code = request.form.get('code', '').upper()
            new_faculty_id = int(request.form['faculty_id'])  # YENİ: Faculty değiştirilebilir
            department.num_grades = int(request.form['num_grades'])
            
            # Faculty kontrolü
            faculty = Faculty.query.get(new_faculty_id)
            if not faculty or not faculty.is_active:
                flash('Selected faculty not found or inactive!', 'error')
                raise ValueError("Invalid faculty")
            
            # YENİ: Code uniqueness check within new faculty (if changed)
            if new_code != department.code or new_faculty_id != department.faculty_id:
                existing_dept = Department.query.filter_by(
                    faculty_id=new_faculty_id, 
                    code=new_code
                ).filter(Department.id != dept_id).first()
                
                if existing_dept:
                    flash(f'Department code "{new_code}" already exists in {faculty.name}!', 'error')
                    raise ValueError("Duplicate code")
            
            # Assignments
            department.code = new_code
            department.faculty_id = new_faculty_id  # YENİ: Faculty güncellemesi
            department.head_of_department = request.form.get('head_of_department')
            department.building = request.form.get('building')
            department.floor = int(request.form['floor']) if request.form.get('floor') else None
            department.phone = request.form.get('phone')
            department.email = request.form.get('email')
            department.is_active = bool(request.form.get('is_active'))
            department.updated_at = datetime.utcnow()  # YENİ: Update timestamp
            
            db.session.commit()
            flash(f'Department "{department.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.departments'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error updating department: {str(e)}', 'error')
    
    # YENİ: Active faculties for selection
    faculties = Faculty.query.filter_by(is_active=True).order_by(Faculty.name).all()
    
    return render_template('data_entry/edit_department.html', 
                         department=department, 
                         faculties=faculties)

@data_entry_bp.route('/departments/<int:dept_id>/delete', methods=['POST'])
def delete_department(dept_id):
    department = Department.query.get_or_404(dept_id)
    
    try:
        # YENİ: Dependency kontrolü - daha ayrıntılı
        lesson_count = Lesson.query.filter_by(department_id=dept_id, is_active=True).count()
        instructor_count = Instructor.query.filter_by(department_id=dept_id, is_active=True).count()
        
        if lesson_count > 0:
            flash(f'Cannot delete department "{department.name}": {lesson_count} active lessons exist!', 'error')
            return redirect(url_for('data_entry.departments'))
        
        if instructor_count > 0:
            flash(f'Cannot delete department "{department.name}": {instructor_count} active instructors exist!', 'error')
            return redirect(url_for('data_entry.departments'))
        
        faculty_name = department.faculty_ref.name
        dept_name = department.name
        
        db.session.delete(department)
        db.session.commit()
        
        flash(f'Department "{dept_name}" deleted successfully from {faculty_name}!', 'success')
        
    except Exception as e:
        flash(f'Error deleting department: {str(e)}', 'error')
        db.session.rollback()
        
    return redirect(url_for('data_entry.departments'))

# YENİ: API endpoint for faculty-department hierarchy
@data_entry_bp.route('/api/faculties/<int:faculty_id>/departments', methods=['GET'])
def api_get_faculty_departments(faculty_id):
    """API: Belirli bir fakülteye ait aktif bölümleri getir"""
    try:
        faculty = Faculty.query.get(faculty_id)
        if not faculty or not faculty.is_active:
            return jsonify({
                'success': False,
                'error': 'Faculty not found or inactive'
            }), 404
        
        departments = Department.query.filter_by(
            faculty_id=faculty_id,
            is_active=True
        ).order_by(Department.name).all()
        
        return jsonify({
            'success': True,
            'data': [dept.to_dict() for dept in departments],
            'faculty': faculty.to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Failed to fetch departments',
            'message': str(e)
        }), 500

# YENİ: Quick stats API
@data_entry_bp.route('/api/departments/stats', methods=['GET'])
def api_department_stats():
    """API: Department istatistikleri"""
    try:
        # Faculty başına department sayıları
        faculty_stats = db.session.query(
            Faculty.name.label('faculty_name'),
            func.count(Department.id).label('department_count'),
            func.count(Lesson.id).label('total_lessons'),
            func.count(Instructor.id).label('total_instructors')
        ).outerjoin(Department, Faculty.id == Department.faculty_id) \
         .outerjoin(Lesson, Department.id == Lesson.department_id) \
         .outerjoin(Instructor, Department.id == Instructor.department_id) \
         .filter(Faculty.is_active == True) \
         .group_by(Faculty.id, Faculty.name) \
         .all()
        
        stats = []
        for stat in faculty_stats:
            stats.append({
                'faculty_name': stat.faculty_name,
                'department_count': stat.department_count,
                'total_lessons': stat.total_lessons or 0,
                'total_instructors': stat.total_instructors or 0
            })
        
        return jsonify({
            'success': True,
            'data': stats,
            'total_departments': sum(s['department_count'] for s in stats)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Failed to fetch statistics',
            'message': str(e)
        }), 500


# ... (diğer department routes aynı kalacak) ...

@data_entry_bp.route('/classrooms')
def classrooms():
    page = request.args.get('page', 1, type=int)
    building = request.args.get('building')
    floor = request.args.get('floor', type=int)
    classroom_type = request.args.get('classroom_type')
    min_capacity = request.args.get('min_capacity', type=int)
    features = request.args.get('features')
    
    query = Classroom.query
    
    # Apply filters
    if building:
        query = query.filter_by(building=building)
    if floor is not None:
        query = query.filter_by(floor=floor)
    if classroom_type:
        query = query.filter_by(classroom_type=classroom_type)
    if min_capacity:
        query = query.filter(Classroom.capacity >= min_capacity)
    if features:
        if features == 'projector':
            query = query.filter_by(has_projector=True)
        elif features == 'computer':
            query = query.filter_by(has_computer=True)
        elif features == 'lab':
            query = query.filter_by(has_lab=True)
        elif features == 'smartboard':
            query = query.filter_by(has_smartboard=True)
        elif features == 'ac':
            query = query.filter_by(has_air_conditioning=True)
    
    # Pagination
    classrooms_paginated = query.order_by(Classroom.building, Classroom.floor, Classroom.code).paginate(
        page=page, per_page=20, error_out=False
    )
    
    # Get filter options
    buildings = db.session.query(distinct(Classroom.building)).filter(Classroom.building.isnot(None)).all()
    buildings = [b[0] for b in buildings]
    
    floors = db.session.query(distinct(Classroom.floor)).filter(Classroom.floor.isnot(None)).all()
    floors = sorted([f[0] for f in floors])
    
    # Statistics
    total_classrooms = Classroom.query.count()
    active_classrooms = Classroom.query.filter_by(is_active=True).count()
    total_capacity = db.session.query(func.sum(Classroom.capacity)).scalar() or 0
    avg_capacity = db.session.query(func.avg(Classroom.capacity)).scalar() or 0
    lab_count = Classroom.query.filter_by(has_lab=True).count()
    
    return render_template('data_entry/classrooms.html', 
                         classrooms=classrooms_paginated.items,
                         pagination=classrooms_paginated,
                         buildings=buildings,
                         floors=floors,
                         selected_building=building,
                         selected_floor=floor,
                         selected_type=classroom_type,
                         selected_feature=features,
                         min_capacity=min_capacity,
                         total_classrooms=total_classrooms,
                         active_classrooms=active_classrooms,
                         total_capacity=total_capacity,
                         avg_capacity=avg_capacity,
                         lab_count=lab_count)

@data_entry_bp.route('/classrooms/create', methods=['GET', 'POST'])
def create_classroom():
    if request.method == 'POST':
        try:
            # Parse technical equipment
            technical_equipment = request.form.get('technical_equipment', '').strip()
            if technical_equipment:
                try:
                    # Try to parse as JSON (from JavaScript)
                    equipment_list = json.loads(technical_equipment)
                except json.JSONDecodeError:
                    # Parse as plain text (newline separated)
                    equipment_list = [item.strip() for item in technical_equipment.split('\n') if item.strip()]
            else:
                equipment_list = []
            
            # Parse department priority
            dept_priority = request.form.getlist('department_priority')
            dept_priority_ids = [int(dp) for dp in dept_priority if dp]
            
            classroom = Classroom(
                name=request.form['name'],
                code=request.form['code'].upper(),
                capacity=int(request.form['capacity']),
                exam_capacity=int(request.form['exam_capacity']) if request.form.get('exam_capacity') else None,
                classroom_type=request.form['classroom_type'],
                has_lab=bool(request.form.get('has_lab')),
                has_projector=bool(request.form.get('has_projector')),
                has_computer=bool(request.form.get('has_computer')),
                has_smartboard=bool(request.form.get('has_smartboard')),
                has_air_conditioning=bool(request.form.get('has_air_conditioning')),
                has_microphone=bool(request.form.get('has_microphone')),
                wifi_available=bool(request.form.get('wifi_available')),
                building=request.form['building'],
                floor=int(request.form['floor']),
                room_number=request.form.get('room_number'),
                setup_time_minutes=int(request.form.get('setup_time_minutes', 5)),
                cleaning_time_minutes=int(request.form.get('cleaning_time_minutes', 15)),
                usage_cost_per_hour=float(request.form['usage_cost_per_hour']) if request.form.get('usage_cost_per_hour') else None,
                technical_equipment=equipment_list if equipment_list else None,
                department_priority=dept_priority_ids if dept_priority_ids else None,
                notes=request.form.get('notes'),
                is_active=bool(request.form.get('is_active', True)),
                is_bookable=bool(request.form.get('is_bookable', True))
            )
            
            db.session.add(classroom)
            db.session.commit()
            
            flash(f'Classroom "{classroom.name}" created successfully!', 'success')
            return redirect(url_for('data_entry.classrooms'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error creating classroom: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    return render_template('data_entry/create_classroom.html', departments=departments)

@data_entry_bp.route('/classrooms/<int:classroom_id>/edit', methods=['GET', 'POST'])
def edit_classroom(classroom_id):
    classroom = Classroom.query.get_or_404(classroom_id)
    
    if request.method == 'POST':
        try:
            # Parse technical equipment
            technical_equipment = request.form.get('technical_equipment', '').strip()
            if technical_equipment:
                try:
                    equipment_list = json.loads(technical_equipment)
                except json.JSONDecodeError:
                    equipment_list = [item.strip() for item in technical_equipment.split('\n') if item.strip()]
            else:
                equipment_list = []
            
            # Parse department priority
            dept_priority = request.form.getlist('department_priority')
            dept_priority_ids = [int(dp) for dp in dept_priority if dp]
            
            classroom.name = request.form['name']
            classroom.code = request.form['code'].upper()
            classroom.capacity = int(request.form['capacity'])
            classroom.exam_capacity = int(request.form['exam_capacity']) if request.form.get('exam_capacity') else None
            classroom.classroom_type = request.form['classroom_type']
            classroom.has_lab = bool(request.form.get('has_lab'))
            classroom.has_projector = bool(request.form.get('has_projector'))
            classroom.has_computer = bool(request.form.get('has_computer'))
            classroom.has_smartboard = bool(request.form.get('has_smartboard'))
            classroom.has_air_conditioning = bool(request.form.get('has_air_conditioning'))
            classroom.has_microphone = bool(request.form.get('has_microphone'))
            classroom.wifi_available = bool(request.form.get('wifi_available'))
            classroom.building = request.form['building']
            classroom.floor = int(request.form['floor'])
            classroom.room_number = request.form.get('room_number')
            classroom.setup_time_minutes = int(request.form.get('setup_time_minutes', 5))
            classroom.cleaning_time_minutes = int(request.form.get('cleaning_time_minutes', 15))
            classroom.usage_cost_per_hour = float(request.form['usage_cost_per_hour']) if request.form.get('usage_cost_per_hour') else None
            classroom.technical_equipment = equipment_list if equipment_list else None
            classroom.department_priority = dept_priority_ids if dept_priority_ids else None
            classroom.notes = request.form.get('notes')
            classroom.is_active = bool(request.form.get('is_active'))
            classroom.is_bookable = bool(request.form.get('is_bookable'))
            
            db.session.commit()
            flash(f'Classroom "{classroom.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.classrooms'))
            
        except ValueError as e:
            flash(f'Error updating classroom: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    
    # Prepare technical equipment for display
    equipment_text = ''
    if classroom.technical_equipment:
        equipment_text = '\n'.join(classroom.technical_equipment)
    
    return render_template('data_entry/edit_classroom.html', 
                         classroom=classroom, 
                         departments=departments,
                         equipment_text=equipment_text)

@data_entry_bp.route('/classrooms/<int:classroom_id>/delete', methods=['POST'])
def delete_classroom(classroom_id):
    classroom = Classroom.query.get_or_404(classroom_id)
    try:
        db.session.delete(classroom)
        db.session.commit()
        flash(f'Classroom "{classroom.name}" deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting classroom: {str(e)}', 'error')
    
    return redirect(url_for('data_entry.classrooms'))

@data_entry_bp.route('/classrooms/<int:classroom_id>/details')
def classroom_details(classroom_id):
    """API endpoint for classroom details modal"""
    classroom = Classroom.query.get_or_404(classroom_id)
    return jsonify(classroom.to_dict())

@data_entry_bp.route('/classrooms/<int:classroom_id>/duplicate', methods=['GET', 'POST'])
def duplicate_classroom(classroom_id):
    original = Classroom.query.get_or_404(classroom_id)
    
    if request.method == 'POST':
        try:
            new_classroom = Classroom(
                name=request.form['name'],
                code=request.form['code'].upper(),
                capacity=original.capacity,
                exam_capacity=original.exam_capacity,
                classroom_type=original.classroom_type,
                has_lab=original.has_lab,
                has_projector=original.has_projector,
                has_computer=original.has_computer,
                has_smartboard=original.has_smartboard,
                has_air_conditioning=original.has_air_conditioning,
                has_microphone=original.has_microphone,
                wifi_available=original.wifi_available,
                building=request.form['building'],
                floor=int(request.form['floor']),
                room_number=request.form.get('room_number'),
                setup_time_minutes=original.setup_time_minutes,
                cleaning_time_minutes=original.cleaning_time_minutes,
                usage_cost_per_hour=original.usage_cost_per_hour,
                technical_equipment=original.technical_equipment.copy() if original.technical_equipment else None,
                department_priority=original.department_priority.copy() if original.department_priority else None,
                notes=request.form.get('notes'),
                is_active=True,
                is_bookable=True
            )
            
            db.session.add(new_classroom)
            db.session.commit()
            
            flash(f'Classroom "{new_classroom.name}" duplicated successfully!', 'success')
            return redirect(url_for('data_entry.classrooms'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error duplicating classroom: {str(e)}', 'error')
    
    return render_template('data_entry/duplicate_classroom.html', original=original)

@data_entry_bp.route('/classrooms/<int:classroom_id>/maintenance', methods=['GET', 'POST'])
def schedule_maintenance(classroom_id):
    classroom = Classroom.query.get_or_404(classroom_id)
    
    if request.method == 'POST':
        try:
            start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
            end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d')
            
            maintenance = ClassroomAvailability(
                classroom_id=classroom_id,
                date_start=start_date,
                date_end=end_date,
                availability_type='maintenance',
                reason=request.form.get('reason', 'Scheduled maintenance'),
                contact_person=request.form.get('contact_person')
            )
            
            # Update classroom maintenance dates
            classroom.last_maintenance_date = start_date
            classroom.next_maintenance_date = end_date
            
            db.session.add(maintenance)
            db.session.commit()
            
            flash(f'Maintenance scheduled for "{classroom.name}"!', 'success')
            return redirect(url_for('data_entry.classrooms'))
            
        except ValueError as e:
            flash(f'Error scheduling maintenance: {str(e)}', 'error')
    
    return render_template('data_entry/schedule_maintenance.html', classroom=classroom)

# Lessons routes (güncellenmiş)
@data_entry_bp.route('/lessons')
def lessons():
    dept_id = request.args.get('department_id', type=int)
    grade = request.args.get('grade', type=int)
    semester = request.args.get('semester', type=int)
    
    query = Lesson.query
    if dept_id:
        query = query.filter_by(department_id=dept_id)
    if grade:
        query = query.filter_by(grade=grade)
    if semester:
        query = query.filter_by(semester=semester)

    lessons = query.order_by(Lesson.department_id, Lesson.grade, Lesson.semester, Lesson.name).all()
    departments = Department.query.order_by(Department.name).all()
    
    return render_template('data_entry/lessons.html', 
                         lessons=lessons, 
                         departments=departments,
                         selected_dept=dept_id,
                         selected_grade=grade,
                         selected_semester=semester)

@data_entry_bp.route('/lessons/create', methods=['GET', 'POST'])
def create_lesson():
    if request.method == 'POST':
        try:
            # Parse prerequisite IDs
            prerequisite_str = request.form.get('prerequisite_ids', '').strip()
            prerequisite_ids = []
            if prerequisite_str:
                try:
                    prerequisite_ids = [int(x.strip()) for x in prerequisite_str.split(',') if x.strip()]
                except ValueError:
                    flash('Invalid prerequisite IDs format. Use comma-separated numbers.', 'error')
                    return render_template('data_entry/create_lesson.html', departments=Department.query.all())
            
            lesson = Lesson(
                name=request.form['name'],
                code=request.form['code'].upper(),
                department_id=int(request.form['department_id']),
                grade=int(request.form['grade']),
                semester=int(request.form.get('semester', 1)),
                theory_hours=int(request.form.get('theory_hours', 0)),
                practice_hours=int(request.form.get('practice_hours', 0)),
                lab_hours=int(request.form.get('lab_hours', 0)),
                akts=int(request.form.get('akts', 0)),
                local_credit=float(request.form.get('local_credit', 0)),
                student_capacity=int(request.form['student_capacity']),
                min_capacity=int(request.form.get('min_capacity', 5)),
                difficulty=int(request.form.get('difficulty', 3)), # YENİ SATIR
                is_online=bool(request.form.get('is_online')),
                requires_lab=bool(request.form.get('requires_lab')),
                requires_computer=bool(request.form.get('requires_computer')),
                requires_projector=bool(request.form.get('requires_projector', True)),
                is_elective=bool(request.form.get('is_elective')),
                language=request.form.get('language', 'Turkish'),
                exam_type=request.form.get('exam_type'),
                prerequisite_ids=prerequisite_ids if prerequisite_ids else None
            )
            
            db.session.add(lesson)
            db.session.commit()
            
            flash(f'Lesson "{lesson.name}" created successfully!', 'success')
            return redirect(url_for('data_entry.lessons'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error creating lesson: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    return render_template('data_entry/create_lesson.html', departments=departments)

@data_entry_bp.route('/lessons/<int:lesson_id>/edit', methods=['GET', 'POST'])
def edit_lesson(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    
    if request.method == 'POST':
        try:
            # Parse prerequisite IDs
            prerequisite_str = request.form.get('prerequisite_ids', '').strip()
            prerequisite_ids = []
            if prerequisite_str:
                prerequisite_ids = [int(x.strip()) for x in prerequisite_str.split(',') if x.strip()]
            
            lesson.name = request.form['name']
            lesson.code = request.form['code'].upper()
            lesson.department_id = int(request.form['department_id'])
            lesson.grade = int(request.form['grade'])
            lesson.semester = int(request.form.get('semester', 1))
            lesson.theory_hours = int(request.form.get('theory_hours', 0))
            lesson.practice_hours = int(request.form.get('practice_hours', 0))
            lesson.lab_hours = int(request.form.get('lab_hours', 0))
            lesson.akts = int(request.form.get('akts', 0))
            lesson.local_credit = float(request.form.get('local_credit', 0))
            lesson.student_capacity = int(request.form['student_capacity'])
            lesson.min_capacity = int(request.form.get('min_capacity', 5))
            lesson.difficulty = int(request.form.get('difficulty', 3)) # YENİ SATIR
            lesson.is_online = bool(request.form.get('is_online'))
            lesson.requires_lab = bool(request.form.get('requires_lab'))
            lesson.requires_computer = bool(request.form.get('requires_computer'))
            lesson.requires_projector = bool(request.form.get('requires_projector'))
            lesson.is_elective = bool(request.form.get('is_elective'))
            lesson.language = request.form.get('language', 'Turkish')
            lesson.exam_type = request.form.get('exam_type')
            lesson.prerequisite_ids = prerequisite_ids if prerequisite_ids else None
            lesson.is_active = bool(request.form.get('is_active', True))
            
            db.session.commit()
            flash(f'Lesson "{lesson.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.lessons'))
            
        except ValueError as e:
            flash(f'Error updating lesson: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    
    # Prepare prerequisite IDs for display
    prerequisite_str = ''
    if lesson.prerequisite_ids:
        prerequisite_str = ','.join(map(str, lesson.prerequisite_ids))
    
    return render_template('data_entry/edit_lesson.html', 
                         lesson=lesson, 
                         departments=departments,
                         prerequisite_str=prerequisite_str)

@data_entry_bp.route('/lessons/<int:lesson_id>/delete', methods=['POST'])
def delete_lesson(lesson_id):
    lesson = Lesson.query.get_or_404(lesson_id)
    try:
        db.session.delete(lesson)
        db.session.commit()
        flash(f'Lesson "{lesson.name}" deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting lesson: {str(e)}', 'error')
    
    return redirect(url_for('data_entry.lessons'))

# Instructor routes (güncellenmiş)
@data_entry_bp.route('/instructors')
def instructors():
    dept_id = request.args.get('department_id', type=int)
    title = request.args.get('title')
    is_active = request.args.get('is_active')
    
    query = Instructor.query
    if dept_id:
        query = query.filter_by(department_id=dept_id)
    if title:
        query = query.filter_by(title=title)
    if is_active is not None:
        query = query.filter_by(is_active=is_active == 'true')
    
    instructors = query.order_by(Instructor.department_id, Instructor.title, Instructor.name).all()
    departments = Department.query.order_by(Department.name).all()
    
    # Get available titles for filter
    titles = db.session.query(distinct(Instructor.title)).filter(Instructor.title.isnot(None)).all()
    available_titles = [t[0] for t in titles]
    
    return render_template('data_entry/instructors.html', 
                         instructors=instructors, 
                         departments=departments,
                         available_titles=available_titles,
                         selected_dept=dept_id,
                         selected_title=title,
                         selected_active=is_active)

@data_entry_bp.route('/instructors/create', methods=['GET', 'POST'])
def create_instructor():
    if request.method == 'POST':
        try:
            # Create default availability matrix (10x5, all True)
            availability = [[True for _ in range(5)] for _ in range(10)]
            
            # Parse languages and specializations
            languages_str = request.form.get('languages', '').strip()
            languages = [lang.strip() for lang in languages_str.split(',') if lang.strip()] if languages_str else ['Turkish']
            
            specialization_str = request.form.get('specialization', '').strip()
            specializations = [spec.strip() for spec in specialization_str.split(',') if spec.strip()] if specialization_str else []
            
            instructor = Instructor(
                name=request.form['name'],
                employee_id=request.form.get('employee_id'),
                email=request.form.get('email'),
                phone=request.form.get('phone'),
                title=request.form.get('title'),
                academic_degree=request.form.get('academic_degree'),
                department_id=int(request.form['department_id']),
                office_location=request.form.get('office_location'),
                specialization=specializations if specializations else None,
                languages=languages,
                max_daily_hours=int(request.form.get('max_daily_hours', 8)),
                max_weekly_hours=int(request.form.get('max_weekly_hours', 30)),
                availability=availability,
                overtime_rate=float(request.form['overtime_rate']) if request.form.get('overtime_rate') else None,
                contract_type=request.form.get('contract_type', 'Full-time'),
                teaching_load_factor=float(request.form.get('teaching_load_factor', 1.0)),
                is_active=True,
                is_available=True
            )
            
            db.session.add(instructor)
            db.session.commit()
            
            flash(f'Instructor "{instructor.name}" created successfully!', 'success')
            return redirect(url_for('data_entry.instructors'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error creating instructor: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    return render_template('data_entry/create_instructor.html', departments=departments)

@data_entry_bp.route('/instructors/<int:instructor_id>/edit', methods=['GET', 'POST'])
def edit_instructor(instructor_id):
    instructor = Instructor.query.get_or_404(instructor_id)
    
    if request.method == 'POST':
        try:
            # Parse languages and specializations
            languages_str = request.form.get('languages', '').strip()
            languages = [lang.strip() for lang in languages_str.split(',') if lang.strip()] if languages_str else ['Turkish']
            
            specialization_str = request.form.get('specialization', '').strip()
            specializations = [spec.strip() for spec in specialization_str.split(',') if spec.strip()] if specialization_str else []
            
            instructor.name = request.form['name']
            instructor.employee_id = request.form.get('employee_id')
            instructor.email = request.form.get('email')
            instructor.phone = request.form.get('phone')
            instructor.title = request.form.get('title')
            instructor.academic_degree = request.form.get('academic_degree')
            instructor.department_id = int(request.form['department_id'])
            instructor.office_location = request.form.get('office_location')
            instructor.specialization = specializations if specializations else None
            instructor.languages = languages
            instructor.max_daily_hours = int(request.form.get('max_daily_hours', 8))
            instructor.max_weekly_hours = int(request.form.get('max_weekly_hours', 30))
            instructor.overtime_rate = float(request.form['overtime_rate']) if request.form.get('overtime_rate') else None
            instructor.contract_type = request.form.get('contract_type', 'Full-time')
            instructor.teaching_load_factor = float(request.form.get('teaching_load_factor', 1.0))
            instructor.is_active = bool(request.form.get('is_active'))
            instructor.is_available = bool(request.form.get('is_available'))
            
            db.session.commit()
            flash(f'Instructor "{instructor.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.instructors'))
            
        except ValueError as e:
            flash(f'Error updating instructor: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    
    # Prepare data for display
    languages_str = ', '.join(instructor.languages) if instructor.languages else 'Turkish'
    specialization_str = ', '.join(instructor.specialization) if instructor.specialization else ''
    
    return render_template('data_entry/edit_instructor.html', 
                         instructor=instructor, 
                         departments=departments,
                         languages_str=languages_str,
                         specialization_str=specialization_str)

@data_entry_bp.route('/instructors/<int:instructor_id>/delete', methods=['POST'])
def delete_instructor(instructor_id):
    instructor = Instructor.query.get_or_404(instructor_id)
    try:
        db.session.delete(instructor)
        db.session.commit()
        flash(f'Instructor "{instructor.name}" deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting instructor: {str(e)}', 'error')
    
    return redirect(url_for('data_entry.instructors'))

@data_entry_bp.route('/instructors/<int:instructor_id>/availability', methods=['GET', 'POST'])
def edit_instructor_availability(instructor_id):
    instructor = Instructor.query.get_or_404(instructor_id)
    
    if request.method == 'POST':
        try:
            # Parse availability matrix from form
            availability = []
            for hour in range(10):
                row = []
                for day in range(5):
                    field_name = f'availability_{hour}_{day}'
                    row.append(bool(request.form.get(field_name)))
                availability.append(row)
            
            instructor.availability = availability
            db.session.commit()
            
            flash(f'Availability updated for "{instructor.name}"!', 'success')
            return redirect(url_for('data_entry.instructors'))
            
        except Exception as e:
            flash(f'Error updating availability: {str(e)}', 'error')
    
    # Ensure availability matrix exists
    if not instructor.availability:
        instructor.availability = [[True for _ in range(5)] for _ in range(10)]
    
    time_slots = [f"{8 + i//2}:{30 if i%2 else '00'}" for i in range(10)]
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    return render_template('data_entry/instructor_availability.html', 
                         instructor=instructor, 
                         time_slots=time_slots,
                         days=days)

@data_entry_bp.route('/instructors/<int:instructor_id>/lessons', methods=['GET', 'POST'])
def instructor_lessons(instructor_id):
    instructor = Instructor.query.get_or_404(instructor_id)
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'auto_assign':
            # Otomatik atama - aynı bölümdeki tüm dersleri ata
            department_lessons = Lesson.query.filter_by(
                department_id=instructor.department_id,
                is_active=True
            ).all()
            
            assigned_count = 0
            for lesson in department_lessons:
                # Zaten atanmış mı kontrol et
                existing = InstructorLesson.query.filter_by(
                    instructor_id=instructor_id,
                    lesson_id=lesson.id
                ).first()
                
                if not existing:
                    assignment = InstructorLesson(
                        instructor_id=instructor_id,
                        lesson_id=lesson.id,
                        competency_level=7,  # Varsayılan yetkinlik
                        preference_level=5,  # Nötr tercih
                        experience_years=1,  # Varsayılan tecrübe
                        can_coordinate=False
                    )
                    db.session.add(assignment)
                    assigned_count += 1
            
            db.session.commit()
            flash(f'{assigned_count} lessons automatically assigned to {instructor.name}!', 'success')
            return redirect(url_for('data_entry.instructor_lessons', instructor_id=instructor_id))
        
        elif action == 'manual_assign':
            # Manuel atama - mevcut kod
            try:
                lesson_id = int(request.form['lesson_id'])
                competency = int(request.form.get('competency_level', 5))
                preference = int(request.form.get('preference_level', 5))
                experience = int(request.form.get('experience_years', 0))
                can_coordinate = bool(request.form.get('can_coordinate'))
                
                existing = InstructorLesson.query.filter_by(
                    instructor_id=instructor_id,
                    lesson_id=lesson_id
                ).first()
                
                if existing:
                    flash('This lesson is already assigned to the instructor!', 'error')
                else:
                    assignment = InstructorLesson(
                        instructor_id=instructor_id,
                        lesson_id=lesson_id,
                        competency_level=competency,
                        preference_level=preference,
                        experience_years=experience,
                        can_coordinate=can_coordinate,
                        notes=request.form.get('notes')
                    )
                    db.session.add(assignment)
                    db.session.commit()
                    flash('Lesson assignment added successfully!', 'success')
                
            except (ValueError, IntegrityError) as e:
                flash(f'Error adding lesson assignment: {str(e)}', 'error')
    
    # GET için
    available_lessons = Lesson.query.filter_by(
        department_id=instructor.department_id, 
        is_active=True
    ).all()
    assigned_lesson_ids = [ia.lesson_id for ia in instructor.lesson_assignments]
    unassigned_lessons = [l for l in available_lessons if l.id not in assigned_lesson_ids]
    
    return render_template('data_entry/instructor_lessons.html', 
                         instructor=instructor, 
                         unassigned_lessons=unassigned_lessons)

@data_entry_bp.route('/instructor_lessons/<int:assignment_id>/edit', methods=['GET', 'POST'])
def edit_instructor_lesson(assignment_id):
    assignment = InstructorLesson.query.get_or_404(assignment_id)
    
    if request.method == 'POST':
        try:
            assignment.competency_level = int(request.form.get('competency_level', 5))
            assignment.preference_level = int(request.form.get('preference_level', 5))
            assignment.experience_years = int(request.form.get('experience_years', 0))
            assignment.can_coordinate = bool(request.form.get('can_coordinate'))
            assignment.teaching_evaluation_score = float(request.form['evaluation_score']) if request.form.get('evaluation_score') else None
            assignment.last_taught_semester = request.form.get('last_taught_semester')
            assignment.notes = request.form.get('notes')
            
            db.session.commit()
            flash('Lesson assignment updated successfully!', 'success')
            return redirect(url_for('data_entry.instructor_lessons', instructor_id=assignment.instructor_id))
            
        except ValueError as e:
            flash(f'Error updating assignment: {str(e)}', 'error')
    
    return render_template('data_entry/edit_instructor_lesson.html', assignment=assignment)

@data_entry_bp.route('/instructor_lessons/<int:assignment_id>/delete', methods=['POST'])
def delete_instructor_lesson(assignment_id):
    assignment = InstructorLesson.query.get_or_404(assignment_id)
    instructor_id = assignment.instructor_id
    
    try:
        db.session.delete(assignment)
        db.session.commit()
        flash('Lesson assignment removed successfully!', 'success')
    except Exception as e:
        flash(f'Error removing assignment: {str(e)}', 'error')
    
    return redirect(url_for('data_entry.instructor_lessons', instructor_id=instructor_id))





@data_entry_bp.route('/departments/<int:dept_id>/edit', methods=['GET', 'POST'])
def edit_department(dept_id):
    department = Department.query.get_or_404(dept_id)
    
    if request.method == 'POST':
        try:
            department.name = request.form['name']
            department.code = request.form.get('code', '').upper()
            department.num_grades = int(request.form['num_grades'])
            department.head_of_department = request.form.get('head_of_department')
            department.building = request.form.get('building')
            department.floor = int(request.form['floor']) if request.form.get('floor') else None
            department.phone = request.form.get('phone')
            department.email = request.form.get('email')
            department.is_active = bool(request.form.get('is_active'))
            
            db.session.commit()
            flash(f'Department "{department.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.departments'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error updating department: {str(e)}', 'error')
            
    return render_template('data_entry/edit_department.html', department=department)

@data_entry_bp.route('/departments/<int:dept_id>/delete', methods=['POST'])
def delete_department(dept_id):
    department = Department.query.get_or_404(dept_id)
    try:
        db.session.delete(department)
        db.session.commit()
        flash(f'Department "{department.name}" deleted successfully!', 'success')
    except Exception as e:
        flash(f'Error deleting department: {str(e)}. It might be in use by lessons or instructors.', 'error')
        db.session.rollback()
        
    return redirect(url_for('data_entry.departments'))



# routes/data_entry.py dosyasına eklenecek yeni fonksiyon
@data_entry_bp.route('/classrooms/<int:classroom_id>/availability', methods=['GET', 'POST'])
def edit_classroom_availability(classroom_id):
    classroom = Classroom.query.get_or_404(classroom_id)
    
    if request.method == 'POST':
        try:
            availability = []
            for hour in range(10): # 10 saat dilimi (örn: 8:30'dan 18:30'a)
                row = [bool(request.form.get(f'availability_{hour}_{day}')) for day in range(5)]
                availability.append(row)
            
            classroom.availability = availability
            db.session.commit()
            
            flash(f'Availability updated for classroom "{classroom.name}"!', 'success')
            return redirect(url_for('data_entry.classrooms'))
            
        except Exception as e:
            flash(f'Error updating availability: {str(e)}', 'error')
    
    # Eğer müsaitlik bilgisi yoksa, varsayılan olarak hepsi müsait (True) bir matris oluştur
    if not classroom.availability:
        classroom.availability = [[True for _ in range(5)] for _ in range(10)]
    
    time_slots = [f"{8 + i//2}:{30 if i%2 else '00'}" for i in range(10)] # Örnek saat dilimleri
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    return render_template('data_entry/classroom_availability.html', 
                         classroom=classroom, 
                         time_slots=time_slots,
                         days=days)