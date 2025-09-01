# ===== routes/data_entry.py =====
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from models.database import db, Department, Lesson, Instructor, Classroom, InstructorLesson
from sqlalchemy.exc import IntegrityError
import json

data_entry_bp = Blueprint('data_entry', __name__)

@data_entry_bp.route('/')
def index():
    departments = Department.query.all()
    return render_template('data_entry/index.html', departments=departments)

@data_entry_bp.route('/departments')
def departments():
    departments = Department.query.order_by(Department.name).all()
    return render_template('data_entry/departments.html', departments=departments)

@data_entry_bp.route('/departments/create', methods=['GET', 'POST'])
@data_entry_bp.route('/departments/create', methods=['GET', 'POST'])
def create_department():
    if request.method == 'POST':
        try:
            name = request.form['name']
            num_grades = int(request.form['num_grades'])
            
            department = Department(
                name=name,
                num_grades=num_grades
            )
            
            db.session.add(department)
            db.session.commit()
            
            flash(f'Department "{name}" created successfully!', 'success')
            return redirect(url_for('data_entry.departments'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error creating department: {str(e)}', 'error')
    
    return render_template('data_entry/create_department.html')

@data_entry_bp.route('/departments/<int:dept_id>/edit', methods=['GET', 'POST'])
def edit_department(dept_id):
    department = Department.query.get_or_404(dept_id)
    
    if request.method == 'POST':
        try:
            department.name = request.form['name']
            department.num_grades = int(request.form['num_grades'])

            enrollment_str = request.form['enrollment']
            department.enrollment = [int(x.strip()) for x in enrollment_str.split(',')]
            department.num_grades = department.num_grades // 2

            db.session.commit()
            flash(f'Department "{department.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.departments'))
            
        except ValueError as e:
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
        flash(f'Error deleting department: {str(e)}', 'error')
    
    return redirect(url_for('data_entry.departments'))

@data_entry_bp.route('/lessons')
def lessons():
    dept_id = request.args.get('department_id', type=int)
    grade = request.args.get('grade', type=int)
    
    query = Lesson.query
    if dept_id:
        query = query.filter_by(department_id=dept_id)
    if grade:
        query = query.filter_by(grade=grade)

    lessons = query.order_by(Lesson.department_id, Lesson.grade, Lesson.name).all()
    departments = Department.query.order_by(Department.name).all()
    
    return render_template('data_entry/lessons.html', 
                         lessons=lessons, 
                         departments=departments,
                         selected_dept=dept_id,
                         selected_grade=grade)

@data_entry_bp.route('/lessons/create', methods=['GET', 'POST'])
def create_lesson():
    if request.method == 'POST':
        try:
            lesson = Lesson(
                name=request.form['name'],
                code=request.form['code'],
                department_id=int(request.form['department_id']),
                grade=int(request.form['grade']),
                theory_hours=int(request.form['theory_hours']),
                practice_hours=int(request.form['practice_hours']),
                lab_hours=int(request.form['lab_hours']),
                student_capacity=int(request.form['student_capacity']),
                is_online=bool(request.form.get('is_online')),
                requires_lab=bool(request.form.get('requires_lab'))
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
            lesson.name = request.form['name']
            lesson.code = request.form['code']
            lesson.department_id = int(request.form['department_id'])
            lesson.grade = int(request.form['grade'])
            lesson.theory_hours = int(request.form['theory_hours'])
            lesson.practice_hours = int(request.form['practice_hours'])
            lesson.lab_hours = int(request.form['lab_hours'])
            # lesson.akts = int(request.form['akts'])
            # lesson.credit = int(request.form['credit'])
            lesson.is_online = bool(request.form.get('is_online'))
            lesson.requires_lab = bool(request.form.get('requires_lab'))
            lesson.min_capacity = int(request.form.get('min_capacity', 0))
            
            db.session.commit()
            flash(f'Lesson "{lesson.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.lessons'))
            
        except ValueError as e:
            flash(f'Error updating lesson: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    return render_template('data_entry/edit_lesson.html', lesson=lesson, departments=departments)

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

@data_entry_bp.route('/instructors')
def instructors():
    dept_id = request.args.get('department_id', type=int)
    
    query = Instructor.query
    if dept_id:
        query = query.filter_by(department_id=dept_id)
    
    instructors = query.order_by(Instructor.department_id, Instructor.name).all()
    departments = Department.query.order_by(Department.name).all()
    
    return render_template('data_entry/instructors.html', 
                         instructors=instructors, 
                         departments=departments,
                         selected_dept=dept_id)

@data_entry_bp.route('/instructors/create', methods=['GET', 'POST'])
def create_instructor():
    if request.method == 'POST':
        try:
            # Create default availability matrix (10x5, all True)
            availability = [[True for _ in range(5)] for _ in range(10)]
            
            instructor = Instructor(
                name=request.form['name'],
                email=request.form.get('email'),
                title=request.form.get('title'),
                department_id=int(request.form['department_id']),
                max_daily_hours=int(request.form.get('max_daily_hours', 8)),
                max_weekly_hours=int(request.form.get('max_weekly_hours', 30)),
                availability=availability,
                is_active=True
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
            instructor.name = request.form['name']
            instructor.email = request.form.get('email')
            instructor.title = request.form.get('title')
            instructor.department_id = int(request.form['department_id'])
            instructor.max_daily_hours = int(request.form.get('max_daily_hours', 8))
            instructor.max_weekly_hours = int(request.form.get('max_weekly_hours', 30))
            instructor.is_active = bool(request.form.get('is_active'))
            
            db.session.commit()
            flash(f'Instructor "{instructor.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.instructors'))
            
        except ValueError as e:
            flash(f'Error updating instructor: {str(e)}', 'error')
    
    departments = Department.query.order_by(Department.name).all()
    return render_template('data_entry/edit_instructor.html', instructor=instructor, departments=departments)

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
        try:
            lesson_id = int(request.form['lesson_id'])
            competency = int(request.form.get('competency_level', 5))
            preference = int(request.form.get('preference_level', 5))
            
            # Check if assignment already exists
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
                    preference_level=preference
                )
                db.session.add(assignment)
                db.session.commit()
                flash('Lesson assignment added successfully!', 'success')
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error adding lesson assignment: {str(e)}', 'error')
    
    # Get available lessons from same department
    available_lessons = Lesson.query.filter_by(department_id=instructor.department_id).all()
    assigned_lesson_ids = [ia.lesson_id for ia in instructor.lesson_assignments]
    unassigned_lessons = [l for l in available_lessons if l.id not in assigned_lesson_ids]
    
    return render_template('data_entry/instructor_lessons.html', 
                         instructor=instructor, 
                         unassigned_lessons=unassigned_lessons)

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

@data_entry_bp.route('/classrooms')
def classrooms():
    classrooms = Classroom.query.order_by(Classroom.name).all()
    return render_template('data_entry/classrooms.html', classrooms=classrooms)

@data_entry_bp.route('/classrooms/create', methods=['GET', 'POST'])
def create_classroom():
    if request.method == 'POST':
        try:
            classroom = Classroom(
                name=request.form['name'],
                capacity=int(request.form['capacity']),
                has_lab=bool(request.form.get('has_lab')),
                has_projector=bool(request.form.get('has_projector')),
                has_computer=bool(request.form.get('has_computer')),
                building=request.form.get('building'),
                floor=int(request.form['floor']) if request.form.get('floor') else None,
                notes=request.form.get('notes'),
                is_active=True
            )
            
            db.session.add(classroom)
            db.session.commit()
            
            flash(f'Classroom "{classroom.name}" created successfully!', 'success')
            return redirect(url_for('data_entry.classrooms'))
            
        except (ValueError, IntegrityError) as e:
            flash(f'Error creating classroom: {str(e)}', 'error')
    
    return render_template('data_entry/create_classroom.html')

@data_entry_bp.route('/classrooms/<int:classroom_id>/edit', methods=['GET', 'POST'])
def edit_classroom(classroom_id):
    classroom = Classroom.query.get_or_404(classroom_id)
    
    if request.method == 'POST':
        try:
            classroom.name = request.form['name']
            classroom.capacity = int(request.form['capacity'])
            classroom.has_lab = bool(request.form.get('has_lab'))
            classroom.has_projector = bool(request.form.get('has_projector'))
            classroom.has_computer = bool(request.form.get('has_computer'))
            classroom.building = request.form.get('building')
            classroom.floor = int(request.form['floor']) if request.form.get('floor') else None
            classroom.notes = request.form.get('notes')
            classroom.is_active = bool(request.form.get('is_active'))
            
            db.session.commit()
            flash(f'Classroom "{classroom.name}" updated successfully!', 'success')
            return redirect(url_for('data_entry.classrooms'))
            
        except ValueError as e:
            flash(f'Error updating classroom: {str(e)}', 'error')
    
    return render_template('data_entry/edit_classroom.html', classroom=classroom)

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