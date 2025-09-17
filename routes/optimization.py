# ===== routes/optimization.py - Faculty & Multi-Department Support =====
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from models.database import db, Faculty, Department, OptimizationRun, Lesson, Instructor, Classroom
from sqlalchemy import func, distinct

# Import kontrolü ile optimization modülü
try:
    from models.optimization import AdvancedScheduleOptimizer, OptimizationConfig
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization module import failed: {e}")
    OPTIMIZATION_AVAILABLE = False
    
import threading
import uuid
import time
import logging

optimization_bp = Blueprint('optimization', __name__)

# Global storage for optimization progress
optimization_sessions = {}

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@optimization_bp.route('/')
def index():
    # YENİ: Faculty hierarchy ile departments
    faculties = Faculty.query.filter_by(is_active=True).order_by(Faculty.name).all()
    recent_runs = OptimizationRun.query.order_by(OptimizationRun.created_at.desc()).limit(10).all()
    
    # YENİ: Statistics by faculty
    faculty_stats = []
    for faculty in faculties:
        departments = Department.query.filter_by(faculty_id=faculty.id, is_active=True).all()
        total_lessons = sum([Lesson.query.filter_by(department_id=d.id, is_active=True).count() for d in departments])
        total_instructors = sum([Instructor.query.filter_by(department_id=d.id, is_active=True).count() for d in departments])
        
        faculty_stats.append({
            'faculty': faculty,
            'department_count': len(departments),
            'lesson_count': total_lessons,
            'instructor_count': total_instructors
        })
    
    return render_template('optimization/index.html', 
                         faculty_stats=faculty_stats,  # YENİ: Faculty stats
                         recent_runs=recent_runs)

@optimization_bp.route('/configure', methods=['GET', 'POST'])
def configure():
    if request.method == 'POST':
        try:
            # YENİ: Multi-department ve faculty desteği
            optimization_type = request.form.get('optimization_type')  # faculty, departments, single
            semester = int(request.form['semester'])
            academic_year = request.form.get('academic_year', '2024-2025')
            
            selected_department_ids = []
            faculty_id = None
            
            if optimization_type == 'faculty':
                # Tüm fakulte için optimization
                faculty_id = int(request.form['faculty_id'])
                faculty = Faculty.query.get(faculty_id)
                if not faculty or not faculty.is_active:
                    flash('Selected faculty not found or inactive!', 'error')
                    return redirect(url_for('optimization.configure'))
                
                # Faculty'deki tüm aktif departmentları al
                departments = Department.query.filter_by(faculty_id=faculty_id, is_active=True).all()
                selected_department_ids = [d.id for d in departments]
                
                optimization_name = f"{faculty.name} - Complete Faculty"
                
            elif optimization_type == 'departments':
                # Seçili departmentlar için optimization
                selected_department_ids = [int(id) for id in request.form.getlist('department_ids') if id]
                if not selected_department_ids:
                    flash('Please select at least one department!', 'error')
                    return redirect(url_for('optimization.configure'))
                
                departments = Department.query.filter(Department.id.in_(selected_department_ids)).all()
                
                # Faculty kontrolü - aynı faculty'den mi?
                faculty_ids = set([d.faculty_id for d in departments])
                if len(faculty_ids) == 1:
                    faculty_id = list(faculty_ids)[0]
                    faculty = Faculty.query.get(faculty_id)
                    optimization_name = f"{faculty.name} - Selected Departments"
                else:
                    # Multi-faculty optimization
                    faculty_names = [Faculty.query.get(fid).name for fid in faculty_ids]
                    optimization_name = f"Multi-Faculty: {', '.join(faculty_names)}"
                
            else:  # single department (backward compatibility)
                department_id = int(request.form['department_id'])
                department = Department.query.get(department_id)
                if not department or not department.is_active:
                    flash('Selected department not found or inactive!', 'error')
                    return redirect(url_for('optimization.configure'))
                
                selected_department_ids = [department_id]
                faculty_id = department.faculty_id
                optimization_name = f"{department.name} - Single Department"

            # YENİ: Building preference kontrolü
            use_building_preference = bool(request.form.get('use_building_preference', False))
            
            # YENİ: Shared lesson detection
            detect_shared_lessons = bool(request.form.get('detect_shared_lessons', True))
            
            # Tüm seçili departmentlardan lesson, instructor, classroom topla
            all_lessons = []
            all_instructors = []
            for dept_id in selected_department_ids:
                dept_lessons = Lesson.query.filter_by(
                    department_id=dept_id, 
                    semester=semester, 
                    is_active=True
                ).all()
                dept_instructors = Instructor.query.filter_by(
                    department_id=dept_id, 
                    is_active=True,
                    is_available=True
                ).all()
                
                all_lessons.extend(dept_lessons)
                all_instructors.extend(dept_instructors)
            
            # YENİ: Ortak ders tespiti
            shared_lessons = []
            if detect_shared_lessons and len(selected_department_ids) > 1:
                # Aynı code'a sahip dersleri bul
                lesson_codes = {}
                for lesson in all_lessons:
                    if lesson.code not in lesson_codes:
                        lesson_codes[lesson.code] = []
                    lesson_codes[lesson.code].append(lesson)
                
                # Birden fazla department'ta olan dersleri tespit et
                for code, lessons in lesson_codes.items():
                    if len(lessons) > 1:
                        # En yüksek kapasiteli olanı ana ders yap
                        main_lesson = max(lessons, key=lambda l: l.student_capacity)
                        total_capacity = sum([l.student_capacity for l in lessons])
                        
                        shared_lessons.append({
                            'main_lesson': main_lesson,
                            'all_lessons': lessons,
                            'total_capacity': total_capacity,
                            'departments': [l.department_id for l in lessons]
                        })
            
            # Kontroller
            if not all_lessons:
                dept_names = [Department.query.get(id).name for id in selected_department_ids]
                flash(f'No active lessons found for semester {semester} in: {", ".join(dept_names)}!', 'error')
                return redirect(url_for('optimization.configure'))

            if not all_instructors:
                dept_names = [Department.query.get(id).name for id in selected_department_ids]
                flash(f'No active instructors found in: {", ".join(dept_names)}!', 'error')
                return redirect(url_for('optimization.configure'))

            classrooms = Classroom.query.filter_by(is_active=True, is_bookable=True).all()
            if not classrooms:
                flash('No available classrooms found!', 'error')
                return redirect(url_for('optimization.configure'))

            # YENİ: Building preference filtering (eğer aktif ise)
            if use_building_preference and faculty_id:
                faculty = Faculty.query.get(faculty_id)
                if faculty and faculty.building:
                    # Same building classrooms get priority
                    priority_classrooms = [c for c in classrooms if c.building == faculty.building]
                    other_classrooms = [c for c in classrooms if c.building != faculty.building]
                    classrooms = priority_classrooms + other_classrooms  # Priority order
                    
                    logger.info(f"Building preference: {len(priority_classrooms)} priority, {len(other_classrooms)} other classrooms")

            # AKILLI PARAMETRE HESAPLAMA - YENİ: Multi-department için
            lesson_count = len(all_lessons)
            instructor_count = len(all_instructors)
            classroom_count = len(classrooms)
            complexity_factor = len(selected_department_ids)  # Department sayısı complexity etkiler

            # Otomatik parametre hesaplama
            if lesson_count <= 5:
                auto_population = 25 * complexity_factor
                auto_generations = 50
            elif lesson_count <= 15:
                auto_population = 40 * complexity_factor
                auto_generations = 100
            elif lesson_count <= 30:
                auto_population = 60 * complexity_factor
                auto_generations = 150
            else:
                auto_population = min(100 * complexity_factor, 500)  # Max 500
                auto_generations = 200

            # Kullanıcı girişi varsa kullan, yoksa otomatik hesaplamaları kullan
            population_size = int(request.form.get('population_size') or auto_population)
            generations = int(request.form.get('generations') or auto_generations)

            logger.info(f"Multi-dept optimization: {lesson_count} lessons, {instructor_count} instructors, {len(selected_department_ids)} departments")
            logger.info(f"Parameters: Population={population_size}, Generations={generations}")

            # YENİ: Advanced optimization config
            config = OptimizationConfig(
                population_size=population_size,
                generations=generations,
                mutation_rate=float(request.form.get('mutation_rate', 0.15)),
                crossover_rate=float(request.form.get('crossover_rate', 0.8)),
                tournament_size=max(3, min(7, population_size // 10)),
                elitism_rate=float(request.form.get('elitism_rate', 0.2)),
                adaptive_mutation=bool(request.form.get('adaptive_mutation', True)),
                local_search_probability=float(request.form.get('local_search_probability', 0.2)),
                diversity_threshold=float(request.form.get('diversity_threshold', 0.1)),
                stagnation_limit=max(15, generations // 10),
                
                # YENİ: Fitness weights - user customizable
                conflict_penalty=float(request.form.get('conflict_weight', 40.0)),
                room_utilization_weight=float(request.form.get('room_utilization_weight', 15.0)),
                instructor_balance_weight=float(request.form.get('instructor_balance_weight', 15.0)),
                preference_weight=float(request.form.get('preference_weight', 10.0)),
                time_distribution_weight=float(request.form.get('time_distribution_weight', 5.0)),
                student_satisfaction_weight=float(request.form.get('student_satisfaction_weight', 15.0))
            )
            
            # Session ID oluştur
            session_id = str(uuid.uuid4())
            
            logger.info(f"Starting optimization: {optimization_name}")
            logger.info(f"Departments: {selected_department_ids}")
            logger.info(f"Shared lessons: {len(shared_lessons)}")
            
            # YENİ: OptimizationRun kayıt - multi-department desteği ile
            optimization_run = OptimizationRun(
                session_id=session_id,
                faculty_id=faculty_id,  # YENİ: Faculty support
                department_ids=selected_department_ids,  # YENİ: Multiple departments
                semester=semester,
                academic_year=academic_year,
                use_building_preference=use_building_preference,  # YENİ: Building preference
                parameters={
                    'optimization_type': optimization_type,
                    'department_count': len(selected_department_ids),
                    'lesson_count': lesson_count,
                    'instructor_count': instructor_count,
                    'classroom_count': classroom_count,
                    'shared_lessons': len(shared_lessons),
                    'config': config.__dict__
                },
                constraints={
                    'use_building_preference': use_building_preference,
                    'detect_shared_lessons': detect_shared_lessons,
                    'selected_departments': selected_department_ids
                },
                objectives={
                    'minimize_conflicts': True,
                    'maximize_room_utilization': True,
                    'balance_instructor_workload': True,
                    'satisfy_preferences': True,
                    'optimize_time_distribution': True,
                    'maximize_student_satisfaction': True
                },
                status='initialized',
                created_by=request.form.get('created_by', 'System')
            )
            
            db.session.add(optimization_run)
            db.session.commit()
            
            # Optimizasyonu background thread'de başlat
            from flask import current_app
            app_context = current_app._get_current_object()
            
            def run_optimization():
                with app_context.app_context():
                    try:
                        optimizer = AdvancedScheduleOptimizer(config)
                        optimization_sessions[session_id] = {
                            'status': 'running', 
                            'progress': {'generation': 0, 'total_generations': config.generations},
                            'optimization_run': optimization_run
                        }
                        
                        def progress_callback(progress_data):
                            optimization_sessions[session_id]['progress'] = progress_data
                            
                            # Database progress update
                            optimization_run.progress = progress_data
                            optimization_run.status = 'running'
                            db.session.commit()
                            
                            logger.info(f"Generation {progress_data.get('generation', 0)}: Fitness {progress_data.get('best_fitness', 0):.2f}")
                        
                        # YENİ: Multi-department optimization call
                        result = optimizer.optimize_multi_department_schedule(
                            selected_department_ids, 
                            semester, 
                            session_id, 
                            progress_callback,
                            shared_lessons=shared_lessons,
                            use_building_preference=use_building_preference
                        )
                        
                        optimization_sessions[session_id]['status'] = 'completed'
                        optimization_sessions[session_id]['result'] = result
                        
                        # Database completion update
                        optimization_run.status = 'completed'
                        optimization_run.completed_at = db.func.now()
                        optimization_run.results = result  # Store optimization results
                        db.session.commit()
                        
                        logger.info(f"Multi-department optimization completed successfully for session {session_id}")
                        
                    except Exception as e:
                        logger.error(f"Optimization failed for session {session_id}: {str(e)}")
                        optimization_sessions[session_id]['status'] = 'error'
                        optimization_sessions[session_id]['error'] = str(e)
                        
                        # Database error update
                        optimization_run.status = 'error'
                        optimization_run.completed_at = db.func.now()
                        optimization_run.results = {'error': str(e)}
                        db.session.commit()
            
            thread = threading.Thread(target=run_optimization, daemon=True)
            thread.start()
            
            flash(f'Optimization started: {optimization_name} - Semester {semester}!', 'success')
            return redirect(url_for('optimization.progress', session_id=session_id))
            
        except (ValueError, Exception) as e:
            logger.error(f"Error starting optimization: {str(e)}")
            flash(f'Error starting optimization: {str(e)}', 'error')
    
    # GET request - show configuration form
    faculties = Faculty.query.filter_by(is_active=True).order_by(Faculty.name).all()
    
    if not faculties:
        flash('No active faculties found. Please create a faculty first!', 'warning')
        return redirect(url_for('faculty.index'))
    
    return render_template('optimization/configure.html', faculties=faculties)

@optimization_bp.route('/test')
def test():
    return "Optimization Blueprint is working!"

@optimization_bp.route('/progress/<session_id>')
def progress(session_id):
    if session_id not in optimization_sessions:
        flash('Optimization session not found!', 'error')
        return redirect(url_for('optimization.index'))
    
    session_data = optimization_sessions[session_id]
    
    # YENİ: Multi-department progress display
    optimization_run = session_data.get('optimization_run')
    department_info = None
    if optimization_run:
        departments = optimization_run.get_departments()
        department_info = {
            'faculty_name': optimization_run.faculty.name if optimization_run.faculty else 'Multi-Faculty',
            'department_names': [d.name for d in departments],
            'department_count': len(departments)
        }
    
    if session_data['status'] == 'completed' and 'result' in session_data:
        return redirect(url_for('results.view', run_id=optimization_run.id))
    
    return render_template('optimization/progress.html', 
                         session_id=session_id, 
                         session_data=session_data,
                         department_info=department_info)  # YENİ: Department info

@optimization_bp.route('/api/progress/<session_id>')
def api_progress(session_id):
    if session_id not in optimization_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    # Session verisini JSON serializable hale getir
    session_data = optimization_sessions[session_id].copy()
    
    # Database nesnelerini serializable hale getir
    if 'result' in session_data and hasattr(session_data['result'], 'to_dict'):
        session_data['result'] = session_data['result'].to_dict()
    elif 'result' in session_data and hasattr(session_data['result'], 'id'):
        session_data['result'] = {'id': session_data['result'].id}
    
    # Optimization run info
    if 'optimization_run' in session_data:
        opt_run = session_data['optimization_run']
        session_data['optimization_info'] = {
            'faculty_name': opt_run.faculty.name if opt_run.faculty else 'Multi-Faculty',
            'department_count': len(opt_run.get_departments()),
            'academic_year': opt_run.academic_year,
            'semester': opt_run.semester
        }
        del session_data['optimization_run']  # Remove non-serializable object
    
    return jsonify(session_data)

# YENİ: Faculty-Department hierarchy API endpoints
@optimization_bp.route('/api/faculties/<int:faculty_id>/departments')
def api_faculty_departments(faculty_id):
    """API: Faculty'ye ait departmentları getir"""
    try:
        departments = Department.query.filter_by(
            faculty_id=faculty_id, 
            is_active=True
        ).order_by(Department.name).all()
        
        # Her department için istatistikler
        dept_data = []
        for dept in departments:
            lesson_count = Lesson.query.filter_by(department_id=dept.id, is_active=True).count()
            instructor_count = Instructor.query.filter_by(department_id=dept.id, is_active=True).count()
            
            dept_data.append({
                'id': dept.id,
                'name': dept.name,
                'code': dept.code,
                'lesson_count': lesson_count,
                'instructor_count': instructor_count
            })
        
        return jsonify({
            'success': True,
            'departments': dept_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@optimization_bp.route('/api/preview')
def api_preview():
    """YENİ: Multi-department optimization preview"""
    try:
        optimization_type = request.args.get('type', 'single')
        semester = int(request.args.get('semester', 1))
        
        if optimization_type == 'faculty':
            faculty_id = int(request.args.get('faculty_id'))
            departments = Department.query.filter_by(faculty_id=faculty_id, is_active=True).all()
            department_ids = [d.id for d in departments]
        else:
            department_ids = [int(id) for id in request.args.getlist('department_ids') if id]
        
        if not department_ids:
            return jsonify({'success': False, 'error': 'No departments selected'})
        
        # Collect all lessons and instructors
        all_lessons = []
        all_instructors = []
        dept_stats = []
        
        for dept_id in department_ids:
            dept = Department.query.get(dept_id)
            lessons = Lesson.query.filter_by(department_id=dept_id, semester=semester, is_active=True).all()
            instructors = Instructor.query.filter_by(department_id=dept_id, is_active=True).all()
            
            all_lessons.extend(lessons)
            all_instructors.extend(instructors)
            
            dept_stats.append({
                'department_name': dept.name,
                'lessons': len(lessons),
                'instructors': len(instructors),
                'total_hours': sum(l.theory_hours + l.practice_hours + l.lab_hours for l in lessons)
            })
        
        # Shared lessons detection
        shared_lesson_codes = {}
        for lesson in all_lessons:
            if lesson.code not in shared_lesson_codes:
                shared_lesson_codes[lesson.code] = []
            shared_lesson_codes[lesson.code].append(lesson)
        
        shared_lessons = [code for code, lessons in shared_lesson_codes.items() if len(lessons) > 1]
        
        # Warnings
        warnings = []
        if len(all_lessons) == 0:
            warnings.append("No lessons found for selected departments and semester")
        if len(all_instructors) == 0:
            warnings.append("No instructors available in selected departments")
        if len(all_instructors) > 0 and len(all_lessons) > 0:
            avg_lessons_per_instructor = len(all_lessons) / len(all_instructors)
            if avg_lessons_per_instructor > 6:
                warnings.append(f"High lesson-to-instructor ratio: {avg_lessons_per_instructor:.1f}")
        
        return jsonify({
            'success': True,
            'total_lessons': len(all_lessons),
            'total_instructors': len(all_instructors),
            'department_count': len(department_ids),
            'shared_lessons': len(shared_lessons),
            'shared_lesson_codes': shared_lessons,
            'department_stats': dept_stats,
            'warnings': warnings,
            'estimated_complexity': 'High' if len(all_lessons) > 30 else 'Medium' if len(all_lessons) > 10 else 'Low'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# YENİ: Quick optimization templates
@optimization_bp.route('/quick/<template_type>')
def quick_optimization(template_type):
    """Quick optimization templates"""
    templates = {
        'full_university': 'Optimize all faculties and departments',
        'single_faculty': 'Optimize one complete faculty',
        'cross_faculty': 'Optimize selected departments across faculties',
        'single_department': 'Traditional single department optimization'
    }
    
    if template_type not in templates:
        flash('Unknown optimization template!', 'error')
        return redirect(url_for('optimization.configure'))
    
    return render_template('optimization/quick_template.html', 
                         template_type=template_type,
                         template_description=templates[template_type])