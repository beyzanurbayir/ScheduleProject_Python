# ===== routes/optimization.py - DÜZELTME =====
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from models.database import db, Department, OptimizationRun, Lesson, Instructor, Classroom
# import app as flask_app

# Import kontrolü ile optimization modülü
try:
    from models.optimization import AdvancedScheduleOptimizer, OptimizationConfig
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Optimization module import failed: {e}")
    OPTIMIZATION_AVAILABLE = False
    
from sqlalchemy import func
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
    departments = Department.query.filter_by(is_active=True).all()
    recent_runs = OptimizationRun.query.order_by(OptimizationRun.created_at.desc()).limit(10).all()
    return render_template('optimization/index.html', departments=departments, recent_runs=recent_runs)

@optimization_bp.route('/configure', methods=['GET', 'POST'])
def configure():
    if request.method == 'POST':
        try:
            # Form verilerini al
            department_id = int(request.form['department_id'])
            semester = int(request.form['semester'])

            # Önce temel verileri kontrol et
            department = Department.query.get(department_id)
            if not department:
                flash('Selected department not found!', 'error')
                return redirect(url_for('optimization.configure'))

            lessons = Lesson.query.filter_by(
                department_id=department_id, 
                semester=semester, 
                is_active=True
            ).all()
            
            if not lessons:
                flash(f'No active lessons found for {department.name} - Semester {semester}!', 'error')
                return redirect(url_for('optimization.configure'))

            instructors = Instructor.query.filter_by(
                department_id=department_id, 
                is_active=True
            ).all()
            
            if not instructors:
                flash(f'No active instructors found for {department.name}!', 'error')
                return redirect(url_for('optimization.configure'))

            classrooms = Classroom.query.filter_by(is_active=True, is_bookable=True).all()
            
            if not classrooms:
                flash('No available classrooms found!', 'error')
                return redirect(url_for('optimization.configure'))

                        # Optimizasyon parametrelerini parse et
            # AKILLI PARAMETRE HESAPLAMA
            lesson_count = len(lessons)
            instructor_count = len(instructors)
            classroom_count = len(classrooms)

            # Otomatik parametre hesaplama
            if lesson_count <= 5:
                auto_population = 20  # Çok küçük problemler
                auto_generations = 50
            elif lesson_count <= 15:
                auto_population = 40  # Küçük problemler  
                auto_generations = 100
            elif lesson_count <= 30:
                auto_population = 60  # Orta problemler
                auto_generations = 150
            else:
                auto_population = 80  # Büyük problemler
                auto_generations = 200

            # Kullanıcı girişi varsa kullan, yoksa otomatik hesaplamaları kullan
            population_size = int(request.form.get('population_size') or auto_population)
            generations = int(request.form.get('generations') or auto_generations)

            logger.info(f"Using: Population={population_size}, Generations={generations} for {lesson_count} lessons")

            # Optimizasyon parametrelerini oluştur - TAMAMLANDI
            config = OptimizationConfig(
                population_size=population_size,
                generations=generations,
                mutation_rate=0.15,
                crossover_rate=0.8,
                tournament_size=max(3, min(7, population_size // 10)),
                elitism_rate=0.2,
                adaptive_mutation=True,
                local_search_probability=0.2,
                diversity_threshold=0.1,
                stagnation_limit=max(15, generations // 10),
                # Fitness ağırlıkları
                conflict_penalty=35.0,
                room_utilization_weight=20.0,
                instructor_balance_weight=15.0,
                preference_weight=12.0,
                time_distribution_weight=10.0,
                student_satisfaction_weight=8.0
            )
            
            # Session ID oluştur
            session_id = str(uuid.uuid4())
            
            logger.info(f"Starting optimization for {department.name}, Semester {semester}")
            logger.info(f"Found: {len(lessons)} lessons, {len(instructors)} instructors, {len(classrooms)} classrooms")
            
            # Optimizasyonu background thread'de başlat
            # Optimizasyonu background thread'de başlat  
            from flask import current_app
            app_context = current_app._get_current_object()
            
            # Sonra run_optimization fonksiyonunu şöyle değiştirin:
            def run_optimization():
                with app_context.app_context():
                    try:
                        optimizer = AdvancedScheduleOptimizer(config)
                        optimization_sessions[session_id] = {
                            'status': 'running', 
                            'progress': {'generation': 0, 'total_generations': config.generations}
                        }
                        
                        def progress_callback(progress_data):
                            optimization_sessions[session_id]['progress'] = progress_data
                            logger.info(f"Generation {progress_data.get('generation', 0)}: Fitness {progress_data.get('best_fitness', 0):.2f}")
                        
                        result = optimizer.optimize_schedule(
                            department_id, semester, session_id, progress_callback
                        )
                        
                        optimization_sessions[session_id]['status'] = 'completed'
                        optimization_sessions[session_id]['result'] = result
                        logger.info(f"Optimization completed successfully for session {session_id}")
                        
                    except Exception as e:
                        logger.error(f"Optimization failed for session {session_id}: {str(e)}")
                        optimization_sessions[session_id]['status'] = 'error'
                        optimization_sessions[session_id]['error'] = str(e)
                        
            
            
            thread = threading.Thread(target=run_optimization, daemon=True)
            thread.start()
            
            flash(f'Optimization started for {department.name} - Semester {semester}!', 'success')
            return redirect(url_for('optimization.progress', session_id=session_id))
            
        except (ValueError, Exception) as e:
            logger.error(f"Error starting optimization: {str(e)}")
            flash(f'Error starting optimization: {str(e)}', 'error')
    
    # GET request - show configuration form
    departments = Department.query.filter_by(is_active=True).all()
    return render_template('optimization/configure.html', departments=departments)

@optimization_bp.route('/test')
def test():
    return "Optimization Blueprint is working!"


@optimization_bp.route('/progress/<session_id>')
def progress(session_id):
    if session_id not in optimization_sessions:
        flash('Optimization session not found!', 'error')
        return redirect(url_for('optimization.index'))
    
    session_data = optimization_sessions[session_id]
    
    if session_data['status'] == 'completed' and 'result' in session_data:
        return redirect(url_for('results.view', run_id=session_data['result'].id))
    
    return render_template('optimization/progress.html', 
                         session_id=session_id, 
                         session_data=session_data)

@optimization_bp.route('/api/progress/')
@optimization_bp.route('/api/progress/<session_id>')
def api_progress(session_id=None):
    if session_id is None:
        return jsonify({'error': 'No session ID provided'}), 400
    
    if session_id not in optimization_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(optimization_sessions[session_id])

@optimization_bp.route('/api/preview/<int:department_id>/<int:semester>')
def api_preview(department_id, semester):
    """Department ve semester seçimi için önizleme verisi"""
    try:
        lessons = Lesson.query.filter_by(
            department_id=department_id, 
            semester=semester, 
            is_active=True
        ).all()
        
        instructors = Instructor.query.filter_by(
            department_id=department_id, 
            is_active=True
        ).all()
        
        classrooms = Classroom.query.filter_by(is_active=True, is_bookable=True).count()
        
        total_hours = sum(lesson.theory_hours + lesson.practice_hours + lesson.lab_hours for lesson in lessons)
        
        warnings = []
        if len(lessons) == 0:
            warnings.append("No lessons found for this semester")
        if len(instructors) == 0:
            warnings.append("No instructors available in this department")
        if classrooms == 0:
            warnings.append("No classrooms available")
        if len(instructors) > 0 and len(lessons) > 0:
            avg_lessons_per_instructor = len(lessons) / len(instructors)
            if avg_lessons_per_instructor > 8:
                warnings.append("High lesson-to-instructor ratio detected")
        
        return jsonify({
            'lessons': len(lessons),
            'instructors': len(instructors),
            'classrooms': classrooms,
            'total_hours': total_hours,
            'warnings': warnings
        })
        
    except Exception as e:
        logger.error(f"Error in preview API: {str(e)}")
        return jsonify({'error': str(e)}), 500

@optimization_bp.route('/runs')
def runs():
    page = request.args.get('page', 1, type=int)
    dept_id = request.args.get('department_id', type=int)
    status = request.args.get('status')
    
    query = OptimizationRun.query
    
    if dept_id:
        query = query.filter_by(department_id=dept_id)
    if status:
        query = query.filter_by(status=status)
    
    runs = query.order_by(OptimizationRun.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False
    )
    
    departments = Department.query.filter_by(is_active=True).all()
    
    return render_template('optimization/runs.html', 
                         runs=runs, 
                         departments=departments,
                         selected_dept=dept_id,
                         selected_status=status)

@optimization_bp.route('/cancel/<session_id>', methods=['POST'])
def cancel_optimization(session_id):
    """Çalışan optimizasyonu iptal et"""
    if session_id in optimization_sessions:
        optimization_sessions[session_id]['status'] = 'cancelled'
        flash('Optimization cancelled successfully!', 'info')
    else:
        flash('Optimization session not found!', 'error')
    
    return redirect(url_for('optimization.index'))

# Cleanup eski session'ları
import atexit
def cleanup_sessions():
    """Uygulama kapanırken session'ları temizle"""
    optimization_sessions.clear()
    logger.info("Optimization sessions cleaned up")

atexit.register(cleanup_sessions)