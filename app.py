from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from models.database import init_db, db, Department, Lesson, Instructor, Classroom, OptimizationRun , InstructorLesson 
from routes.data_entry import data_entry_bp
from routes.optimization import optimization_bp
from routes.results import results_bp
from sqlalchemy import func
import os
from datetime import datetime, timedelta

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'sqlite:///schedule_optimizer.db')
    
    # Initialize database
    init_db(app)
    
    # Register blueprints
    app.register_blueprint(data_entry_bp, url_prefix='/data')
    app.register_blueprint(optimization_bp, url_prefix='/optimize')
    app.register_blueprint(results_bp, url_prefix='/results')
    
    @app.route('/')
    def index():
        """Enhanced dashboard with comprehensive statistics"""
        try:
            # Basic counts
            departments_count = Department.query.filter_by(is_active=True).count()
            lessons_count = Lesson.query.filter_by(is_active=True).count()
            instructors_count = Instructor.query.filter_by(is_active=True).count()
            classrooms_count = Classroom.query.filter_by(is_active=True).count()
            
            # Advanced statistics
            total_capacity = db.session.query(func.sum(Classroom.capacity)).filter_by(is_active=True).scalar() or 0
            active_optimizations = OptimizationRun.query.filter_by(status='running').count()
            
            # Recent optimizations
            recent_optimizations = OptimizationRun.query.order_by(
                OptimizationRun.created_at.desc()
            ).limit(5).all()
            
            # System health metrics
            departments_with_lessons = db.session.query(func.count(func.distinct(Lesson.department_id))).filter_by(is_active=True).scalar() or 0
            instructors_assigned = db.session.query(func.count(func.distinct(InstructorLesson.instructor_id))).select_from(
                db.session.query(Instructor).join('Instructor.lesson_assignments')
            ).scalar() or 0
            lab_classrooms = Classroom.query.filter_by(has_lab=True, is_active=True).count()
            
            # Calculate averages
            avg_lesson_hours = db.session.query(
                func.avg(Lesson.theory_hours + Lesson.practice_hours + Lesson.lab_hours)
            ).filter_by(is_active=True).scalar() or 0
            
            avg_class_size = db.session.query(func.avg(Lesson.student_capacity)).filter_by(is_active=True).scalar() or 0
            
            # Room utilization estimation (simplified)
            total_classroom_hours = classrooms_count * 10 * 5  # 10 hours/day, 5 days/week
            estimated_usage = lessons_count * avg_lesson_hours if lessons_count and avg_lesson_hours else 0
            room_utilization = min((estimated_usage / total_classroom_hours * 100) if total_classroom_hours > 0 else 0, 100)
            
            # Data completeness score
            data_completeness = calculate_data_completeness(
                departments_count, lessons_count, instructors_count, classrooms_count
            )
            
            # Setup progress
            setup_progress = calculate_setup_progress(
                departments_count, lessons_count, instructors_count, classrooms_count, instructors_assigned
            )
            
            # System alerts
            system_alerts = []
            if departments_count == 0:
                system_alerts.append("No departments configured")
            if lessons_count < departments_count * 5:  # Assuming at least 5 lessons per department
                system_alerts.append("Insufficient lessons configured")
            if instructors_assigned < instructors_count * 0.8:
                system_alerts.append("Many instructors without lesson assignments")
            if classrooms_count == 0:
                system_alerts.append("No classrooms configured")
            
            # Average instructor workload estimation
            if instructors_count > 0:
                total_lesson_hours = db.session.query(
                    func.sum(Lesson.theory_hours + Lesson.practice_hours + Lesson.lab_hours)
                ).filter_by(is_active=True).scalar() or 0
                instructor_workload = total_lesson_hours / instructors_count if instructors_count > 0 else 0
            else:
                instructor_workload = 0
            
            return render_template('index.html',
                                 # Basic counts
                                 departments_count=departments_count,
                                 lessons_count=lessons_count,
                                 instructors_count=instructors_count,
                                 classrooms_count=classrooms_count,
                                 
                                 # Advanced metrics
                                 total_capacity=total_capacity,
                                 active_optimizations=active_optimizations,
                                 recent_optimizations=recent_optimizations,
                                 
                                 # Health indicators
                                 departments_with_lessons=departments_with_lessons,
                                 instructors_assigned=instructors_assigned,
                                 lab_classrooms=lab_classrooms,
                                 avg_lesson_hours=round(avg_lesson_hours, 1),
                                 avg_class_size=round(avg_class_size),
                                 room_utilization=round(room_utilization),
                                 instructor_workload=round(instructor_workload),
                                 
                                 # System status
                                 data_completeness=round(data_completeness),
                                 setup_progress=round(setup_progress),
                                 system_alerts=system_alerts)
        
        except Exception as e:
            # Fallback with minimal data in case of database errors
            app.logger.error(f"Dashboard error: {str(e)}")
            return render_template('index.html',
                                 departments_count=0,
                                 lessons_count=0,
                                 instructors_count=0,
                                 classrooms_count=0,
                                 total_capacity=0,
                                 active_optimizations=0,
                                 recent_optimizations=[],
                                 system_alerts=['Database connection error'])
    
    @app.route('/api/dashboard-stats')
    def dashboard_stats_api():
        """API endpoint for real-time dashboard updates"""
        try:
            stats = {
                'departments_count': Department.query.filter_by(is_active=True).count(),
                'lessons_count': Lesson.query.filter_by(is_active=True).count(),
                'instructors_count': Instructor.query.filter_by(is_active=True).count(),
                'classrooms_count': Classroom.query.filter_by(is_active=True).count(),
                'active_optimizations': OptimizationRun.query.filter_by(status='running').count(),
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(stats)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    def calculate_data_completeness(departments, lessons, instructors, classrooms):
        """Calculate overall data completeness percentage"""
        score = 0
        
        # Basic data presence (40 points)
        if departments > 0: score += 10
        if lessons > 0: score += 10
        if instructors > 0: score += 10
        if classrooms > 0: score += 10
        
        # Data ratios (40 points)
        if departments > 0:
            lessons_per_dept = lessons / departments
            if lessons_per_dept >= 5: score += 10  # At least 5 lessons per department
            elif lessons_per_dept >= 3: score += 5
            
            instructors_per_dept = instructors / departments
            if instructors_per_dept >= 3: score += 10  # At least 3 instructors per department
            elif instructors_per_dept >= 1: score += 5
        
        # Classroom adequacy (20 points)
        if lessons > 0 and classrooms > 0:
            classroom_ratio = classrooms / (lessons * 0.3)  # Assuming 30% of lessons need unique classrooms
            if classroom_ratio >= 1: score += 15
            elif classroom_ratio >= 0.5: score += 10
            elif classroom_ratio >= 0.2: score += 5
        
        # Bonus for well-structured data
        if all([departments >= 2, lessons >= 10, instructors >= 5, classrooms >= 5]):
            score += 5
        
        return min(score, 100)
    
    def calculate_setup_progress(departments, lessons, instructors, classrooms, instructors_assigned):
        """Calculate setup progress percentage"""
        total_steps = 5
        completed_steps = 0
        
        if departments > 0: completed_steps += 1
        if lessons > 0: completed_steps += 1
        if instructors > 0: completed_steps += 1
        if classrooms > 0: completed_steps += 1
        if instructors_assigned > 0: completed_steps += 1
        
        base_progress = (completed_steps / total_steps) * 80  # Base 80%
        
        # Bonus for sufficient data
        bonus = 0
        if departments >= 2: bonus += 5
        if lessons >= 10: bonus += 5
        if instructors >= 5: bonus += 5
        if classrooms >= 5: bonus += 5
        
        return min(base_progress + bonus, 100)
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(403)
    def forbidden(error):
        return render_template('errors/403.html'), 403
    
    # Template filters
    @app.template_filter('datetime')
    def datetime_filter(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value.strftime('%Y-%m-%d %H:%M')
    
    @app.template_filter('date')
    def date_filter(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value.strftime('%Y-%m-%d')
    
    @app.template_filter('time_ago')
    def time_ago_filter(value):
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        
        now = datetime.now(value.tzinfo) if value.tzinfo else datetime.now()
        diff = now - value
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Development configuration
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True, threaded=True, port=5000)
    else:
        # Production configuration
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), threaded=True)