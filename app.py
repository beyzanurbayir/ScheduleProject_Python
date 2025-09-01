from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from models.database import init_db
from routes.data_entry import data_entry_bp
from routes.optimization import optimization_bp
from routes.results import results_bp
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'sqlite:///schedule_optimizer.db')
    
    # Initialize database
    init_db(app)
    
    # Register blueprints
    app.register_blueprint(data_entry_bp, url_prefix='/data')
    app.register_blueprint(optimization_bp, url_prefix='/optimize')
    app.register_blueprint(results_bp, url_prefix='/results')
    
    @app.route('/')
    def index():
        from models.database import db, Department, Lesson, Instructor, Classroom
        
        departments_count = Department.query.count()
        lessons_count = Lesson.query.count()
        instructors_count = Instructor.query.count()
        classrooms_count = Classroom.query.count()
        
        return render_template('index.html',
                            departments_count=departments_count,
                            lessons_count=lessons_count,
                            instructors_count=instructors_count,
                            classrooms_count=classrooms_count)
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, threaded=True)