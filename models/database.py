from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import JSON

db = SQLAlchemy()

def init_db(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['DATABASE_URL']
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    
    with app.app_context():
        db.create_all()

class Department(db.Model):
    __tablename__ = 'departments'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True)
    num_grades = db.Column(db.Integer, nullable=False)  # Sadece sınıf sayısı
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    lessons = db.relationship('Lesson', backref='department_ref', lazy=True, cascade='all, delete-orphan')
    instructors = db.relationship('Instructor', backref='department_ref', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'num_grades': self.num_grades,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Lesson(db.Model):
    __tablename__ = 'lessons'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    code = db.Column(db.String(50), nullable=False)
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    grade = db.Column(db.Integer, nullable=False)
    theory_hours = db.Column(db.Integer, nullable=False, default=0)
    practice_hours = db.Column(db.Integer, nullable=False, default=0)
    lab_hours = db.Column(db.Integer, nullable=False, default=0)
    # akts = db.Column(db.Integer, nullable=False)
    # credit = db.Column(db.Integer, nullable=False)
    student_capacity = db.Column(db.Integer, nullable=False, default=40)
    is_online = db.Column(db.Boolean, nullable=False, default=False)
    requires_lab = db.Column(db.Boolean, nullable=False, default=False)
    # min_capacity = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def total_hours(self):
        return self.theory_hours + self.practice_hours + self.lab_hours
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'department_id': self.department_id,
            'department_name': self.department_ref.name,
            'grade': self.grade,
            'theory_hours': self.theory_hours,
            'practice_hours': self.practice_hours,
            'lab_hours': self.lab_hours,
            'total_hours': self.total_hours,
            'is_online': self.is_online,
            'requires_lab': self.requires_lab,
            'student_capacity': self.student_capacity,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Classroom(db.Model):
    __tablename__ = 'classrooms'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    capacity = db.Column(db.Integer, nullable=False)
    has_lab = db.Column(db.Boolean, nullable=False, default=False)
    has_projector = db.Column(db.Boolean, nullable=False, default=True)
    has_computer = db.Column(db.Boolean, nullable=False, default=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    building = db.Column(db.String(50), nullable=True)
    floor = db.Column(db.Integer, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'capacity': self.capacity,
            'has_lab': self.has_lab,
            'has_projector': self.has_projector,
            'has_computer': self.has_computer,
            'is_active': self.is_active,
            'building': self.building,
            'floor': self.floor,
            'notes': self.notes,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Instructor(db.Model):
    __tablename__ = 'instructors'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=True)
    title = db.Column(db.String(50), nullable=True)  # Prof, Doç, Dr, etc.
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    max_daily_hours = db.Column(db.Integer, nullable=False, default=8)
    max_weekly_hours = db.Column(db.Integer, nullable=False, default=30)
    availability = db.Column(JSON, nullable=False)  # 10x5 matrix
    preferred_times = db.Column(JSON, nullable=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    lesson_assignments = db.relationship('InstructorLesson', backref='instructor_ref', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'title': self.title,
            'department_id': self.department_id,
            'department_name': self.department_ref.name,
            'max_daily_hours': self.max_daily_hours,
            'max_weekly_hours': self.max_weekly_hours,
            'availability': self.availability,
            'preferred_times': self.preferred_times,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class InstructorLesson(db.Model):
    __tablename__ = 'instructor_lessons'
    
    id = db.Column(db.Integer, primary_key=True)
    instructor_id = db.Column(db.Integer, db.ForeignKey('instructors.id'), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lessons.id'), nullable=False)
    competency_level = db.Column(db.Integer, nullable=False, default=5)  # 1-10 scale
    preference_level = db.Column(db.Integer, nullable=False, default=5)  # 1-10 scale
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    lesson = db.relationship('Lesson', backref='instructor_assignments', lazy=True)
    
    __table_args__ = (db.UniqueConstraint('instructor_id', 'lesson_id'),)

class OptimizationRun(db.Model):
    __tablename__ = 'optimization_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False, unique=True)
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    grade = db.Column(db.Integer, nullable=False)
    parameters = db.Column(JSON, nullable=False)  # algorithm parameters
    status = db.Column(db.String(20), nullable=False, default='initialized')  # initialized, running, completed, error
    progress = db.Column(JSON, nullable=True)
    results = db.Column(JSON, nullable=True)
    fitness_score = db.Column(db.Float, nullable=True)
    conflicts_count = db.Column(db.Integer, nullable=True)
    runtime_seconds = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    department = db.relationship('Department', backref='optimization_runs', lazy=True)
    schedules = db.relationship('Schedule', backref='optimization_run', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'department_id': self.department_id,
            'department_name': self.department.name,
            'grade': self.grade,
            'parameters': self.parameters,
            'status': self.status,
            'progress': self.progress,
            'results': self.results,
            'fitness_score': self.fitness_score,
            'conflicts_count': self.conflicts_count,
            'runtime_seconds': self.runtime_seconds,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class Schedule(db.Model):
    __tablename__ = 'schedules'
    
    id = db.Column(db.Integer, primary_key=True)
    optimization_run_id = db.Column(db.Integer, db.ForeignKey('optimization_runs.id'), nullable=False)
    lesson_id = db.Column(db.Integer, db.ForeignKey('lessons.id'), nullable=False)
    instructor_id = db.Column(db.Integer, db.ForeignKey('instructors.id'), nullable=True)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=True)
    day_of_week = db.Column(db.Integer, nullable=False)  # 0-4 (Monday-Friday)
    start_hour = db.Column(db.Integer, nullable=False)  # 0-9 (time slots)
    duration = db.Column(db.Integer, nullable=False)  # hours
    is_valid = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    lesson = db.relationship('Lesson', backref='schedule_entries', lazy=True)
    instructor = db.relationship('Instructor', backref='schedule_entries', lazy=True)
    classroom = db.relationship('Classroom', backref='schedule_entries', lazy=True)
    
    def to_dict(self):
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        start_time = f"{8 + self.start_hour//2}:{30 if self.start_hour%2 else '00'}"
        
        return {
            'id': self.id,
            'lesson_name': self.lesson.name,
            'lesson_code': self.lesson.code,
            'instructor_name': self.instructor.name if self.instructor else 'Unassigned',
            'classroom_name': self.classroom.name if self.classroom else 'Online',
            'day_name': day_names[self.day_of_week],
            'day_of_week': self.day_of_week,
            'start_hour': self.start_hour,
            'start_time': start_time,
            'duration': self.duration,
            'is_valid': self.is_valid,
            'created_at': self.created_at.isoformat()
        }
