# ===== models/database.py - Geliştirilmiş =====
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import JSON, Float

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
    code = db.Column(db.String(10), nullable=False, unique=True)  # Bölüm kodu (CS, EE, etc.)
    num_grades = db.Column(db.Integer, nullable=False)
    head_of_department = db.Column(db.String(200), nullable=True)  # Bölüm başkanı
    building = db.Column(db.String(100), nullable=True)  # Ana bina
    floor = db.Column(db.Integer, nullable=True)  # Ana kat
    phone = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(200), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    lessons = db.relationship('Lesson', backref='department_ref', lazy=True, cascade='all, delete-orphan')
    instructors = db.relationship('Instructor', backref='department_ref', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'num_grades': self.num_grades,
            'head_of_department': self.head_of_department,
            'building': self.building,
            'floor': self.floor,
            'phone': self.phone,
            'email': self.email,
            'is_active': self.is_active,
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
    semester = db.Column(db.Integer, nullable=False, default=1)  # 1=Güz, 2=Bahar
    theory_hours = db.Column(db.Integer, nullable=False, default=0)
    practice_hours = db.Column(db.Integer, nullable=False, default=0)
    lab_hours = db.Column(db.Integer, nullable=False, default=0)
    akts = db.Column(db.Integer, nullable=False, default=0)  # AKTS kredisi
    local_credit = db.Column(db.Float, nullable=False, default=0)  # Yerel kredi
    student_capacity = db.Column(db.Integer, nullable=False, default=40)
    min_capacity = db.Column(db.Integer, nullable=False, default=5)  # Minimum açılacak öğrenci sayısı
    is_online = db.Column(db.Boolean, nullable=False, default=False)
    requires_lab = db.Column(db.Boolean, nullable=False, default=False)
    requires_computer = db.Column(db.Boolean, nullable=False, default=False)
    requires_projector = db.Column(db.Boolean, nullable=False, default=True)
    is_elective = db.Column(db.Boolean, nullable=False, default=False)  # Seçmeli ders mi?
    prerequisite_ids = db.Column(JSON, nullable=True)  # Önkoşul ders ID'leri
    language = db.Column(db.String(20), nullable=False, default='Turkish')  # Ders dili
    difficulty = db.Column(db.Integer, nullable=False, default=3) # 1-5 (Kolay-Zor)
    exam_type = db.Column(db.String(50), nullable=True)  # Final, Vize+Final, Proje vb.
    is_active = db.Column(db.Boolean, nullable=False, default=True)
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
            'semester': self.semester,
            'theory_hours': self.theory_hours,
            'practice_hours': self.practice_hours,
            'lab_hours': self.lab_hours,
            'total_hours': self.total_hours,
            'akts': self.akts,
            'local_credit': self.local_credit,
            'student_capacity': self.student_capacity,
            'min_capacity': self.min_capacity,
            'is_online': self.is_online,
            'requires_lab': self.requires_lab,
            'requires_computer': self.requires_computer,
            'requires_projector': self.requires_projector,
            'is_elective': self.is_elective,
            'language': self.language,
            'exam_type': self.exam_type,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class Classroom(db.Model):
    __tablename__ = 'classrooms'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    code = db.Column(db.String(20), nullable=False, unique=True)  # A101, B205 gibi
    capacity = db.Column(db.Integer, nullable=False)
    exam_capacity = db.Column(db.Integer, nullable=True)  # Sınav kapasitesi (genelde normal kapasitenin yarısı)
    classroom_type = db.Column(db.String(50), nullable=False, default='Standard')  # Standard, Lab, Amphitheater, Conference
    has_lab = db.Column(db.Boolean, nullable=False, default=False)
    has_projector = db.Column(db.Boolean, nullable=False, default=True)
    has_computer = db.Column(db.Boolean, nullable=False, default=False)
    has_smartboard = db.Column(db.Boolean, nullable=False, default=False)
    has_air_conditioning = db.Column(db.Boolean, nullable=False, default=False)
    has_microphone = db.Column(db.Boolean, nullable=False, default=False)
    wifi_available = db.Column(db.Boolean, nullable=False, default=True)
    accessibility_features = db.Column(JSON, nullable=True)  # Engelli erişimi özellikleri
    technical_equipment = db.Column(JSON, nullable=True)  # Teknik ekipman listesi
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    is_bookable = db.Column(db.Boolean, nullable=False, default=True)  # Rezervasyon yapılabilir mi?
    maintenance_schedule = db.Column(JSON, nullable=True)  # Bakım programı
    building = db.Column(db.String(50), nullable=False)
    floor = db.Column(db.Integer, nullable=False)
    room_number = db.Column(db.String(20), nullable=True)
    department_priority = db.Column(JSON, nullable=True)  # Hangi bölümler öncelikli kullanır
    usage_cost_per_hour = db.Column(db.Float, nullable=True)  # Saatlik kullanım maliyeti
    cleaning_time_minutes = db.Column(db.Integer, nullable=False, default=15)  # Temizlik süresi
    setup_time_minutes = db.Column(db.Integer, nullable=False, default=5)  # Hazırlık süresi
    notes = db.Column(db.Text, nullable=True)
    location_coordinates = db.Column(JSON, nullable=True)  # GPS koordinatları
    photo_urls = db.Column(JSON, nullable=True)  # Derslik fotoğrafları
    last_maintenance_date = db.Column(db.DateTime, nullable=True)
    next_maintenance_date = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    availability = db.Column(JSON, nullable=True)  # Saatlik müsaitlik matrisi (10x5)
    
    # İlişkiler
    availability_exceptions = db.relationship('ClassroomAvailability', backref='classroom_ref', lazy=True, cascade='all, delete-orphan')
    
    @property
    def effective_capacity(self):
        """Sınav döneminde farklı kapasite kullanılabilir"""
        return self.exam_capacity if self.exam_capacity else self.capacity
    
    def is_suitable_for_lesson(self, lesson):
        """Dersin gereksinimlerine göre dersliğin uygunluğunu kontrol eder"""
        if not self.is_active or not self.is_bookable:
            return False
        
        if self.capacity < lesson.student_capacity:
            return False
            
        if lesson.requires_lab and not self.has_lab:
            return False
            
        if lesson.requires_computer and not self.has_computer:
            return False
            
        if lesson.requires_projector and not self.has_projector:
            return False
            
        return True
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'code': self.code,
            'capacity': self.capacity,
            'exam_capacity': self.exam_capacity,
            'classroom_type': self.classroom_type,
            'has_lab': self.has_lab,
            'has_projector': self.has_projector,
            'has_computer': self.has_computer,
            'has_smartboard': self.has_smartboard,
            'has_air_conditioning': self.has_air_conditioning,
            'has_microphone': self.has_microphone,
            'wifi_available': self.wifi_available,
            'accessibility_features': self.accessibility_features,
            'technical_equipment': self.technical_equipment,
            'is_active': self.is_active,
            'is_bookable': self.is_bookable,
            'building': self.building,
            'floor': self.floor,
            'room_number': self.room_number,
            'department_priority': self.department_priority,
            'usage_cost_per_hour': self.usage_cost_per_hour,
            'cleaning_time_minutes': self.cleaning_time_minutes,
            'setup_time_minutes': self.setup_time_minutes,
            'notes': self.notes,
            'location_coordinates': self.location_coordinates,
            'photo_urls': self.photo_urls,
            'last_maintenance_date': self.last_maintenance_date.isoformat() if self.last_maintenance_date else None,
            'next_maintenance_date': self.next_maintenance_date.isoformat() if self.next_maintenance_date else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class ClassroomAvailability(db.Model):
    """Derslik kullanım kısıtlamaları ve özel durumlar"""
    __tablename__ = 'classroom_availability'
    
    id = db.Column(db.Integer, primary_key=True)
    classroom_id = db.Column(db.Integer, db.ForeignKey('classrooms.id'), nullable=False)
    date_start = db.Column(db.DateTime, nullable=False)
    date_end = db.Column(db.DateTime, nullable=False)
    availability_type = db.Column(db.String(50), nullable=False)  # maintenance, reserved, blocked, available
    reason = db.Column(db.String(200), nullable=True)
    contact_person = db.Column(db.String(200), nullable=True)
    recurring = db.Column(db.Boolean, default=False)  # Tekrarlayan kısıtlama mı?
    recurring_pattern = db.Column(JSON, nullable=True)  # Tekrarlama deseni
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Instructor(db.Model):
    __tablename__ = 'instructors'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    employee_id = db.Column(db.String(50), nullable=True, unique=True)  # Personel numarası
    email = db.Column(db.String(200), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    title = db.Column(db.String(50), nullable=True)
    academic_degree = db.Column(db.String(50), nullable=True)  # PhD, MSc, BSc
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    office_location = db.Column(db.String(100), nullable=True)  # Ofis konumu
    specialization = db.Column(JSON, nullable=True)  # Uzmanlık alanları
    languages = db.Column(JSON, nullable=True)  # Konuştuğu diller
    max_daily_hours = db.Column(db.Integer, nullable=False, default=8)
    max_weekly_hours = db.Column(db.Integer, nullable=False, default=30)
    preferred_days = db.Column(JSON, nullable=True)  # Tercih ettiği günler
    preferred_times = db.Column(JSON, nullable=True)  # Tercih ettiği saatler
    availability = db.Column(JSON, nullable=False)  # 10x5 matrix (saat x gün)
    overtime_rate = db.Column(db.Float, nullable=True)  # Fazla mesai ücreti
    contract_type = db.Column(db.String(50), nullable=False, default='Full-time')  # Full-time, Part-time, Visiting
    start_date = db.Column(db.DateTime, nullable=True)  # İşe başlama tarihi
    end_date = db.Column(db.DateTime, nullable=True)  # Sözleşme bitiş tarihi
    sabbatical_periods = db.Column(JSON, nullable=True)  # Araştırma izni dönemleri
    teaching_load_factor = db.Column(db.Float, nullable=False, default=1.0)  # Ders yükü katsayısı
    research_activities = db.Column(JSON, nullable=True)  # Araştırma faaliyetleri
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    is_available = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    lesson_assignments = db.relationship('InstructorLesson', backref='instructor_ref', lazy=True, cascade='all, delete-orphan')
    
    def get_current_weekly_hours(self):
        """Mevcut haftalık ders yükünü hesapla"""
        total_hours = 0
        for assignment in self.lesson_assignments:
            if assignment.lesson.is_active:
                total_hours += assignment.lesson.total_hours
        return total_hours
    
    def can_teach_lesson(self, lesson):
        """Akademisyenin dersi verebilip veremediğini kontrol eder"""
        if not self.is_active or not self.is_available:
            return False
            
        # Bölüm kontrolü
        if self.department_id != lesson.department_id:
            return False
            
        # Dil kontrolü
        if self.languages and lesson.language not in self.languages:
            return False
            
        # Uzmanlık alanı kontrolü
        assignment = next((ia for ia in self.lesson_assignments if ia.lesson_id == lesson.id), None)
        return assignment is not None
    
    
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'employee_id': self.employee_id,
            'email': self.email,
            'phone': self.phone,
            'title': self.title,
            'academic_degree': self.academic_degree,
            'department_id': self.department_id,
            'department_name': self.department_ref.name,
            'office_location': self.office_location,
            'specialization': self.specialization,
            'languages': self.languages,
            'max_daily_hours': self.max_daily_hours,
            'max_weekly_hours': self.max_weekly_hours,
            'preferred_days': self.preferred_days,
            'preferred_times': self.preferred_times,
            'availability': self.availability,
            'overtime_rate': self.overtime_rate,
            'contract_type': self.contract_type,
            'teaching_load_factor': self.teaching_load_factor,
            'current_weekly_hours': self.get_current_weekly_hours(),
            'is_active': self.is_active,
            'is_available': self.is_available,
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
    experience_years = db.Column(db.Integer, nullable=False, default=0)  # Bu dersi kaç yıldır veriyor
    last_taught_semester = db.Column(db.String(20), nullable=True)  # En son hangi dönem verdi
    teaching_evaluation_score = db.Column(db.Float, nullable=True)  # Ders değerlendirme puanı
    can_coordinate = db.Column(db.Boolean, nullable=False, default=False)  # Koordinatörlük yapabilir mi?
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    lesson = db.relationship('Lesson', backref='instructor_assignments', lazy=True)
    
    __table_args__ = (db.UniqueConstraint('instructor_id', 'lesson_id'),)

class OptimizationRun(db.Model):
    __tablename__ = 'optimization_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False, unique=True)
    department_id = db.Column(db.Integer, db.ForeignKey('departments.id'), nullable=False)
    semester = db.Column(db.Integer, nullable=False)  # 1=Güz, 2=Bahar
    academic_year = db.Column(db.String(9), nullable=False)  # 2023-2024
    parameters = db.Column(JSON, nullable=False)
    constraints = db.Column(JSON, nullable=True)  # Ek kısıtlamalar
    objectives = db.Column(JSON, nullable=True)  # Optimizasyon hedefleri
    status = db.Column(db.String(20), nullable=False, default='initialized')
    progress = db.Column(JSON, nullable=True)
    results = db.Column(JSON, nullable=True)
    fitness_score = db.Column(db.Float, nullable=True)
    conflicts_count = db.Column(db.Integer, nullable=True)
    room_utilization = db.Column(db.Float, nullable=True)
    instructor_balance = db.Column(db.Float, nullable=True)
    student_satisfaction = db.Column(db.Float, nullable=True)
    runtime_seconds = db.Column(db.Float, nullable=True)
    memory_usage_mb = db.Column(db.Float, nullable=True)
    created_by = db.Column(db.String(200), nullable=True)  # Kullanıcı bilgisi
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
            'semester': self.semester,
            'academic_year': self.academic_year,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'objectives': self.objectives,
            'status': self.status,
            'progress': self.progress,
            'results': self.results,
            'fitness_score': self.fitness_score,
            'conflicts_count': self.conflicts_count,
            'room_utilization': self.room_utilization,
            'instructor_balance': self.instructor_balance,
            'student_satisfaction': self.student_satisfaction,
            'runtime_seconds': self.runtime_seconds,
            'memory_usage_mb': self.memory_usage_mb,
            'created_by': self.created_by,
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
    lesson_type = db.Column(db.String(20), nullable=False, default='Theory')  # Theory, Practice, Lab
    group_number = db.Column(db.Integer, nullable=True)  # Grup numarası (lab/practice için)
    student_count = db.Column(db.Integer, nullable=True)  # Kayıtlı öğrenci sayısı
    is_valid = db.Column(db.Boolean, nullable=False, default=True)
    conflict_reasons = db.Column(JSON, nullable=True)  # Çakışma nedenleri
    quality_score = db.Column(db.Float, nullable=True)  # Ders programının kalite puanı
    preference_satisfaction = db.Column(db.Float, nullable=True)  # Tercih memnuniyeti
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    lesson = db.relationship('Lesson', backref='schedule_entries', lazy=True)
    instructor = db.relationship('Instructor', backref='schedule_entries', lazy=True)
    classroom = db.relationship('Classroom', backref='schedule_entries', lazy=True)
    
    def to_dict(self):
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        start_time = f"{8 + self.start_hour//2}:{30 if self.start_hour%2 else '00'}"
        end_hour = self.start_hour + self.duration
        end_time = f"{8 + end_hour//2}:{30 if end_hour%2 else '00'}"
        
        return {
            'id': self.id,
            'lesson_name': self.lesson.name,
            'lesson_code': self.lesson.code,
            'lesson_type': self.lesson_type,
            'instructor_name': self.instructor.name if self.instructor else 'Unassigned',
            'instructor_title': self.instructor.title if self.instructor else None,
            'classroom_name': self.classroom.name if self.classroom else 'Online',
            'classroom_building': self.classroom.building if self.classroom else None,
            'classroom_capacity': self.classroom.capacity if self.classroom else None,
            'day_name': day_names[self.day_of_week],
            'day_of_week': self.day_of_week,
            'start_hour': self.start_hour,
            'start_time': start_time,
            'end_time': end_time,
            'duration': self.duration,
            'group_number': self.group_number,
            'student_count': self.student_count,
            'is_valid': self.is_valid,
            'conflict_reasons': self.conflict_reasons,
            'quality_score': self.quality_score,
            'preference_satisfaction': self.preference_satisfaction,
            'created_at': self.created_at.isoformat()
        }