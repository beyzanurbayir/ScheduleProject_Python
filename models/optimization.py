# ===== models/optimization.py - Geliştirilmiş =====
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import random
import numpy as np
from copy import deepcopy
import time
import logging
# Yeni Doğru Satır
from models.database import Faculty , Department, db, Lesson, Instructor, Classroom, OptimizationRun, Schedule, ClassroomAvailability
from datetime import datetime, timedelta

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    population_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    tournament_size: int = 5
    elitism_rate: float = 0.2
    
    # Advanced parameters
    adaptive_mutation: bool = True
    diversity_threshold: float = 0.1
    stagnation_limit: int = 20
    local_search_probability: float = 0.3

    # Simulated Annealing for Local Search
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    
    # Fitness weights - normalized to sum to 100
    conflict_penalty: float = 40.0          # Hard constraints (Ağırlığı arttı)
    room_utilization_weight: float = 15.0   # Efficient room usage (Ağırlığı azaldı)
    instructor_balance_weight: float = 15.0  # Workload distribution
    preference_weight: float = 10.0         # Instructor preferences (Ağırlığı azaldı)
    time_distribution_weight: float = 5.0  # Daily/weekly distribution (Ağırlığı azaldı)
    student_satisfaction_weight: float = 15.0 # Student schedule quality (AĞIRLIĞI ARTTI)

class AdvancedScheduleOptimizer:
    #GÜNCEELLENDİ
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.time_slots = 10  # 8:30-18:30 (30-min slots)
        self.days = 5  # Monday-Friday
        self.generation_stats = []
        self.best_fitness_history = []
        self.diversity_history = []
        self.current_temperature = self.config.initial_temperature # YENİ SATIR
        
    def optimize_schedule(self, department_id: int, semester: int, session_id: str, 
                         progress_callback=None) -> OptimizationRun:
        """Enhanced genetic algorithm with advanced features"""
        
        start_time = time.time()
        
        # Create optimization run record
        opt_run = OptimizationRun(
            session_id=session_id,
            department_id=department_id,
            semester=semester,
            academic_year=self._get_current_academic_year(),
            parameters=self.config.__dict__.copy(),
            status='running',
            created_by='System'
        )
        db.session.add(opt_run)
        db.session.commit()
        
        try:
            # Enhanced data loading
            lessons = self._get_lessons(department_id, semester)
            instructors = self._get_instructors(department_id)
            classrooms = self._get_available_classrooms()
            
            if not lessons:
                raise ValueError("No lessons found for the specified department and semester")
            
            logger.info(f"Starting optimization: {len(lessons)} lessons, {len(instructors)} instructors, {len(classrooms)} classrooms")
            
            # Run enhanced genetic algorithm
            best_schedule, best_fitness, generation_data = self._enhanced_genetic_algorithm(
                lessons, instructors, classrooms, opt_run, progress_callback
            )
            
            # Comprehensive evaluation
            evaluation = self._comprehensive_evaluation(best_schedule, lessons, instructors, classrooms)
            
            # Save results
            runtime = time.time() - start_time
            
            opt_run.status = 'completed'
            opt_run.fitness_score = best_fitness
            opt_run.conflicts_count = evaluation['conflicts']
            opt_run.room_utilization = evaluation['room_utilization']
            opt_run.instructor_balance = evaluation['instructor_balance']
            opt_run.student_satisfaction = evaluation['student_satisfaction']
            opt_run.runtime_seconds = runtime
            opt_run.completed_at = datetime.utcnow()
            opt_run.results = {
                'evaluation': evaluation,
                'generation_stats': generation_data[-10:],
                'optimization_summary': self._create_optimization_summary(evaluation)
            }
            
            # Save schedule to database with enhanced details
            self._save_enhanced_schedule_to_db(opt_run, best_schedule)
            
            db.session.commit()
            logger.info(f"Optimization completed in {runtime:.2f}s with fitness {best_fitness:.2f}")
            return opt_run
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            opt_run.status = 'error'
            opt_run.results = {'error': str(e)}
            db.session.commit()
            raise e
    
    def _enhanced_genetic_algorithm(self, lessons, instructors, classrooms, opt_run, progress_callback):
        """Enhanced genetic algorithm with adaptive parameters"""
        
        # Initialize population with diverse strategies
        population = self._create_diverse_population(lessons, instructors, classrooms)
        
        best_schedule = None
        best_fitness = float('-inf')
        stagnation_counter = 0
        generation_data = []
        
        for generation in range(self.config.generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = self._enhanced_fitness_function(individual, lessons, instructors, classrooms)
                fitness_scores.append(fitness)
            
            # Update best solution
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_schedule = deepcopy(population[gen_best_idx])
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Calculate diversity
            diversity = self._calculate_population_diversity(population)
            
            # Store generation statistics
            gen_stats = {
                'generation': generation,
                'best_fitness': gen_best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'diversity': diversity,
                'conflicts': self._count_hard_conflicts(population[gen_best_idx]),
                'stagnation': stagnation_counter
            }
            generation_data.append(gen_stats)
            
            # Update database progress
            opt_run.progress = {
                'generation': generation,
                'total_generations': self.config.generations,
                'best_fitness': gen_best_fitness,
                'conflicts': gen_stats['conflicts'],
                'diversity': diversity
            }
            db.session.commit()
            
            if progress_callback:
                progress_callback(gen_stats)
            
            # Adaptive parameter adjustment
            if self.config.adaptive_mutation:
                self._adjust_parameters(diversity, stagnation_counter)
            
            # Early termination conditions
            if gen_stats['conflicts'] == 0 and gen_best_fitness > 950:  # Near-optimal solution
                logger.info(f"Early termination at generation {generation} - optimal solution found")
                break
            
            # Create new population
            population = self._create_new_population(
                population, fitness_scores, lessons, instructors, classrooms
            )
            
            # Apply local search to promising individuals
            if random.random() < self.config.local_search_probability:
                population = self._apply_local_search(population, lessons, instructors, classrooms)

            # Cool down the temperature for simulated annealing
            self.current_temperature *= self.config.cooling_rate # YENİ SATIR

        self.generation_stats = generation_data
        return best_schedule, best_fitness, generation_data
    
    #GÜNCELLENDİ
    def _create_diverse_population(self, lessons, instructors, classrooms):
        """Create initial population with diverse construction strategies"""
        population = []
        
        for i in range(self.config.population_size):
            if i < self.config.population_size * 0.25:  # 25% - Graph Coloring Heuristic (YENİ)
                schedule = self._create_graph_coloring_schedule(lessons, instructors, classrooms)
            elif i < self.config.population_size * 0.5:  # 25% - Priority-based construction
                schedule = self._create_priority_based_schedule(lessons, instructors, classrooms)
            elif i < self.config.population_size * 0.75:  # 25% - Greedy construction
                schedule = self._create_greedy_schedule(lessons, instructors, classrooms)
            else:  # 25% - Random construction
                schedule = self._create_random_schedule(lessons, instructors, classrooms)
            
            population.append(schedule)
        
        # Constraint-focused'ı kaldırdık çünkü Graph Coloring benzer bir amaca hizmet ediyor.
        return population
    
    def _create_priority_based_schedule(self, lessons, instructors, classrooms):
        """Create schedule prioritizing difficult-to-place lessons"""
        schedule = {}
        
        # Sort lessons by difficulty (constraints, requirements, capacity)
        sorted_lessons = sorted(lessons, key=lambda l: self._calculate_lesson_difficulty(l), reverse=True)
        
        for lesson in sorted_lessons:
            placement = self._find_best_placement(lesson, schedule, instructors, classrooms)
            if placement:
                schedule[lesson.id] = placement
        
        return schedule
    
    def _create_greedy_schedule(self, lessons, instructors, classrooms):
        """Create schedule using greedy approach - best local choices"""
        schedule = {}
        
        # Sort lessons by total hours (longer lessons first)
        sorted_lessons = sorted(lessons, key=lambda l: l.total_hours, reverse=True)
        
        for lesson in sorted_lessons:
            best_placement = None
            best_score = float('-inf')
            
            # Try all possible placements and pick the best one
            for day in range(self.days):
                for start_hour in range(self.time_slots - lesson.total_hours + 1):
                    for instructor in instructors:
                        for classroom in classrooms:
                            if self._is_valid_placement(schedule, lesson, instructor, classroom, start_hour, day):
                                score = self._evaluate_placement(lesson, instructor, classroom, start_hour, day, schedule)
                                if score > best_score:
                                    best_score = score
                                    best_placement = {
                                        'lesson': lesson,
                                        'instructor': instructor,
                                        'classroom': classroom,
                                        'start_hour': start_hour,
                                        'day': day,
                                        'duration': lesson.total_hours
                                    }
            
            if best_placement:
                schedule[lesson.id] = best_placement
        
        return schedule
    
    def _create_constraint_focused_schedule(self, lessons, instructors, classrooms):
        """Create schedule focusing on constraint satisfaction"""
        schedule = {}
        
        # Group lessons by constraints
        lab_lessons = [l for l in lessons if l.requires_lab]
        computer_lessons = [l for l in lessons if l.requires_computer]
        large_lessons = [l for l in lessons if l.student_capacity > 100]
        regular_lessons = [l for l in lessons if l not in lab_lessons + computer_lessons + large_lessons]
        
        # Schedule in order of constraint difficulty
        for lesson_group in [lab_lessons, computer_lessons, large_lessons, regular_lessons]:
            for lesson in lesson_group:
                placement = self._find_constraint_satisfying_placement(lesson, schedule, instructors, classrooms)
                if placement:
                    schedule[lesson.id] = placement
        
        return schedule
    
    # YENİ EKLENEN METOD

    def _create_graph_coloring_schedule(self, lessons, instructors, classrooms):
        """Creates a schedule using a greedy graph coloring approach to minimize initial conflicts."""
        schedule = {}
        
        # 1. Create the conflict graph
        # Key: lesson_id, Value: set of conflicting lesson_ids
        conflict_graph = {lesson.id: set() for lesson in lessons}
        # Group lessons by grade
        lessons_by_grade = {}
        for lesson in lessons:
            if lesson.grade not in lessons_by_grade:
                lessons_by_grade[lesson.grade] = []
            lessons_by_grade[lesson.grade].append(lesson)

        # Assume lessons for the same grade conflict with each other
        for grade, grade_lessons in lessons_by_grade.items():
            for i in range(len(grade_lessons)):
                for j in range(i + 1, len(grade_lessons)):
                    l1_id = grade_lessons[i].id
                    l2_id = grade_lessons[j].id
                    conflict_graph[l1_id].add(l2_id)
                    conflict_graph[l2_id].add(l1_id)

        # 2. Sort nodes (lessons) by degree (number of conflicts) in descending order
        sorted_lessons = sorted(lessons, key=lambda l: len(conflict_graph[l.id]), reverse=True)
        
        # 3. Color the graph (assign time slots)
        # Key: lesson_id, Value: color (time_slot_index)
        lesson_colors = {}
        total_slots = self.days * self.time_slots

        for lesson in sorted_lessons:
            # Find the first available color (time slot)
            used_colors = {lesson_colors[neighbor_id] for neighbor_id in conflict_graph[lesson.id] if neighbor_id in lesson_colors}
            
            for color in range(total_slots):
                if color not in used_colors:
                    lesson_colors[lesson.id] = color
                    break
            else:
                # If no color is available (should be rare with enough slots), assign a random one
                lesson_colors[lesson.id] = random.randint(0, total_slots - 1)

        # 4. Build the final schedule from the colored graph
        for lesson in lessons:
            if lesson.id in lesson_colors:
                color = lesson_colors[lesson.id]
                day = color // self.time_slots
                start_hour = color % self.time_slots

                # Find a suitable instructor and classroom (simple greedy search)
                instructor = next((i for i in instructors if i.department_id == lesson.department_id), random.choice(instructors))
                classroom = None
                if not lesson.is_online:
                    suitable_classrooms = [c for c in classrooms if c.is_suitable_for_lesson(lesson)]
                    if suitable_classrooms:
                        classroom = random.choice(suitable_classrooms)
                    else: # Fallback if no perfectly suitable classroom found
                        classroom = random.choice([c for c in classrooms if c.capacity >= lesson.student_capacity])

                # Check placement validity (ignoring student group conflicts as graph handles it)
                temp_schedule = {k: v for k, v in schedule.items() if k != lesson.id}
                if self._is_valid_placement(temp_schedule, lesson, instructor, classroom, start_hour, day):
                     schedule[lesson.id] = {
                        'lesson': lesson, 'instructor': instructor, 'classroom': classroom,
                        'start_hour': start_hour, 'day': day, 'duration': lesson.total_hours
                    }

        return schedule


    def _create_random_schedule(self, lessons, instructors, classrooms):
        """Create completely random schedule"""
        schedule = {}
        
        shuffled_lessons = lessons.copy()
        random.shuffle(shuffled_lessons)
        
        for lesson in shuffled_lessons:
            max_attempts = 50
            for _ in range(max_attempts):
                day = random.randint(0, self.days - 1)
                start_hour = random.randint(0, max(0, self.time_slots - lesson.total_hours))
                instructor = random.choice(instructors)
                classroom = random.choice(classrooms) if not lesson.is_online else None
                
                if self._is_valid_placement(schedule, lesson, instructor, classroom, start_hour, day):
                    schedule[lesson.id] = {
                        'lesson': lesson,
                        'instructor': instructor,
                        'classroom': classroom,
                        'start_hour': start_hour,
                        'day': day,
                        'duration': lesson.total_hours
                    }
                    break
        
        return schedule
    
    def _calculate_lesson_difficulty(self, lesson):
        """Calculate how difficult a lesson is to schedule"""
        difficulty = 0
        
        difficulty += lesson.total_hours * 2  # Longer lessons are harder
        difficulty += 10 if lesson.requires_lab else 0
        difficulty += 5 if lesson.requires_computer else 0
        difficulty += lesson.student_capacity // 20  # Large capacity lessons are harder
        difficulty += 5 if lesson.is_elective else 0  # Electives have timing constraints
        
        return difficulty
    
    def _find_best_placement(self, lesson, schedule, instructors, classrooms):
        """Find the best placement for a lesson using heuristics"""
        suitable_instructors = [i for i in instructors if i.can_teach_lesson(lesson)]
        suitable_classrooms = [c for c in classrooms if c.is_suitable_for_lesson(lesson)]
        
        if not suitable_instructors:
            suitable_instructors = [i for i in instructors if i.department_id == lesson.department_id]
        
        if not suitable_classrooms and not lesson.is_online:
            suitable_classrooms = [c for c in classrooms if c.capacity >= lesson.student_capacity]
        
        best_placement = None
        best_score = float('-inf')
        
        for instructor in suitable_instructors[:3]:  # Limit search to top candidates
            for classroom in (suitable_classrooms[:3] if suitable_classrooms else [None]):
                for day in range(self.days):
                    for start_hour in range(max(1, self.time_slots - lesson.total_hours)):
                        if self._is_valid_placement(schedule, lesson, instructor, classroom, start_hour, day):
                            score = self._evaluate_placement(lesson, instructor, classroom, start_hour, day, schedule)
                            if score > best_score:
                                best_score = score
                                best_placement = {
                                    'lesson': lesson,
                                    'instructor': instructor,
                                    'classroom': classroom,
                                    'start_hour': start_hour,
                                    'day': day,
                                    'duration': lesson.total_hours
                                }
        
        return best_placement
    
    def _evaluate_placement(self, lesson, instructor, classroom, start_hour, day, schedule):
        """Evaluate the quality of a specific placement"""
        score = 100  # Base score
        
        # Instructor preference
        if instructor:
            assignment = next((ia for ia in instructor.lesson_assignments if ia.lesson_id == lesson.id), None)
            if assignment:
                score += assignment.preference_level * 5
                score += assignment.competency_level * 3
        
        # Time preference (avoid very early or very late)
        if start_hour < 2 or start_hour > 7:  # Before 9:30 or after 16:30
            score -= 10
        
        # Classroom suitability
        if classroom:
            capacity_ratio = lesson.student_capacity / classroom.capacity
            if 0.7 <= capacity_ratio <= 0.9:  # Good capacity utilization
                score += 15
            elif capacity_ratio > 1.0:  # Overcrowded
                score -= 30
        
        # Day distribution (prefer spreading across week)
        lessons_on_day = sum(1 for p in schedule.values() if p.get('day') == day)
        if lessons_on_day > 3:
            score -= lessons_on_day * 5
        
        return score
    
    def _enhanced_fitness_function(self, schedule, lessons, instructors, classrooms):
        """Comprehensive fitness function with multiple objectives"""
        if not schedule:
            return 0
        
        # Initialize base fitness
        fitness = 1000
        
        # Hard constraints (critical penalties)
        conflicts = self._count_hard_conflicts(schedule)
        fitness -= conflicts * self.config.conflict_penalty
        
        # Soft constraints (optimization objectives)
        room_util = self._calculate_room_utilization_advanced(schedule, classrooms)
        fitness += room_util * self.config.room_utilization_weight
        
        instructor_balance = self._calculate_instructor_balance_advanced(schedule, instructors)
        fitness += instructor_balance * self.config.instructor_balance_weight
        
        preference_score = self._calculate_preference_satisfaction(schedule)
        fitness += preference_score * self.config.preference_weight
        
        time_distribution = self._calculate_time_distribution_quality(schedule)
        fitness += time_distribution * self.config.time_distribution_weight
        
        student_satisfaction = self._calculate_student_satisfaction(schedule, lessons)
        fitness += student_satisfaction * self.config.student_satisfaction_weight
        
        return max(0, fitness)
    
    def _count_hard_conflicts(self, schedule):
        """Count all hard constraint violations"""
        conflicts = 0
        time_slots_used = {}  # Track what's scheduled when
        
        for lesson_id, placement in schedule.items():
            if placement.get('forced'):
                conflicts += 5  # Penalize forced placements
                continue
            
            lesson = placement['lesson']
            instructor = placement.get('instructor')
            classroom = placement.get('classroom')
            day = placement['day']
            start_hour = placement['start_hour']
            duration = placement['duration']
            
            # Check each time slot used by this lesson
            for hour_offset in range(duration):
                current_hour = start_hour + hour_offset
                time_key = (day, current_hour)
                
                # Initialize time slot tracking
                if time_key not in time_slots_used:
                    time_slots_used[time_key] = {'instructors': set(), 'classrooms': set()}
                
                # Instructor conflicts
                if instructor:
                    if instructor.id in time_slots_used[time_key]['instructors']:
                        conflicts += 1
                    time_slots_used[time_key]['instructors'].add(instructor.id)
                    
                    # Check instructor availability
                    if (instructor.availability and 
                        current_hour < len(instructor.availability) and 
                        day < len(instructor.availability[current_hour]) and
                        not instructor.availability[current_hour][day]):
                        conflicts += 1
                
                # Classroom conflicts
                if classroom:
                    if classroom.id in time_slots_used[time_key]['classrooms']:
                        conflicts += 1
                    time_slots_used[time_key]['classrooms'].add(classroom.id)
                
                # Capacity violations
                if classroom and classroom.capacity < lesson.student_capacity:
                    conflicts += 1
                
                # Equipment requirements
                if lesson.requires_lab and (not classroom or not classroom.has_lab):
                    conflicts += 1
                if lesson.requires_computer and (not classroom or not classroom.has_computer):
                    conflicts += 1
        
        return conflicts
    
    def _calculate_room_utilization_advanced(self, schedule, classrooms):
        """Calculate advanced room utilization metrics"""
        if not classrooms:
            return 0
        
        total_possible_hours = len(classrooms) * self.time_slots * self.days
        classroom_usage = {}
        
        for placement in schedule.values():
            if placement.get('classroom') and not placement.get('forced'):
                classroom_id = placement['classroom'].id
                classroom_usage[classroom_id] = classroom_usage.get(classroom_id, 0) + placement['duration']
        
        # Calculate utilization rate
        total_used_hours = sum(classroom_usage.values())
        utilization_rate = (total_used_hours / total_possible_hours) * 100 if total_possible_hours > 0 else 0
        
        # Bonus for balanced usage across classrooms
        if classroom_usage:
            usage_values = list(classroom_usage.values())
            mean_usage = np.mean(usage_values)
            variance = np.var(usage_values)
            balance_bonus = max(0, 20 - variance/mean_usage) if mean_usage > 0 else 0
            utilization_rate += balance_bonus
        
        return min(utilization_rate, 100)
    
    def _calculate_instructor_balance_advanced(self, schedule, instructors):
        """Calculate instructor workload balance with advanced metrics"""
        instructor_loads = {}
        instructor_daily_loads = {}
        
        # Calculate loads
        for placement in schedule.values():
            if placement.get('instructor') and not placement.get('forced'):
                instructor_id = placement['instructor'].id
                day = placement['day']
                duration = placement['duration']
                
                # Weekly load
                instructor_loads[instructor_id] = instructor_loads.get(instructor_id, 0) + duration
                
                # Daily load tracking
                if instructor_id not in instructor_daily_loads:
                    instructor_daily_loads[instructor_id] = [0] * self.days
                instructor_daily_loads[instructor_id][day] += duration
        
        if not instructor_loads:
            return 0
        
        # Calculate balance metrics
        loads = list(instructor_loads.values())
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        
        # Balance score (lower std deviation is better)
        balance_score = max(0, 50 - std_load * 2)
        
        # Check daily hour limits
        daily_violations = 0
        for instructor_id, daily_loads in instructor_daily_loads.items():
            instructor = next((i for i in instructors if i.id == instructor_id), None)
            if instructor:
                for daily_load in daily_loads:
                    if daily_load > instructor.max_daily_hours:
                        daily_violations += 1
        
        balance_score -= daily_violations * 5
        
        return max(0, balance_score)
    
    def _calculate_preference_satisfaction(self, schedule):
        """Calculate how well instructor preferences are satisfied"""
        total_satisfaction = 0
        assignment_count = 0
        
        for placement in schedule.values():
            if placement.get('instructor') and not placement.get('forced'):
                instructor = placement['instructor']
                lesson = placement['lesson']
                
                # Find assignment preferences
                assignment = next((ia for ia in instructor.lesson_assignments if ia.lesson_id == lesson.id), None)
                if assignment:
                    # Weight by competency and preference
                    satisfaction = (assignment.preference_level + assignment.competency_level) / 2
                    total_satisfaction += satisfaction
                    assignment_count += 1
                
                # Time preferences
                day = placement['day']
                start_hour = placement['start_hour']
                
                # Check preferred days
                if instructor.preferred_days and day in instructor.preferred_days:
                    total_satisfaction += 2
                
                # Check preferred times
                if instructor.preferred_times:
                    for time_range in instructor.preferred_times:
                        if time_range.get('start', 0) <= start_hour <= time_range.get('end', 9):
                            total_satisfaction += 1.5
        
        return (total_satisfaction / assignment_count * 10) if assignment_count > 0 else 0
    
    def _calculate_time_distribution_quality(self, schedule):
        """Calculate quality of time distribution"""
        daily_counts = [0] * self.days
        hourly_counts = [0] * self.time_slots
        
        for placement in schedule.values():
            if not placement.get('forced'):
                daily_counts[placement['day']] += 1
                start_hour = placement['start_hour']
                for h in range(placement['duration']):
                    if start_hour + h < self.time_slots:
                        hourly_counts[start_hour + h] += 1
        
        # Calculate distribution quality
        daily_variance = np.var(daily_counts) if daily_counts else 0
        hourly_variance = np.var(hourly_counts) if hourly_counts else 0
        
        # Lower variance is better (more even distribution)
        daily_score = max(0, 25 - daily_variance)
        hourly_score = max(0, 25 - hourly_variance * 0.5)
        
        return daily_score + hourly_score
    
    # _calculate_student_satisfaction fonksiyonu değiştirildi
    def _calculate_student_satisfaction(self, schedule, lessons):
        """Estimate student satisfaction based on schedule quality (ADVANCED)"""
        if not lessons:
            return 0

        # Her sınıf seviyesi için günlük programları grupla
        grade_schedules = {}
        for lesson in lessons:
            if lesson.grade not in grade_schedules:
                grade_schedules[lesson.grade] = [[] for _ in range(self.days)]

        for placement in schedule.values():
            lesson = placement['lesson']
            day = placement['day']
            start_hour = placement['start_hour']
            duration = placement['duration']
            
            if lesson.grade in grade_schedules:
                grade_schedules[lesson.grade][day].append({
                    'start': start_hour,
                    'end': start_hour + duration,
                    'difficulty': lesson.difficulty
                })

        total_score = 0
        grade_count = len(grade_schedules)

        if grade_count == 0:
            return 50 # Nötr puan

        # Her sınıf seviyesinin programını değerlendir
        for grade, daily_schedules in grade_schedules.items():
            grade_score = 100 # Her sınıf 100 puan üzerinden başlar
            
            for day_schedule in daily_schedules:
                if not day_schedule:
                    continue

                # O gün sadece 1 ders varsa ceza uygula (E3)
                if len(day_schedule) == 1:
                    grade_score -= 15

                # Gün içi boş zamanı ve zor dersleri değerlendir
                if len(day_schedule) > 1:
                    day_schedule.sort(key=lambda x: x['start'])
                    
                    # İlk ve son ders arasındaki toplam süreyi kontrol et (A7)
                    first_lesson_start = day_schedule[0]['start']
                    last_lesson_end = day_schedule[-1]['end']
                    total_span = last_lesson_end - first_lesson_start
                    
                    if total_span > 6: # 3 saatten fazla yayılma varsa
                        grade_score -= (total_span - 6) * 2 # Her yarım saatlik ek yayılma için ceza

                    # Dersler arası boşlukları kontrol et
                    for i in range(len(day_schedule) - 1):
                        gap = day_schedule[i+1]['start'] - day_schedule[i]['end']
                        if gap > 4: # 2 saatten uzun boşluk varsa
                            grade_score -= 10
                
                # Zor derslerin sabah saatlerinde olmasını ödüllendir (E11)
                for lesson_slot in day_schedule:
                    # Zorluk 4 veya 5 ise ve ders sabah saatlerindeyse (ilk 4 slot)
                    if lesson_slot['difficulty'] >= 4 and lesson_slot['start'] < 4:
                        grade_score += 5 * (lesson_slot['difficulty'] - 3) # Zorluk 4 için +5, 5 için +10 puan
            
            total_score += max(0, grade_score) # Negatif puana düşmesini engelle

        # Ortalamayı al ve 0-100 aralığına normalize et
        final_satisfaction = (total_score / grade_count)
        return max(0, min(final_satisfaction, 100))
    
    def _get_current_academic_year(self):
        """Get current academic year string"""
        now = datetime.now()
        if now.month >= 9:  # September or later
            return f"{now.year}-{now.year + 1}"
        else:
            return f"{now.year - 1}-{now.year}"
    
    def _get_lessons(self, department_id: int, semester: int) -> List[Lesson]:
        return Lesson.query.filter_by(
            department_id=department_id,
            semester=semester,
            is_active=True
        ).all()
    
    def _get_instructors(self, department_id: int) -> List[Instructor]:
        return Instructor.query.filter_by(
            department_id=department_id,
            is_active=True,
            is_available=True
        ).all()
    
    def _get_available_classrooms(self) -> List[Classroom]:
        return Classroom.query.filter_by(
            is_active=True,
            is_bookable=True
        ).all()
    
    def _comprehensive_evaluation(self, schedule, lessons, instructors, classrooms):
        """Perform comprehensive evaluation of the final schedule"""
        evaluation = {
            'conflicts': self._count_hard_conflicts(schedule),
            'room_utilization': self._calculate_room_utilization_advanced(schedule, classrooms),
            'instructor_balance': self._calculate_instructor_balance_advanced(schedule, instructors),
            'preference_satisfaction': self._calculate_preference_satisfaction(schedule),
            'time_distribution': self._calculate_time_distribution_quality(schedule),
            'student_satisfaction': self._calculate_student_satisfaction(schedule, lessons),
            'lessons_scheduled': len([p for p in schedule.values() if not p.get('forced')]),
            'total_lessons': len(lessons),
            'scheduling_success_rate': len([p for p in schedule.values() if not p.get('forced')]) / len(lessons) * 100 if lessons else 0
        }
        
        return evaluation
    
    def _create_optimization_summary(self, evaluation):
        """Create a summary of optimization results"""
        return {
            'quality_grade': self._calculate_quality_grade(evaluation),
            'recommendations': self._generate_recommendations(evaluation),
            'key_metrics': {
                'conflicts': evaluation['conflicts'],
                'success_rate': evaluation['scheduling_success_rate'],
                'room_efficiency': evaluation['room_utilization'],
                'instructor_balance': evaluation['instructor_balance']
            }
        }
    
    def _calculate_quality_grade(self, evaluation):
        """Calculate overall quality grade A-F"""
        score = 0
        score += 30 if evaluation['conflicts'] == 0 else max(0, 30 - evaluation['conflicts'] * 5)
        score += evaluation['room_utilization'] * 0.25
        score += evaluation['instructor_balance'] * 0.2
        score += evaluation['student_satisfaction'] * 0.15
        score += evaluation['scheduling_success_rate'] * 0.1
        
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    def _generate_recommendations(self, evaluation):
        """Generate recommendations based on evaluation"""
        recommendations = []
        
        if evaluation['conflicts'] > 0:
            recommendations.append("Resolve scheduling conflicts by adjusting instructor availability or classroom assignments")
        
        if evaluation['room_utilization'] < 60:
            recommendations.append("Consider consolidating classes or using fewer classrooms to improve utilization")
        
        if evaluation['instructor_balance'] < 30:
            recommendations.append("Redistribute teaching load more evenly among instructors")
        
        if evaluation['student_satisfaction'] < 60:
            recommendations.append("Reduce gaps between classes and avoid very early or late time slots")
        
        return recommendations
    
    def _save_enhanced_schedule_to_db(self, opt_run, schedule):
        """Save schedule to database with enhanced information"""
        for lesson_id, placement in schedule.items():
            if placement.get('forced'):
                continue
            
            # Calculate quality metrics for this specific schedule entry
            quality_score = self._calculate_entry_quality(placement, schedule)
            preference_satisfaction = self._calculate_entry_preference_satisfaction(placement)
            
            schedule_entry = Schedule(
                optimization_run_id=opt_run.id,
                lesson_id=lesson_id,
                instructor_id=placement['instructor'].id if placement['instructor'] else None,
                classroom_id=placement['classroom'].id if placement['classroom'] else None,
                day_of_week=placement['day'],
                start_hour=placement['start_hour'],
                duration=placement['duration'],
                lesson_type='Theory',  # Can be enhanced to detect type
                student_count=placement['lesson'].student_capacity,
                is_valid=not placement.get('forced', False),
                quality_score=quality_score,
                preference_satisfaction=preference_satisfaction,
                conflict_reasons=placement.get('conflicts', None)
            )
            db.session.add(schedule_entry)
    
    def _calculate_entry_quality(self, placement, schedule):
        """Calculate quality score for individual schedule entry"""
        score = 80  # Base quality
        
        # Time slot preference
        start_hour = placement['start_hour']
        if 2 <= start_hour <= 6:  # Good time slots
            score += 10
        elif start_hour < 2 or start_hour > 7:  # Poor time slots
            score -= 15
        
        # Classroom utilization
        if placement.get('classroom'):
            classroom = placement['classroom']
            lesson = placement['lesson']
            utilization = lesson.student_capacity / classroom.capacity
            if 0.7 <= utilization <= 0.9:
                score += 10
            elif utilization > 1.0:
                score -= 20
        
        return max(0, min(score, 100))
    
    def _calculate_entry_preference_satisfaction(self, placement):
        """Calculate preference satisfaction for individual entry"""
        if not placement.get('instructor'):
            return 0
        
        instructor = placement['instructor']
        lesson = placement['lesson']
        
        # Find assignment
        assignment = next((ia for ia in instructor.lesson_assignments if ia.lesson_id == lesson.id), None)
        if assignment:
            return (assignment.preference_level + assignment.competency_level) / 2 * 10
        
        return 0
    
    
    
    
    
    
    
    
    
    def _find_constraint_satisfying_placement(self, lesson, schedule, instructors, classrooms):
        """Find placement that satisfies lesson constraints"""
        # Filter suitable resources
        suitable_instructors = [i for i in instructors if i.can_teach_lesson(lesson)]
        suitable_classrooms = [c for c in classrooms if c.is_suitable_for_lesson(lesson)]
        
        if not suitable_instructors:
            suitable_instructors = instructors  # Fallback
        
        if lesson.is_online:
            suitable_classrooms = [None]
        elif not suitable_classrooms:
            suitable_classrooms = [c for c in classrooms if c.capacity >= lesson.student_capacity]
        
        # Try to find valid placement
        for instructor in suitable_instructors:
            for classroom in suitable_classrooms:
                for day in range(self.days):
                    for start_hour in range(self.time_slots - lesson.total_hours + 1):
                        if self._is_valid_placement(schedule, lesson, instructor, classroom, start_hour, day):
                            return {
                                'lesson': lesson,
                                'instructor': instructor,
                                'classroom': classroom,
                                'start_hour': start_hour,
                                'day': day,
                                'duration': lesson.total_hours
                            }
        return None
    
# models/optimization.py dosyasındaki _is_valid_placement fonksiyonunun TAMAMI

    def _is_valid_placement(self, schedule, lesson, instructor, classroom, start_hour, day):
        """Enhanced validity check for lesson placement"""
        end_hour = start_hour + lesson.total_hours
        
        # Time bounds check
        if end_hour > self.time_slots:
            return False
        
        # Instructor checks
        if instructor:
            # Instructor Availability check
            if instructor.availability:
                for hour_offset in range(lesson.total_hours):
                    hour = start_hour + hour_offset
                    if (hour < len(instructor.availability) and 
                        day < len(instructor.availability[hour]) and 
                        not instructor.availability[hour][day]):
                        return False
            
            # Instructor Daily hour limit check
            daily_hours = sum(
                p['duration'] for p in schedule.values() 
                if p.get('instructor') and p['instructor'].id == instructor.id and p['day'] == day
            )
            if daily_hours + lesson.total_hours > instructor.max_daily_hours:
                return False
        
        # Classroom checks
        if classroom:
            # #############################################################
            # ## YENİ DERSLİK KONTROLÜNÜN DOĞRU YERİ BURASI ##
            # #############################################################
            if classroom.availability:
                # Dersin süresi boyunca her saat dilimini kontrol et
                for hour_offset in range(lesson.total_hours):
                    current_hour = start_hour + hour_offset
                    if current_hour >= self.time_slots: continue

                    # Dersliğin o gün ve o saatte müsait olup olmadığını kontrol et
                    if not classroom.availability[current_hour][day]:
                        return False # Derslik bu saatte müsait değil
            # #############################################################

            # Capacity check
            if classroom.capacity < lesson.student_capacity:
                return False
            
            # Equipment requirements
            if lesson.requires_lab and not classroom.has_lab:
                return False
            if lesson.requires_computer and not classroom.has_computer:
                return False
            if lesson.requires_projector and not lesson.requires_projector: # Düzeltme: lesson.requires_projector olmalı
                return False
        elif not lesson.is_online:
            return False  # Non-online lessons need classrooms
        
        # Conflict checks with existing schedule
        for existing_placement in schedule.values():
            if existing_placement.get('forced'):
                continue
            
            existing_day = existing_placement['day']
            existing_start = existing_placement['start_hour']
            existing_end = existing_start + existing_placement['duration']
            
            if existing_day != day:
                continue
            
            if not (end_hour <= existing_start or start_hour >= existing_end):
                if (instructor and existing_placement.get('instructor') and 
                    instructor.id == existing_placement['instructor'].id):
                    return False
                
                if (classroom and existing_placement.get('classroom') and 
                    classroom.id == existing_placement['classroom'].id):
                    return False
        
        return True

    def _calculate_population_diversity(self, population):
        """Calculate diversity of current population"""
        if len(population) < 2:
            return 100
        
        total_differences = 0
        comparisons = 0
        
        # Sample pairs for efficiency
        sample_size = min(20, len(population))
        sample_indices = random.sample(range(len(population)), sample_size)
        
        for i in range(len(sample_indices)):
            for j in range(i + 1, len(sample_indices)):
                idx1, idx2 = sample_indices[i], sample_indices[j]
                diff = self._calculate_schedule_difference(population[idx1], population[idx2])
                total_differences += diff
                comparisons += 1
        
        return (total_differences / comparisons) if comparisons > 0 else 0
    
    def _calculate_schedule_difference(self, schedule1, schedule2):
        """Calculate difference between two schedules"""
        differences = 0
        all_lessons = set(schedule1.keys()) | set(schedule2.keys())
        
        for lesson_id in all_lessons:
            p1 = schedule1.get(lesson_id)
            p2 = schedule2.get(lesson_id)
            
            if not p1 or not p2:
                differences += 10  # One schedule has lesson, other doesn't
                continue
            
            # Compare placement attributes
            if p1.get('day') != p2.get('day'):
                differences += 3
            if p1.get('start_hour') != p2.get('start_hour'):
                differences += 2
            if (p1.get('instructor') and p2.get('instructor') and 
                p1['instructor'].id != p2['instructor'].id):
                differences += 2
            if (p1.get('classroom') and p2.get('classroom') and 
                p1['classroom'].id != p2['classroom'].id):
                differences += 1
        
        return min(differences, 100)  # Cap at 100%
    
    def _adjust_parameters(self, diversity, stagnation_counter):
        """Dynamically adjust algorithm parameters"""
        # Increase mutation rate if diversity is low or stagnation is high
        if diversity < self.config.diversity_threshold or stagnation_counter > 10:
            self.config.mutation_rate = min(0.3, self.config.mutation_rate * 1.1)
        else:
            self.config.mutation_rate = max(0.05, self.config.mutation_rate * 0.95)
        
        # Adjust tournament size based on diversity
        if diversity < 20:
            self.config.tournament_size = max(3, self.config.tournament_size - 1)
        else:
            self.config.tournament_size = min(7, self.config.tournament_size + 1)
    
    def _create_new_population(self, population, fitness_scores, lessons, instructors, classrooms):
        """Create new population using selection, crossover, and mutation"""
        new_population = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        
        # Elitism - keep best individuals
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        for i in range(elite_count):
            new_population.append(deepcopy(population[sorted_indices[i]]))
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._advanced_crossover(parent1, parent2, lessons)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._advanced_mutation(child1, lessons, instructors, classrooms)
            if random.random() < self.config.mutation_rate:
                child2 = self._advanced_mutation(child2, lessons, instructors, classrooms)
            
            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)
        
        return new_population[:self.config.population_size]
    
    def _tournament_selection(self, population, fitness_scores):
        """Tournament selection with fitness-based selection"""
        tournament_indices = random.sample(range(len(population)), 
                                          min(self.config.tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return deepcopy(population[winner_idx])
    
    def _advanced_crossover(self, parent1, parent2, lessons):
        """Advanced crossover combining multiple strategies"""
        child1, child2 = {}, {}
        
        # Get all lesson IDs
        all_lesson_ids = list(set(parent1.keys()) | set(parent2.keys()))
        
        # Multi-point crossover
        crossover_points = sorted(random.sample(range(len(all_lesson_ids)), 
                                               min(3, len(all_lesson_ids)//2)))
        crossover_points.append(len(all_lesson_ids))
        
        current_parent = 1
        start_idx = 0
        
        for point in crossover_points:
            for i in range(start_idx, point):
                if i < len(all_lesson_ids):
                    lesson_id = all_lesson_ids[i]
                    
                    if current_parent == 1:
                        if lesson_id in parent1:
                            child1[lesson_id] = deepcopy(parent1[lesson_id])
                        if lesson_id in parent2:
                            child2[lesson_id] = deepcopy(parent2[lesson_id])
                    else:
                        if lesson_id in parent2:
                            child1[lesson_id] = deepcopy(parent2[lesson_id])
                        if lesson_id in parent1:
                            child2[lesson_id] = deepcopy(parent1[lesson_id])
            
            current_parent = 3 - current_parent  # Switch between 1 and 2
            start_idx = point
        
        # Repair conflicts
        child1 = self._repair_schedule(child1, lessons)
        child2 = self._repair_schedule(child2, lessons)
        
        return child1, child2
    
    def _advanced_mutation(self, schedule, lessons, instructors, classrooms):
        """Advanced mutation with multiple strategies"""
        if not schedule:
            return schedule
        
        mutated = deepcopy(schedule)
        mutation_type = random.choice(['swap', 'relocate', 'instructor_change', 'time_shift'])
        
        if mutation_type == 'swap' and len(mutated) >= 2:
            # Swap two lessons' time slots
            lesson_ids = list(mutated.keys())
            id1, id2 = random.sample(lesson_ids, 2)
            
            if id1 in mutated and id2 in mutated:
                # Swap time and day
                mutated[id1]['day'], mutated[id2]['day'] = mutated[id2]['day'], mutated[id1]['day']
                mutated[id1]['start_hour'], mutated[id2]['start_hour'] = mutated[id2]['start_hour'], mutated[id1]['start_hour']
        
        elif mutation_type == 'relocate':
            # Move a lesson to a new time slot
            lesson_ids = list(mutated.keys())
            if lesson_ids:
                lesson_id = random.choice(lesson_ids)
                placement = mutated[lesson_id]
                lesson = placement['lesson']
                
                # Try new time slot
                for _ in range(20):  # Limited attempts
                    new_day = random.randint(0, self.days - 1)
                    max_start = max(0, self.time_slots - lesson.total_hours)
                    new_start_hour = random.randint(0, max_start)
                    
                    # Temporarily remove this lesson to check conflicts
                    temp_schedule = {k: v for k, v in mutated.items() if k != lesson_id}
                    
                    if self._is_valid_placement(temp_schedule, lesson, placement['instructor'], 
                                              placement['classroom'], new_start_hour, new_day):
                        mutated[lesson_id]['day'] = new_day
                        mutated[lesson_id]['start_hour'] = new_start_hour
                        break
        
        elif mutation_type == 'instructor_change':
            # Change instructor for a lesson
            lesson_ids = list(mutated.keys())
            if lesson_ids:
                lesson_id = random.choice(lesson_ids)
                placement = mutated[lesson_id]
                lesson = placement['lesson']
                
                # Find alternative instructors
                suitable_instructors = [i for i in instructors if i.can_teach_lesson(lesson)]
                if not suitable_instructors:
                    suitable_instructors = [i for i in instructors if i.department_id == lesson.department_id]
                
                if suitable_instructors:
                    new_instructor = random.choice(suitable_instructors)
                    
                    # Check if new instructor is available
                    temp_schedule = {k: v for k, v in mutated.items() if k != lesson_id}
                    if self._is_valid_placement(temp_schedule, lesson, new_instructor,
                                              placement['classroom'], placement['start_hour'], placement['day']):
                        mutated[lesson_id]['instructor'] = new_instructor
        
        elif mutation_type == 'time_shift':
            # Shift a lesson by ±1 hour
            lesson_ids = list(mutated.keys())
            if lesson_ids:
                lesson_id = random.choice(lesson_ids)
                placement = mutated[lesson_id]
                lesson = placement['lesson']
                
                shift = random.choice([-1, 1])
                new_start_hour = placement['start_hour'] + shift
                
                if (0 <= new_start_hour <= self.time_slots - lesson.total_hours):
                    temp_schedule = {k: v for k, v in mutated.items() if k != lesson_id}
                    if self._is_valid_placement(temp_schedule, lesson, placement['instructor'],
                                              placement['classroom'], new_start_hour, placement['day']):
                        mutated[lesson_id]['start_hour'] = new_start_hour
        
        return mutated
    
    def _repair_schedule(self, schedule, lessons):
        """Repair schedule by resolving conflicts"""
        if not schedule:
            return schedule
        
        repaired = deepcopy(schedule)
        conflicts_found = True
        repair_attempts = 0
        
        while conflicts_found and repair_attempts < 50:
            conflicts_found = False
            repair_attempts += 1
            
            # Find conflicts
            time_usage = {}
            for lesson_id, placement in repaired.items():
                if placement.get('forced'):
                    continue
                
                day = placement['day']
                start_hour = placement['start_hour']
                duration = placement['duration']
                instructor = placement.get('instructor')
                classroom = placement.get('classroom')
                
                for hour_offset in range(duration):
                    current_hour = start_hour + hour_offset
                    time_key = (day, current_hour)
                    
                    if time_key not in time_usage:
                        time_usage[time_key] = {'instructors': [], 'classrooms': [], 'lessons': []}
                    
                    # Check for conflicts
                    if instructor:
                        if instructor.id in [i.id for i in time_usage[time_key]['instructors']]:
                            conflicts_found = True
                            # Try to move this lesson
                            new_placement = self._find_alternative_time(placement, repaired, lessons)
                            if new_placement:
                                repaired[lesson_id] = new_placement
                            break
                        else:
                            time_usage[time_key]['instructors'].append(instructor)
                    
                    if classroom:
                        if classroom.id in [c.id for c in time_usage[time_key]['classrooms']]:
                            conflicts_found = True
                            # Try to move this lesson
                            new_placement = self._find_alternative_time(placement, repaired, lessons)
                            if new_placement:
                                repaired[lesson_id] = new_placement
                            break
                        else:
                            time_usage[time_key]['classrooms'].append(classroom)
                    
                    time_usage[time_key]['lessons'].append(lesson_id)
        
        return repaired
    
    def _find_alternative_time(self, placement, schedule, lessons):
        """Find alternative time slot for a conflicting lesson"""
        lesson = placement['lesson']
        instructor = placement['instructor']
        classroom = placement['classroom']
        
        # Remove this lesson temporarily
        temp_schedule = {k: v for k, v in schedule.items() 
                        if v != placement}
        
        # Try alternative time slots
        for day in range(self.days):
            for start_hour in range(self.time_slots - lesson.total_hours + 1):
                if self._is_valid_placement(temp_schedule, lesson, instructor, classroom, start_hour, day):
                    new_placement = deepcopy(placement)
                    new_placement['day'] = day
                    new_placement['start_hour'] = start_hour
                    return new_placement
        
        return None  # No alternative found
    
    def _apply_local_search(self, population, lessons, instructors, classrooms):
        """Apply local search to improve promising individuals"""
        improved_population = []
        
        for individual in population:
            # Apply local search with some probability
            if random.random() < 0.3:  # 30% chance
                improved = self._local_search_improvement(individual, lessons, instructors, classrooms)
                improved_population.append(improved)
            else:
                improved_population.append(individual)
        
        return improved_population
    
    #GÜNCELLENDİ
    def _local_search_improvement(self, schedule, lessons, instructors, classrooms):
        """Apply local search to improve a single schedule using a simulated annealing approach."""
        improved = deepcopy(schedule)
        current_fitness = self._enhanced_fitness_function(improved, lessons, instructors, classrooms)
        
        # Try a few random improvements
        for _ in range(10): # Try 10 local moves
            
            # Select a random lesson to modify
            if not improved:
                continue
            lesson_id_to_move = random.choice(list(improved.keys()))
            placement = improved[lesson_id_to_move]
            lesson = placement['lesson']

            # Create a temporary schedule without this lesson
            temp_schedule = {k: v for k, v in improved.items() if k != lesson_id_to_move}

            # Find a new valid random placement
            for attempt in range(20): # Try to find a new spot
                new_day = random.randint(0, self.days - 1)
                max_start = max(0, self.time_slots - lesson.total_hours)
                new_start_hour = random.randint(0, max_start)
                
                if self._is_valid_placement(temp_schedule, lesson, placement['instructor'],
                                          placement['classroom'], new_start_hour, placement['day']):
                    
                    test_schedule = deepcopy(improved)
                    test_schedule[lesson_id_to_move]['day'] = new_day
                    test_schedule[lesson_id_to_move]['start_hour'] = new_start_hour
                    
                    test_fitness = self._enhanced_fitness_function(test_schedule, lessons, instructors, classrooms)
                    
                    # Decision logic (Simulated Annealing)
                    if test_fitness > current_fitness:
                        improved = test_schedule
                        current_fitness = test_fitness
                    else:
                        # Accept a worse solution with a certain probability
                        delta = current_fitness - test_fitness
                        acceptance_probability = np.exp(-delta / self.current_temperature)
                        if random.random() < acceptance_probability:
                            improved = test_schedule
                            current_fitness = test_fitness
                    break # Move to the next local search iteration
        
        return improved
    
    def optimize_multi_department_schedule(self, department_ids: List[int], semester: int, 
                                        session_id: str, progress_callback=None,
                                        shared_lessons=None, use_building_preference=False) -> OptimizationRun:
        """Multi-department optimization with shared lessons support"""
        
        start_time = time.time()
        shared_lessons = shared_lessons or []
        
        logger.info(f"Starting multi-department optimization for {len(department_ids)} departments")
        
        try:
            # Load data from all departments
            all_lessons = []
            all_instructors = []
            department_info = {}
            
            for dept_id in department_ids:
                dept = Department.query.get(dept_id)
                dept_lessons = self._get_lessons(dept_id, semester)
                dept_instructors = self._get_instructors(dept_id)
                
                all_lessons.extend(dept_lessons)
                all_instructors.extend(dept_instructors)
                
                department_info[dept_id] = {
                    'department': dept,
                    'lessons': dept_lessons,
                    'instructors': dept_instructors
                }
            
            # Get classrooms (with building preference if enabled)
            classrooms = self._get_multi_department_classrooms(department_ids, use_building_preference)
            
            if not all_lessons:
                raise ValueError("No lessons found for the specified departments and semester")
            
            # Process shared lessons
            lesson_mapping = self._process_shared_lessons(shared_lessons, all_lessons)
            
            logger.info(f"Multi-dept optimization: {len(all_lessons)} lessons, {len(all_instructors)} instructors, {len(classrooms)} classrooms")
            logger.info(f"Shared lessons: {len(shared_lessons)}")
            
            # Create optimization run record (YENİ DATABASE MODEL FIELDS kullanılıyor)
            opt_run = OptimizationRun(
                session_id=session_id,
                faculty_id=department_info[department_ids[0]]['department'].faculty_id if len(set([department_info[d]['department'].faculty_id for d in department_ids])) == 1 else None,
                department_ids=department_ids,  # YENİ FIELD
                semester=semester,
                academic_year=self._get_current_academic_year(),
                use_building_preference=use_building_preference,  # YENİ FIELD
                parameters={
                    'department_count': len(department_ids),
                    'lesson_count': len(all_lessons),
                    'instructor_count': len(all_instructors),
                    'classroom_count': len(classrooms),
                    'shared_lessons': len(shared_lessons),
                    'config': self.config.__dict__
                },
                status='running',
                created_by='System'
            )
            db.session.add(opt_run)
            db.session.commit()
            
            # Run enhanced genetic algorithm with multi-department support
            best_schedule, best_fitness, generation_data = self._enhanced_genetic_algorithm_multi_dept(
                all_lessons, all_instructors, classrooms, department_info, 
                lesson_mapping, use_building_preference, opt_run, progress_callback
            )
            
            # Comprehensive evaluation
            evaluation = self._comprehensive_evaluation_multi_dept(
                best_schedule, all_lessons, all_instructors, classrooms, department_info
            )
            
            # Save results
            runtime = time.time() - start_time
            
            opt_run.status = 'completed'
            opt_run.fitness_score = best_fitness
            opt_run.conflicts_count = evaluation['conflicts']
            opt_run.room_utilization = evaluation['room_utilization']
            opt_run.instructor_balance = evaluation['instructor_balance']
            opt_run.student_satisfaction = evaluation['student_satisfaction']
            opt_run.runtime_seconds = runtime
            opt_run.completed_at = datetime.utcnow()
            opt_run.results = {
                'evaluation': evaluation,
                'generation_stats': generation_data[-10:],
                'optimization_summary': self._create_optimization_summary(evaluation),
                'shared_lessons_summary': self._create_shared_lessons_summary(shared_lessons, best_schedule),
                'department_breakdown': self._create_department_breakdown(best_schedule, department_info)
            }
            
            # Save multi-department schedule to database
            self._save_multi_department_schedule_to_db(opt_run, best_schedule, lesson_mapping)
            
            db.session.commit()
            logger.info(f"Multi-department optimization completed in {runtime:.2f}s with fitness {best_fitness:.2f}")
            return opt_run
            
        except Exception as e:
            logger.error(f"Multi-department optimization failed: {str(e)}")
            opt_run.status = 'error'
            opt_run.results = {'error': str(e)}
            db.session.commit()
            raise e

    # 4. YENİ HELPER FUNCTIONS EKLE

    def _get_multi_department_classrooms(self, department_ids: List[int], use_building_preference=False) -> List[Classroom]:
        """Get classrooms with building preference support"""
        all_classrooms = Classroom.query.filter_by(is_active=True, is_bookable=True).all()
        
        if not use_building_preference:
            return all_classrooms
        
        # Get faculty buildings from departments
        faculty_buildings = set()
        for dept_id in department_ids:
            dept = Department.query.get(dept_id)
            if dept and dept.faculty_ref and dept.faculty_ref.building:
                faculty_buildings.add(dept.faculty_ref.building)
            if dept and dept.building:
                faculty_buildings.add(dept.building)
        
        if not faculty_buildings:
            return all_classrooms
        
        # Prioritize classrooms in faculty buildings
        priority_classrooms = []
        other_classrooms = []
        
        for classroom in all_classrooms:
            if classroom.building and classroom.building in faculty_buildings:
                priority_classrooms.append(classroom)
            else:
                other_classrooms.append(classroom)
        
        return priority_classrooms + other_classrooms

    def _process_shared_lessons(self, shared_lessons: List[Dict], all_lessons: List[Lesson]) -> Dict:
        """Process shared lessons and create mapping"""
        lesson_mapping = {}
        
        for shared_info in shared_lessons:
            main_lesson = shared_info['main_lesson']
            all_shared_lessons = shared_info['all_lessons']
            
            # Map all lessons to the main one
            for lesson in all_shared_lessons:
                lesson_mapping[lesson.id] = {
                    'main_lesson_id': main_lesson.id,
                    'is_shared': True,
                    'departments': shared_info['departments'],
                    'total_capacity': shared_info['total_capacity']
                }
        
        # Map non-shared lessons to themselves
        shared_lesson_ids = set(lesson_mapping.keys())
        for lesson in all_lessons:
            if lesson.id not in shared_lesson_ids:
                lesson_mapping[lesson.id] = {
                    'main_lesson_id': lesson.id,
                    'is_shared': False,
                    'departments': [lesson.department_id],
                    'total_capacity': lesson.student_capacity
                }
        
        return lesson_mapping

    # 5. MULTI-DEPARTMENT GENETIC ALGORITHM EKLE

    def _enhanced_genetic_algorithm_multi_dept(self, lessons, instructors, classrooms, 
                                            department_info, lesson_mapping, 
                                            use_building_preference, opt_run, progress_callback):
        """Enhanced genetic algorithm for multi-department optimization"""
        
        # Initialize population with multi-department awareness
        population = self._create_diverse_population_multi_dept(
            lessons, instructors, classrooms, department_info, lesson_mapping
        )
        
        best_schedule = None
        best_fitness = float('-inf')
        stagnation_counter = 0
        generation_data = []
        
        for generation in range(self.config.generations):
            # Evaluate population with multi-department fitness
            fitness_scores = []
            for individual in population:
                fitness = self._enhanced_fitness_function_multi_dept(
                    individual, lessons, instructors, classrooms, 
                    department_info, lesson_mapping, use_building_preference
                )
                fitness_scores.append(fitness)
            
            # Update best solution
            gen_best_idx = np.argmax(fitness_scores)
            gen_best_fitness = fitness_scores[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_schedule = deepcopy(population[gen_best_idx])
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Calculate diversity
            diversity = self._calculate_population_diversity(population)
            
            # Store generation statistics
            gen_stats = {
                'generation': generation,
                'best_fitness': gen_best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'diversity': diversity,
                'conflicts': self._count_multi_dept_conflicts(population[gen_best_idx], lesson_mapping),
                'stagnation': stagnation_counter
            }
            generation_data.append(gen_stats)
            
            # Update database progress
            opt_run.progress = {
                'generation': generation,
                'total_generations': self.config.generations,
                'best_fitness': gen_best_fitness,
                'conflicts': gen_stats['conflicts'],
                'diversity': diversity
            }
            db.session.commit()
            
            if progress_callback:
                progress_callback(gen_stats)
            
            # Adaptive parameter adjustment
            if self.config.adaptive_mutation:
                self._adjust_parameters(diversity, stagnation_counter)
            
            # Early termination conditions
            if gen_stats['conflicts'] == 0 and gen_best_fitness > 950:
                logger.info(f"Early termination at generation {generation} - optimal solution found")
                break
            
            if stagnation_counter >= self.config.stagnation_limit:
                logger.info(f"Early termination at generation {generation} due to stagnation")
                break
            
            # Create next generation
            population = self._create_next_generation_multi_dept(
                population, fitness_scores, lessons, instructors, classrooms, 
                department_info, lesson_mapping
            )
            
            # Apply local search to promising individuals
            if random.random() < self.config.local_search_probability:
                population = self._apply_local_search_multi_dept(
                    population, lessons, instructors, classrooms, lesson_mapping
                )

            # Cool down the temperature for simulated annealing
            self.current_temperature *= self.config.cooling_rate
        
        return best_schedule, best_fitness, generation_data

    # 6. MULTI-DEPARTMENT POPULATION CREATION EKLE

    def _create_diverse_population_multi_dept(self, lessons, instructors, classrooms, 
                                            department_info, lesson_mapping):
        """Create diverse initial population for multi-department optimization"""
        population = []
        
        for i in range(self.config.population_size):
            if i < self.config.population_size * 0.2:
                # Department-aware scheduling strategy
                schedule = self._create_department_aware_schedule(
                    lessons, instructors, classrooms, department_info, lesson_mapping
                )
            elif i < self.config.population_size * 0.4:
                # Shared lessons priority strategy
                schedule = self._create_shared_lessons_priority_schedule(
                    lessons, instructors, classrooms, lesson_mapping
                )
            elif i < self.config.population_size * 0.6:
                # Graph coloring with multi-department awareness
                schedule = self._create_graph_coloring_schedule_multi_dept(
                    lessons, instructors, classrooms, lesson_mapping
                )
            elif i < self.config.population_size * 0.8:
                # Priority-based with multi-department
                schedule = self._create_priority_based_schedule_multi_dept(
                    lessons, instructors, classrooms, lesson_mapping
                )
            else:
                # Random with multi-department awareness
                schedule = self._create_random_schedule_multi_dept(
                    lessons, instructors, classrooms, lesson_mapping
                )
            
            population.append(schedule)
        
        return population

    def _create_department_aware_schedule(self, lessons, instructors, classrooms, 
                                        department_info, lesson_mapping):
        """Create schedule considering department boundaries"""
        schedule = {}
        
        # Schedule each department separately
        for dept_id, info in department_info.items():
            dept_lessons = info['lessons']
            dept_instructors = info['instructors']
            
            for lesson in dept_lessons:
                # Skip if already scheduled as shared lesson
                mapping = lesson_mapping[lesson.id]
                if mapping['is_shared'] and mapping['main_lesson_id'] != lesson.id:
                    continue
                
                placement = self._find_best_placement_multi_dept(
                    lesson, schedule, dept_instructors, classrooms, lesson_mapping
                )
                if placement:
                    schedule[lesson.id] = placement
        
        return schedule

    def _create_shared_lessons_priority_schedule(self, lessons, instructors, classrooms, lesson_mapping):
        """Create schedule prioritizing shared lessons first"""
        schedule = {}
        
        # Separate shared and individual lessons
        shared_lessons = []
        individual_lessons = []
        
        for lesson in lessons:
            mapping = lesson_mapping[lesson.id]
            if mapping['is_shared'] and mapping['main_lesson_id'] == lesson.id:
                shared_lessons.append(lesson)
            elif not mapping['is_shared']:
                individual_lessons.append(lesson)
        
        # Schedule shared lessons first (higher priority)
        for lesson in shared_lessons:
            placement = self._find_best_placement_multi_dept(
                lesson, schedule, instructors, classrooms, lesson_mapping
            )
            if placement:
                schedule[lesson.id] = placement
        
        # Then schedule individual lessons
        for lesson in individual_lessons:
            placement = self._find_best_placement_multi_dept(
                lesson, schedule, instructors, classrooms, lesson_mapping
            )
            if placement:
                schedule[lesson.id] = placement
        
        return schedule

    # 7. MULTI-DEPARTMENT FITNESS FUNCTION EKLE

    def _enhanced_fitness_function_multi_dept(self, schedule, lessons, instructors, 
                                            classrooms, department_info, lesson_mapping, 
                                            use_building_preference):
        """Enhanced fitness function for multi-department optimization"""
        if not schedule:
            return 0
        
        # Base fitness calculation (mevcut _enhanced_fitness_function'ı kullan)
        fitness = self._enhanced_fitness_function(schedule, lessons, instructors, classrooms)
        
        # Multi-department specific bonuses/penalties
        
        # Department balance bonus
        dept_balance = self._calculate_department_balance(schedule, department_info)
        fitness += dept_balance * 5.0
        
        # Shared lessons efficiency bonus
        shared_efficiency = self._calculate_shared_lessons_efficiency(schedule, lesson_mapping)
        fitness += shared_efficiency * 8.0
        
        # Building preference bonus (if enabled)
        if use_building_preference:
            building_bonus = self._calculate_building_preference_bonus(schedule, department_info)
            fitness += building_bonus * 6.0
        
        # Cross-department instructor utilization penalty
        cross_dept_penalty = self._calculate_cross_department_penalty(schedule, department_info)
        fitness -= cross_dept_penalty * 3.0
        
        return max(0, fitness)

    # 8. MULTI-DEPARTMENT CONFLICT DETECTION EKLE

    def _count_multi_dept_conflicts(self, schedule, lesson_mapping):
        """Count conflicts with multi-department awareness"""
        conflicts = 0
        time_slots_used = {}
        
        for lesson_id, placement in schedule.items():
            if placement.get('forced'):
                conflicts += 5
                continue
            
            lesson = placement['lesson']
            instructor = placement['instructor']
            classroom = placement['classroom']
            day = placement['day']
            start_hour = placement['start_hour']
            duration = placement['duration']
            
            # Check for time slot conflicts
            for hour_offset in range(duration):
                current_hour = start_hour + hour_offset
                time_key = (day, current_hour)
                
                if time_key not in time_slots_used:
                    time_slots_used[time_key] = {
                        'instructors': set(),
                        'classrooms': set(),
                        'lessons': []
                    }
                
                # Instructor conflicts
                if instructor.id in time_slots_used[time_key]['instructors']:
                    conflicts += 10  # High penalty for instructor conflicts
                time_slots_used[time_key]['instructors'].add(instructor.id)
                
                # Classroom conflicts (but allow shared lessons in same classroom)
                mapping = lesson_mapping.get(lesson_id, {})
                if mapping.get('is_shared'):
                    # Shared lessons can use the same classroom
                    shared_in_same_slot = any(
                        lesson_mapping.get(other_lesson.id, {}).get('main_lesson_id') == mapping.get('main_lesson_id')
                        for other_lesson in time_slots_used[time_key]['lessons']
                    )
                    if not shared_in_same_slot and classroom.id in time_slots_used[time_key]['classrooms']:
                        conflicts += 8
                else:
                    if classroom.id in time_slots_used[time_key]['classrooms']:
                        conflicts += 8
                
                time_slots_used[time_key]['classrooms'].add(classroom.id)
                time_slots_used[time_key]['lessons'].append(lesson)
                
                # Capacity conflicts (with shared lesson consideration)
                total_capacity_needed = mapping.get('total_capacity', lesson.student_capacity)
                if total_capacity_needed > classroom.capacity:
                    conflicts += 15
        
        return conflicts

    # 9. DEPARTMENT BALANCE CALCULATION EKLE

    def _calculate_department_balance(self, schedule, department_info):
        """Calculate balance between departments"""
        if not schedule:
            return 0
        
        dept_hours = {}
        total_hours = 0
        
        for lesson_id, placement in schedule.items():
            lesson = placement['lesson']
            dept_id = lesson.department_id
            hours = placement['duration']
            
            if dept_id not in dept_hours:
                dept_hours[dept_id] = 0
            dept_hours[dept_id] += hours
            total_hours += hours
        
        if total_hours == 0:
            return 0
        
        # Calculate balance score (lower variance = better balance)
        dept_count = len(department_info)
        expected_hours_per_dept = total_hours / dept_count
        
        variance = 0
        for dept_id in department_info.keys():
            actual_hours = dept_hours.get(dept_id, 0)
            variance += (actual_hours - expected_hours_per_dept) ** 2
        
        balance_score = max(0, 100 - (variance / dept_count))
        return balance_score

    # 10. SHARED LESSONS EFFICIENCY EKLE

    def _calculate_shared_lessons_efficiency(self, schedule, lesson_mapping):
        """Calculate efficiency of shared lesson scheduling"""
        shared_lesson_count = 0
        efficiently_scheduled = 0
        
        for lesson_id, placement in schedule.items():
            mapping = lesson_mapping.get(lesson_id, {})
            if mapping.get('is_shared') and mapping.get('main_lesson_id') == lesson_id:
                shared_lesson_count += 1
                
                # Check if classroom capacity is efficiently used
                classroom = placement['classroom']
                if classroom.capacity >= mapping.get('total_capacity', 0) * 0.8:
                    efficiently_scheduled += 1
        
        if shared_lesson_count == 0:
            return 50  # Neutral score when no shared lessons
        
        efficiency_ratio = efficiently_scheduled / shared_lesson_count
        return efficiency_ratio * 100

    # 11. DIĞER YARDIMCI FONKSIYONLAR EKLE

    def _calculate_building_preference_bonus(self, schedule, department_info):
        """Calculate bonus for using preferred buildings"""
        # Implementation here
        return 0  # Placeholder

    def _calculate_cross_department_penalty(self, schedule, department_info):
        """Penalty for instructors teaching across multiple departments"""
        # Implementation here
        return 0  # Placeholder

    def _find_best_placement_multi_dept(self, lesson, schedule, instructors, classrooms, lesson_mapping):
        """Find best placement for a lesson in multi-department context"""
        return self._find_best_placement(lesson, schedule, instructors, classrooms)

    def _create_next_generation_multi_dept(self, population, fitness_scores, lessons, 
                                        instructors, classrooms, department_info, lesson_mapping):
        """Create next generation for multi-department optimization"""
        return self._create_new_population(population, fitness_scores, lessons, instructors, classrooms)

    def _apply_local_search_multi_dept(self, population, lessons, instructors, classrooms, lesson_mapping):
        """Apply local search to improve promising individuals"""
        return self._apply_local_search(population, lessons, instructors, classrooms)

    def _comprehensive_evaluation_multi_dept(self, schedule, lessons, instructors, classrooms, department_info):
        """Comprehensive evaluation for multi-department schedule"""
        evaluation = self._comprehensive_evaluation(schedule, lessons, instructors, classrooms)
        evaluation['department_balance'] = self._calculate_department_balance(schedule, department_info)
        return evaluation

    def _save_multi_department_schedule_to_db(self, opt_run, schedule, lesson_mapping):
        """Save multi-department schedule to database"""
        for lesson_id, placement in schedule.items():
            lesson = placement['lesson']
            instructor = placement['instructor']
            classroom = placement['classroom']
            mapping = lesson_mapping.get(lesson_id, {})
            
            schedule_entry = Schedule(
                optimization_run_id=opt_run.id,
                lesson_id=lesson.id,
                instructor_id=instructor.id if instructor else None,
                classroom_id=classroom.id if classroom else None,
                day_of_week=placement['day'],
                start_hour=placement['start_hour'],
                duration=placement['duration'],
                is_valid=not placement.get('forced', False),
                is_shared_lesson=mapping.get('is_shared', False),
                shared_lesson_departments=mapping.get('departments', []) if mapping.get('is_shared') else None
            )
            db.session.add(schedule_entry)

    def _create_shared_lessons_summary(self, shared_lessons, schedule):
        """Create summary of shared lessons scheduling"""
        return {'shared_count': len(shared_lessons)}

    def _create_department_breakdown(self, schedule, department_info):
        """Create breakdown by department"""
        return {dept_id: {'lessons_scheduled': 0} for dept_id in department_info.keys()}

    # 12. PLACEHOLDER FUNCTIONS (Önceki versiyonlarla uyumluluk için)
    def _create_graph_coloring_schedule_multi_dept(self, lessons, instructors, classrooms, lesson_mapping):
        """Graph coloring with multi-department awareness"""
        return self._create_graph_coloring_schedule(lessons, instructors, classrooms)

    def _create_priority_based_schedule_multi_dept(self, lessons, instructors, classrooms, lesson_mapping):
        """Priority-based scheduling with multi-department awareness"""
        return self._create_priority_based_schedule(lessons, instructors, classrooms)

    def _create_random_schedule_multi_dept(self, lessons, instructors, classrooms, lesson_mapping):
        """Random scheduling with multi-department awareness"""
        return self._create_random_schedule(lessons, instructors, classrooms)