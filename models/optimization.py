# ===== models/optimization.py - Geliştirilmiş =====
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import random
import numpy as np
from copy import deepcopy
import time
import logging
# Yeni Doğru Satır
from models.database import db, Lesson, Instructor, Classroom, OptimizationRun, Schedule, ClassroomAvailability
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
    
    # Fitness weights - normalized to sum to 100
    conflict_penalty: float = 35.0          # Hard constraints
    room_utilization_weight: float = 20.0   # Efficient room usage
    instructor_balance_weight: float = 15.0  # Workload distribution
    preference_weight: float = 12.0         # Instructor preferences
    time_distribution_weight: float = 10.0  # Daily/weekly distribution
    student_satisfaction_weight: float = 8.0 # Student schedule quality

class AdvancedScheduleOptimizer:
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.time_slots = 10  # 8:30-18:30 (30-min slots)
        self.days = 5  # Monday-Friday
        self.generation_stats = []
        self.best_fitness_history = []
        self.diversity_history = []
        
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
        
        self.generation_stats = generation_data
        return best_schedule, best_fitness, generation_data
    
    def _create_diverse_population(self, lessons, instructors, classrooms):
        """Create initial population with diverse construction strategies"""
        population = []
        
        for i in range(self.config.population_size):
            if i < self.config.population_size * 0.3:  # 30% - Priority-based construction
                schedule = self._create_priority_based_schedule(lessons, instructors, classrooms)
            elif i < self.config.population_size * 0.6:  # 30% - Greedy construction
                schedule = self._create_greedy_schedule(lessons, instructors, classrooms)
            elif i < self.config.population_size * 0.8:  # 20% - Constraint-focused construction
                schedule = self._create_constraint_focused_schedule(lessons, instructors, classrooms)
            else:  # 20% - Random construction
                schedule = self._create_random_schedule(lessons, instructors, classrooms)
            
            population.append(schedule)
        
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
    
    def _calculate_student_satisfaction(self, schedule, lessons):
        """Estimate student satisfaction based on schedule quality"""
        satisfaction = 50  # Base satisfaction
        
        # Analyze gaps and clustering
        daily_schedules = {}
        for placement in schedule.values():
            if not placement.get('forced'):
                day = placement['day']
                if day not in daily_schedules:
                    daily_schedules[day] = []
                daily_schedules[day].append((placement['start_hour'], placement['start_hour'] + placement['duration']))
        
        for day, day_schedule in daily_schedules.items():
            if len(day_schedule) > 1:
                day_schedule.sort()
                
                # Calculate gaps between classes
                for i in range(len(day_schedule) - 1):
                    gap = day_schedule[i+1][0] - day_schedule[i][1]
                    if gap == 1:  # 30-min break (good)
                        satisfaction += 2
                    elif gap == 2:  # 1-hour break (acceptable)
                        satisfaction += 1
                    elif gap > 4:  # Long gap (bad)
                        satisfaction -= 3
                
                # Prefer morning starts
                first_class_start = day_schedule[0][0]
                if 2 <= first_class_start <= 4:  # 9:30-11:30 start
                    satisfaction += 3
                elif first_class_start < 2:  # Too early
                    satisfaction -= 2
        
        return max(0, min(satisfaction, 100))
    
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
    
    def _is_valid_placement(self, schedule, lesson, instructor, classroom, start_hour, day):
        """Enhanced validity check for lesson placement"""
        end_hour = start_hour + lesson.total_hours
        
        # Time bounds check
        if end_hour > self.time_slots:
            return False
        
        # Instructor checks
        if instructor:
            # Availability check
            if instructor.availability:
                for hour in range(start_hour, end_hour):
                    if (hour < len(instructor.availability) and 
                        day < len(instructor.availability[hour]) and 
                        not instructor.availability[hour][day]):
                        return False
            
            # Daily hour limit check
            daily_hours = sum(
                p['duration'] for p in schedule.values() 
                if p.get('instructor') and p['instructor'].id == instructor.id and p['day'] == day
            )
            if daily_hours + lesson.total_hours > instructor.max_daily_hours:
                return False
        
        # Classroom checks
        if classroom:
            # Capacity check
            if classroom.capacity < lesson.student_capacity:
                return False
            
            # Equipment requirements
            if lesson.requires_lab and not classroom.has_lab:
                return False
            if lesson.requires_computer and not classroom.has_computer:
                return False
            if lesson.requires_projector and not classroom.has_projector:
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
            
            # Skip if different days
            if existing_day != day:
                continue
            
            # Check time overlap
            if not (end_hour <= existing_start or start_hour >= existing_end):
                # Time conflict exists
                
                # Instructor conflict
                if (instructor and existing_placement.get('instructor') and 
                    instructor.id == existing_placement['instructor'].id):
                    return False
                
                # Classroom conflict
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
    
    def _local_search_improvement(self, schedule, lessons, instructors, classrooms):
        """Apply local search to improve a single schedule"""
        improved = deepcopy(schedule)
        current_fitness = self._enhanced_fitness_function(improved, lessons, instructors, classrooms)
        
        improvement_found = True
        iterations = 0
        
        while improvement_found and iterations < 10:
            improvement_found = False
            iterations += 1
            
            # Try small improvements
            for lesson_id, placement in list(improved.items()):
                lesson = placement['lesson']
                
                # Try adjacent time slots
                for time_delta in [-1, 1]:
                    new_start_hour = placement['start_hour'] + time_delta
                    if 0 <= new_start_hour <= self.time_slots - lesson.total_hours:
                        
                        temp_schedule = {k: v for k, v in improved.items() if k != lesson_id}
                        if self._is_valid_placement(temp_schedule, lesson, placement['instructor'],
                                                  placement['classroom'], new_start_hour, placement['day']):
                            
                            test_schedule = deepcopy(improved)
                            test_schedule[lesson_id]['start_hour'] = new_start_hour
                            
                            test_fitness = self._enhanced_fitness_function(test_schedule, lessons, instructors, classrooms)
                            if test_fitness > current_fitness:
                                improved = test_schedule
                                current_fitness = test_fitness
                                improvement_found = True
                                break
                
                if improvement_found:
                    break
        
        return improved