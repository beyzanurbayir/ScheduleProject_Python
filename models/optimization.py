# ===== models/optimization.py =====
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import random
import numpy as np
from copy import deepcopy
import time
from .database import db, Lesson, Instructor, Classroom, OptimizationRun, Schedule
from datetime import datetime

@dataclass
class OptimizationConfig:
    population_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 5
    elitism_rate: float = 0.2
    
    # Fitness weights
    conflict_penalty: float = 100.0
    instructor_balance_weight: float = 10.0
    daily_distribution_weight: float = 5.0
    classroom_utilization_weight: float = 15.0
    preference_weight: float = 8.0

class ScheduleOptimizer:
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.time_slots = 10  # 8:30-18:30
        self.days = 5  # Monday-Friday
        
    def optimize_schedule(self, department_id: int, semester: int, session_id: str, 
                         progress_callback=None) -> OptimizationRun:
        """Main optimization function using genetic algorithm"""
        
        start_time = time.time()
        
        # Create optimization run record
        opt_run = OptimizationRun(
            session_id=session_id,
            department_id=department_id,
            semester=semester,
            parameters=self.config.__dict__.copy(),
            status='running'
        )
        db.session.add(opt_run)
        db.session.commit()
        
        try:
            # Get data from database
            lessons = self._get_lessons(department_id, semester)
            instructors = self._get_instructors(department_id)
            classrooms = self._get_classrooms()
            
            if not lessons:
                raise ValueError("No lessons found for the specified department and semester")
            
            # Run genetic algorithm
            best_schedule, best_fitness, generation_data = self._genetic_algorithm(
                lessons, instructors, classrooms, opt_run, progress_callback
            )
            
            # Save results
            runtime = time.time() - start_time
            conflicts = self._count_conflicts(best_schedule)
            
            opt_run.status = 'completed'
            opt_run.fitness_score = best_fitness
            opt_run.conflicts_count = conflicts
            opt_run.runtime_seconds = runtime
            opt_run.completed_at = datetime.utcnow()
            opt_run.results = {
                'classroom_utilization': self._calculate_classroom_utilization(best_schedule, classrooms),
                'instructor_balance': self._calculate_instructor_balance(best_schedule),
                'generation_data': generation_data[-10:]  # Last 10 generations
            }
            
            # Save schedule to database
            self._save_schedule_to_db(opt_run, best_schedule)
            
            db.session.commit()
            return opt_run
            
        except Exception as e:
            opt_run.status = 'error'
            opt_run.results = {'error': str(e)}
            db.session.commit()
            raise e
    
    def _get_lessons(self, department_id: int, semester: int) -> List[Lesson]:
        return Lesson.query.filter_by(
            department_id=department_id,
            semester=semester
        ).all()
    
    def _get_instructors(self, department_id: int) -> List[Instructor]:
        return Instructor.query.filter_by(
            department_id=department_id,
            is_active=True
        ).all()
    
    def _get_classrooms(self) -> List[Classroom]:
        return Classroom.query.filter_by(is_active=True).all()
    
    def _genetic_algorithm(self, lessons, instructors, classrooms, opt_run, progress_callback):
        """Implementation of genetic algorithm with constraint satisfaction"""
        
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            schedule = self._create_random_schedule(lessons, instructors, classrooms)
            fitness = self._fitness_function(schedule, instructors, classrooms)
            population.append((schedule, fitness))
        
        best_schedule = None
        best_fitness = 0
        generation_data = []
        
        for generation in range(self.config.generations):
            # Sort by fitness (descending)
            population.sort(key=lambda x: x[1], reverse=True)
            
            # Update best solution
            if population[0][1] > best_fitness:
                best_fitness = population[0][1]
                best_schedule = deepcopy(population[0][0])
            
            # Store generation stats
            fitnesses = [ind[1] for ind in population]
            gen_stats = {
                'generation': generation,
                'best_fitness': population[0][1],
                'avg_fitness': sum(fitnesses) / len(fitnesses),
                'conflicts': self._count_conflicts(population[0][0])
            }
            generation_data.append(gen_stats)
            
            # Update progress in database
            opt_run.progress = {
                'generation': generation,
                'total_generations': self.config.generations,
                'best_fitness': population[0][1],
                'conflicts': self._count_conflicts(population[0][0])
            }
            db.session.commit()
            
            # Progress callback
            if progress_callback:
                progress_callback(gen_stats)
            
            # Create new population
            new_population = []
            
            # Elitism - keep top individuals
            elite_count = int(self.config.population_size * self.config.elitism_rate)
            new_population.extend(population[:elite_count])
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1[0], parent2[0], lessons)
                else:
                    child1, child2 = deepcopy(parent1[0]), deepcopy(parent2[0])
                
                if random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1, lessons, instructors, classrooms)
                if random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2, lessons, instructors, classrooms)
                
                new_population.append((child1, self._fitness_function(child1, instructors, classrooms)))
                if len(new_population) < self.config.population_size:
                    new_population.append((child2, self._fitness_function(child2, instructors, classrooms)))
            
            population = new_population[:self.config.population_size]
        
        return best_schedule, best_fitness, generation_data
    
    def _create_random_schedule(self, lessons, instructors, classrooms):
        """Create a random valid schedule using constraint satisfaction principles"""
        schedule = {}
        
        # Sort lessons by difficulty (more constraints = higher priority)
        sorted_lessons = sorted(lessons, key=lambda l: l.total_hours + (10 if l.requires_lab else 0), reverse=True)
        
        for lesson in sorted_lessons:
            placed = False
            attempts = 0
            max_attempts = 100
            
            while not placed and attempts < max_attempts:
                # Select time slot
                max_start_hour = self.time_slots - lesson.total_hours
                if max_start_hour < 0:
                    # Lesson too long for single block, force placement
                    start_hour = 0
                else:
                    start_hour = random.randint(0, max_start_hour)
                
                day = random.randint(0, self.days - 1)
                
                # Select instructor
                suitable_instructors = [i for i in instructors 
                                      if any(ia.lesson_id == lesson.id for ia in i.lesson_assignments)]
                if not suitable_instructors:
                    suitable_instructors = [i for i in instructors if i.department_id == lesson.department_id]
                
                if suitable_instructors:
                    instructor = random.choice(suitable_instructors)
                else:
                    instructor = None
                
                # Select classroom
                if lesson.is_online:
                    classroom = None
                else:
                    suitable_classrooms = [
                        c for c in classrooms 
                        if c.capacity >= lesson.min_capacity
                        and (not lesson.requires_lab or c.has_lab)
                    ]
                    classroom = random.choice(suitable_classrooms) if suitable_classrooms else None
                
                # Check validity
                if self._is_valid_placement(schedule, lesson, instructor, classroom, start_hour, day):
                    schedule[lesson.id] = {
                        'lesson': lesson,
                        'instructor': instructor,
                        'classroom': classroom,
                        'start_hour': start_hour,
                        'day': day,
                        'duration': lesson.total_hours
                    }
                    placed = True
                
                attempts += 1
            
            if not placed:
                # Force placement with penalty
                schedule[lesson.id] = {
                    'lesson': lesson,
                    'instructor': None,
                    'classroom': None,
                    'start_hour': 0,
                    'day': 0,
                    'duration': lesson.total_hours,
                    'forced': True
                }
        
        return schedule
    
    def _is_valid_placement(self, schedule, lesson, instructor, classroom, start_hour, day):
        """Check if placement violates any constraints"""
        end_hour = start_hour + lesson.total_hours
        
        # Time bounds check
        if end_hour > self.time_slots:
            return False
        
        # Instructor availability check
        if instructor and instructor.availability:
            for hour in range(start_hour, end_hour):
                if hour < len(instructor.availability) and day < len(instructor.availability[hour]):
                    if not instructor.availability[hour][day]:
                        return False
        
        # Check conflicts with existing schedule
        for existing_id, existing in schedule.items():
            if existing.get('forced'):
                continue
                
            # Instructor conflict
            if (instructor and existing['instructor'] and 
                instructor.id == existing['instructor'].id and 
                existing['day'] == day and 
                not (end_hour <= existing['start_hour'] or 
                     start_hour >= existing['start_hour'] + existing['duration'])):
                return False
            
            # Classroom conflict
            if (classroom and existing['classroom'] and 
                classroom.id == existing['classroom'].id and 
                existing['day'] == day and 
                not (end_hour <= existing['start_hour'] or 
                     start_hour >= existing['start_hour'] + existing['duration'])):
                return False
        
        return True
    
    def _fitness_function(self, schedule, instructors, classrooms):
        """Multi-objective fitness function based on research"""
        score = 1000  # Start with perfect score
        
        # Hard constraints (heavy penalties)
        conflicts = self._count_conflicts(schedule)
        score -= conflicts * self.config.conflict_penalty
        
        # Soft constraints (optimization objectives)
        instructor_balance = self._calculate_instructor_balance(schedule)
        score += instructor_balance * self.config.instructor_balance_weight
        
        daily_distribution = self._calculate_daily_distribution(schedule)
        score += daily_distribution * self.config.daily_distribution_weight
        
        classroom_utilization = self._calculate_classroom_utilization(schedule, classrooms)
        score += classroom_utilization * self.config.classroom_utilization_weight
        
        preference_score = self._calculate_preference_score(schedule)
        score += preference_score * self.config.preference_weight
        
        return max(0, score)
    
    def _count_conflicts(self, schedule):
        """Count hard constraint violations"""
        conflicts = 0
        time_usage = {}  # (entity_id, entity_type, day, hour) -> lesson_id
        
        for lesson_id, placement in schedule.items():
            if placement.get('forced'):
                conflicts += 10
                continue
            
            lesson = placement['lesson']
            instructor = placement['instructor']
            classroom = placement['classroom']
            day = placement['day']
            start_hour = placement['start_hour']
            duration = placement['duration']
            
            for hour in range(start_hour, start_hour + duration):
                # Instructor conflicts
                if instructor:
                    key = (instructor.id, 'instructor', day, hour)
                    if key in time_usage:
                        conflicts += 1
                    time_usage[key] = lesson_id
                
                # Classroom conflicts
                if classroom:
                    key = (classroom.id, 'classroom', day, hour)
                    if key in time_usage:
                        conflicts += 1
                    time_usage[key] = lesson_id
        
        return conflicts
    
    def _calculate_instructor_balance(self, schedule):
        """Calculate workload balance among instructors"""
        instructor_hours = {}
        
        for placement in schedule.values():
            if placement.get('forced') or not placement['instructor']:
                continue
                
            instructor_id = placement['instructor'].id
            instructor_hours[instructor_id] = instructor_hours.get(instructor_id, 0) + placement['duration']
        
        if len(instructor_hours) < 2:
            return 10
        
        hours_list = list(instructor_hours.values())
        mean_hours = sum(hours_list) / len(hours_list)
        variance = sum((h - mean_hours) ** 2 for h in hours_list) / len(hours_list)
        
        return max(0, 10 - variance)
    
    def _calculate_daily_distribution(self, schedule):
        """Calculate how evenly lessons are distributed across days"""
        daily_counts = [0] * self.days
        
        for placement in schedule.values():
            if not placement.get('forced'):
                daily_counts[placement['day']] += 1
        
        if sum(daily_counts) == 0:
            return 0
        
        mean_daily = sum(daily_counts) / self.days
        variance = sum((c - mean_daily) ** 2 for c in daily_counts) / self.days
        
        return max(0, 10 - variance)
    
    def _calculate_classroom_utilization(self, schedule, classrooms):
        """Calculate classroom utilization efficiency"""
        if not classrooms:
            return 0
        
        classroom_hours = {}
        total_possible_hours = len(classrooms) * self.time_slots * self.days
        
        for placement in schedule.values():
            if placement.get('forced') or not placement['classroom']:
                continue
                
            classroom_id = placement['classroom'].id
            classroom_hours[classroom_id] = classroom_hours.get(classroom_id, 0) + placement['duration']
        
        total_used = sum(classroom_hours.values())
        utilization = (total_used / total_possible_hours) * 100 if total_possible_hours > 0 else 0
        
        return min(utilization, 100)  # Cap at 100%
    
    def _calculate_preference_score(self, schedule):
        """Calculate instructor preference satisfaction"""
        total_score = 0
        count = 0
        
        for placement in schedule.values():
            if placement.get('forced') or not placement['instructor']:
                continue
            
            instructor = placement['instructor']
            lesson = placement['lesson']
            
            # Find instructor-lesson assignment
            assignment = next((ia for ia in instructor.lesson_assignments 
                             if ia.lesson_id == lesson.id), None)
            
            if assignment:
                total_score += assignment.preference_level
                count += 1
        
        return (total_score / count) if count > 0 else 5
    
    def _tournament_selection(self, population):
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda x: x[1])
    
    def _crossover(self, parent1, parent2, lessons):
        """Order crossover for schedule optimization"""
        lesson_ids = [lesson.id for lesson in lessons]
        crossover_point = random.randint(1, len(lesson_ids) - 1)
        
        child1 = {}
        child2 = {}
        
        for i, lesson_id in enumerate(lesson_ids):
            if i < crossover_point:
                if lesson_id in parent1:
                    child1[lesson_id] = deepcopy(parent1[lesson_id])
                if lesson_id in parent2:
                    child2[lesson_id] = deepcopy(parent2[lesson_id])
            else:
                if lesson_id in parent2:
                    child1[lesson_id] = deepcopy(parent2[lesson_id])
                if lesson_id in parent1:
                    child2[lesson_id] = deepcopy(parent1[lesson_id])
        
        return child1, child2
    
    def _mutate(self, schedule, lessons, instructors, classrooms):
        """Mutation operation with constraint checking"""
        mutated = deepcopy(schedule)
        
        if not mutated:
            return mutated
        
        # Select random lesson to mutate
        lesson_ids = list(mutated.keys())
        lesson_id_to_mutate = random.choice(lesson_ids)
        lesson = mutated[lesson_id_to_mutate]['lesson']
        
        # Remove current placement temporarily
        temp_placement = mutated.pop(lesson_id_to_mutate)
        
        # Try to find new valid placement
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts:
            # Random new placement
            max_start_hour = self.time_slots - lesson.total_hours
            start_hour = random.randint(0, max(0, max_start_hour))
            day = random.randint(0, self.days - 1)
            
            # Select instructor
            suitable_instructors = [i for i in instructors if i.department_id == lesson.department_id]
            instructor = random.choice(suitable_instructors) if suitable_instructors else None
            
            # Select classroom
            if lesson.is_online:
                classroom = None
            else:
                suitable_classrooms = [
                    c for c in classrooms 
                    if c.capacity >= lesson.min_capacity and (not lesson.requires_lab or c.has_lab)
                ]
                classroom = random.choice(suitable_classrooms) if suitable_classrooms else None
            
            # Check if valid
            if self._is_valid_placement(mutated, lesson, instructor, classroom, start_hour, day):
                mutated[lesson_id_to_mutate] = {
                    'lesson': lesson,
                    'instructor': instructor,
                    'classroom': classroom,
                    'start_hour': start_hour,
                    'day': day,
                    'duration': lesson.total_hours
                }
                break
            
            attempts += 1
        
        # If no valid placement found, restore original
        if lesson_id_to_mutate not in mutated:
            mutated[lesson_id_to_mutate] = temp_placement
        
        return mutated
    
    def _save_schedule_to_db(self, opt_run, schedule):
        """Save optimized schedule to database"""
        for lesson_id, placement in schedule.items():
            if placement.get('forced'):
                continue
            
            schedule_entry = Schedule(
                optimization_run_id=opt_run.id,
                lesson_id=lesson_id,
                instructor_id=placement['instructor'].id if placement['instructor'] else None,
                classroom_id=placement['classroom'].id if placement['classroom'] else None,
                day_of_week=placement['day'],
                start_hour=placement['start_hour'],
                duration=placement['duration'],
                is_valid=not placement.get('forced', False)
            )
            db.session.add(schedule_entry)