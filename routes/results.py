# ===== routes/results.py =====
from flask import Blueprint, render_template, request, jsonify
from models.database import db, OptimizationRun, Schedule, Department
from sqlalchemy import func

results_bp = Blueprint('results', __name__)

@results_bp.route('/')
def index():
    # Get recent successful runs
    recent_runs = OptimizationRun.query.filter_by(status='completed').order_by(
        OptimizationRun.completed_at.desc()
    ).limit(10).all()
    
    return render_template('results/index.html', recent_runs=recent_runs)

@results_bp.route('/run/<int:run_id>')
def view(run_id):
    opt_run = OptimizationRun.query.get_or_404(run_id)
    
    if opt_run.status != 'completed':
        return render_template('results/incomplete.html', opt_run=opt_run)
    
    # Get schedule entries
    schedules = Schedule.query.filter_by(optimization_run_id=run_id).all()
    schedule_data = [s.to_dict() for s in schedules]
    
    # Calculate statistics
    stats = {
        'total_lessons': len(schedules),
        'assigned_lessons': len([s for s in schedules if s.instructor_id and s.classroom_id]),
        'online_lessons': len([s for s in schedules if not s.classroom_id]),
        'conflicts': opt_run.conflicts_count,
        'fitness_score': opt_run.fitness_score,
        'runtime': opt_run.runtime_seconds
    }
    
    return render_template('results/view.html', 
                         opt_run=opt_run, 
                         schedules=schedule_data,
                         stats=stats)

@results_bp.route('/run/<int:run_id>/export')
def export(run_id):
    opt_run = OptimizationRun.query.get_or_404(run_id)
    schedules = Schedule.query.filter_by(optimization_run_id=run_id).all()
    
    # Create export data
    export_data = {
        'optimization_info': opt_run.to_dict(),
        'schedule': [s.to_dict() for s in schedules]
    }
    
    return jsonify(export_data)

@results_bp.route('/compare')
def compare():
    run_ids = request.args.getlist('run_ids', type=int)
    
    if len(run_ids) < 2:
        return render_template('results/compare_select.html')
    
    runs = OptimizationRun.query.filter(OptimizationRun.id.in_(run_ids)).all()
    
    comparison_data = []
    for run in runs:
        schedules = Schedule.query.filter_by(optimization_run_id=run.id).all()
        
        comparison_data.append({
            'run': run.to_dict(),
            'schedule_count': len(schedules),
            'assigned_count': len([s for s in schedules if s.instructor_id]),
            'classroom_utilization': run.results.get('classroom_utilization', 0) if run.results else 0
        })
    
    return render_template('results/compare.html', comparison_data=comparison_data)