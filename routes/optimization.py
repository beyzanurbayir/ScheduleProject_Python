# ===== routes/optimization.py =====
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from models.database import db, Department, OptimizationRun
from models.optimization import AdvancedScheduleOptimizer, OptimizationConfig
import threading
import uuid
import time

optimization_bp = Blueprint('optimization', __name__)

# Global storage for optimization progress
optimization_sessions = {}

@optimization_bp.route('/')
def index():
    departments = Department.query.all()
    recent_runs = OptimizationRun.query.order_by(OptimizationRun.created_at.desc()).limit(10).all()
    return render_template('optimization/index.html', departments=departments, recent_runs=recent_runs)

@optimization_bp.route('/configure', methods=['GET', 'POST'])
def configure():
    if request.method == 'POST':
        try:
            department_id = int(request.form['department_id'])
            grade = int(request.form['grade'])

            # Parse optimization parameters
            config = OptimizationConfig(
                population_size=int(request.form.get('population_size', 100)),
                generations=int(request.form.get('generations', 200)),
                mutation_rate=float(request.form.get('mutation_rate', 0.1)),
                crossover_rate=float(request.form.get('crossover_rate', 0.8)),
                tournament_size=int(request.form.get('tournament_size', 5)),
                elitism_rate=float(request.form.get('elitism_rate', 0.2))
            )
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Start optimization in background thread
            def run_optimization():
                try:
                    optimizer = AdvancedScheduleOptimizer(config)
                    optimization_sessions[session_id] = {'status': 'running', 'progress': {}}
                    
                    def progress_callback(progress_data):
                        optimization_sessions[session_id]['progress'] = progress_data
                    
                    result = optimizer.optimize_schedule(
                        department_id, grade, session_id, progress_callback
                    )
                    
                    optimization_sessions[session_id]['status'] = 'completed'
                    optimization_sessions[session_id]['result'] = result
                    
                except Exception as e:
                    optimization_sessions[session_id]['status'] = 'error'
                    optimization_sessions[session_id]['error'] = str(e)
            
            thread = threading.Thread(target=run_optimization, daemon=True)
            thread.start()
            
            return redirect(url_for('optimization.progress', session_id=session_id))
            
        except (ValueError, Exception) as e:
            flash(f'Error starting optimization: {str(e)}', 'error')
    
    departments = Department.query.all()
    return render_template('optimization/configure.html', departments=departments)

@optimization_bp.route('/progress/<session_id>')
def progress(session_id):
    if session_id not in optimization_sessions:
        flash('Optimization session not found!', 'error')
        return redirect(url_for('optimization.index'))
    
    session_data = optimization_sessions[session_id]
    
    if session_data['status'] == 'completed':
        return redirect(url_for('results.view', run_id=session_data['result'].id))
    
    return render_template('optimization/progress.html', 
                         session_id=session_id, 
                         session_data=session_data)

@optimization_bp.route('/api/progress/<session_id>')
def api_progress(session_id):
    if session_id not in optimization_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(optimization_sessions[session_id])

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
    
    departments = Department.query.all()
    
    return render_template('optimization/runs.html', 
                         runs=runs, 
                         departments=departments,
                         selected_dept=dept_id,
                         selected_status=status)