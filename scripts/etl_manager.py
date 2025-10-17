#!/usr/bin/env python3
"""
ETL Pipeline Management Script for RiskX.

This script provides easy management of the ETL pipeline including:
- Starting/stopping the scheduler
- Running manual data updates
- Checking pipeline status
- Viewing task history
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import signal
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from etl.scheduler import ETLScheduler
from etl.tasks.fred_fetch import run_fred_etl_task, FREDDataFetcher


def start_scheduler(background: bool = False):
    """Start the ETL scheduler."""
    print("Starting ETL scheduler...")
    
    if background:
        # Run in background
        cmd = [sys.executable, "-m", "etl.scheduler"]
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Save PID for later stopping
        pid_file = project_root / "logs" / "etl_scheduler.pid"
        pid_file.parent.mkdir(exist_ok=True)
        
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"ETL scheduler started in background (PID: {process.pid})")
        print(f"PID saved to: {pid_file}")
        print("Use 'python scripts/etl_manager.py stop' to stop the scheduler")
        
    else:
        # Run in foreground
        scheduler = ETLScheduler()
        try:
            scheduler.start()
        except KeyboardInterrupt:
            print("\nStopping scheduler...")
            scheduler.stop()


def stop_scheduler():
    """Stop the ETL scheduler."""
    pid_file = project_root / "logs" / "etl_scheduler.pid"
    
    if not pid_file.exists():
        print("No scheduler PID file found. Scheduler may not be running.")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Try to terminate the process
        os.kill(pid, signal.SIGTERM)
        print(f"ETL scheduler (PID: {pid}) stopped successfully")
        
        # Remove PID file
        pid_file.unlink()
        
    except (FileNotFoundError, ProcessLookupError):
        print("Scheduler process not found. It may have already stopped.")
        if pid_file.exists():
            pid_file.unlink()
    except Exception as e:
        print(f"Error stopping scheduler: {e}")


def run_manual_update(force_refresh: bool = False):
    """Run a manual data update."""
    print("Running manual FRED data update...")
    
    try:
        result = run_fred_etl_task(force_refresh=force_refresh)
        
        print(f"Update completed with status: {result['status']}")
        print(f"Timestamp: {result['timestamp']}")
        
        if result['status'] == 'success':
            print(f"Indicators updated: {len(result['indicators'])}")
        
        if result.get('errors'):
            print(f"Errors encountered: {len(result['errors'])}")
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        # Save result to file
        result_file = project_root / "logs" / f"manual_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_file.parent.mkdir(exist_ok=True)
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"Detailed results saved to: {result_file}")
        
    except Exception as e:
        print(f"Manual update failed: {e}")


def check_status():
    """Check the status of the ETL pipeline."""
    print("ETL Pipeline Status Report")
    print("=" * 50)
    
    # Check if scheduler is running
    pid_file = project_root / "logs" / "etl_scheduler.pid"
    scheduler_running = False
    
    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process is actually running
            os.kill(pid, 0)  # This will raise an exception if process doesn't exist
            scheduler_running = True
            print(f"✅ Scheduler running (PID: {pid})")
            
        except (FileNotFoundError, ProcessLookupError, OSError):
            print("❌ Scheduler not running")
            if pid_file.exists():
                pid_file.unlink()  # Clean up stale PID file
    else:
        print("❌ Scheduler not running")
    
    # Check data freshness
    print("\nData Freshness Check:")
    try:
        fetcher = FREDDataFetcher()
        validation = fetcher.validate_data_freshness()
        
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌"
        }
        
        print(f"{status_emoji.get(validation['overall_status'], '❓')} Overall status: {validation['overall_status']}")
        
        for indicator, status in validation['indicators'].items():
            status_emoji_ind = {
                "fresh": "✅",
                "stale": "⚠️",
                "missing": "❌",
                "error": "❌"
            }
            print(f"  {status_emoji_ind.get(status, '❓')} {indicator}: {status}")
        
        if validation['issues']:
            print("\nIssues found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
    
    except Exception as e:
        print(f"❌ Error checking data status: {e}")
    
    # Check ETL metrics
    print("\nETL Metrics:")
    try:
        fetcher = FREDDataFetcher()
        metrics = fetcher.get_etl_metrics()
        
        fred_metrics = metrics['data_sources']['fred']
        print(f"  FRED Status: {fred_metrics['status']}")
        print(f"  Cache Hit Rate: {fred_metrics['cache_hit_rate']:.1%}")
        print(f"  Cached Indicators: {fred_metrics['cached_indicators']}/{fred_metrics['total_indicators']}")
        
    except Exception as e:
        print(f"❌ Error getting ETL metrics: {e}")


def view_task_history(limit: int = 10):
    """View recent task execution history."""
    history_file = project_root / "logs" / "etl_task_history.json"
    
    if not history_file.exists():
        print("No task history found.")
        return
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print(f"Recent ETL Task History (last {limit} tasks)")
        print("=" * 60)
        
        recent_tasks = history[-limit:] if len(history) > limit else history
        
        for task in reversed(recent_tasks):  # Show most recent first
            status_emoji = {
                "completed": "✅",
                "failed": "❌",
                "running": "🔄"
            }
            
            emoji = status_emoji.get(task['status'], '❓')
            duration = f"{task.get('duration_seconds', 0):.1f}s" if task.get('duration_seconds') else "N/A"
            
            print(f"{emoji} {task['task_name']}")
            print(f"   Start: {task['start_time']}")
            print(f"   Status: {task['status']} (Duration: {duration})")
            
            if task.get('error'):
                print(f"   Error: {task['error']}")
            
            print()
    
    except Exception as e:
        print(f"Error reading task history: {e}")


def validate_data():
    """Run data validation checks."""
    print("Running data validation...")
    
    try:
        fetcher = FREDDataFetcher()
        validation = fetcher.validate_data_freshness()
        
        print("Data Validation Report")
        print("=" * 40)
        print(f"Overall Status: {validation['overall_status']}")
        print(f"Validation Time: {validation['timestamp']}")
        
        print("\nIndicator Status:")
        for indicator, status in validation['indicators'].items():
            print(f"  {indicator}: {status}")
        
        if validation['issues']:
            print("\nIssues Found:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        else:
            print("\n✅ No issues found!")
        
        return validation['overall_status'] == 'healthy'
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline Management for RiskX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/etl_manager.py start              # Start scheduler in foreground
  python scripts/etl_manager.py start --background # Start scheduler in background
  python scripts/etl_manager.py stop               # Stop background scheduler
  python scripts/etl_manager.py update             # Run manual data update
  python scripts/etl_manager.py update --force     # Force refresh all data
  python scripts/etl_manager.py status             # Check pipeline status
  python scripts/etl_manager.py history            # View task history
  python scripts/etl_manager.py validate           # Run data validation
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the ETL scheduler')
    start_parser.add_argument('--background', '-b', action='store_true',
                            help='Run scheduler in background')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop the ETL scheduler')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Run manual data update')
    update_parser.add_argument('--force', '-f', action='store_true',
                             help='Force refresh all data')
    
    # Status command
    subparsers.add_parser('status', help='Check ETL pipeline status')
    
    # History command
    history_parser = subparsers.add_parser('history', help='View task execution history')
    history_parser.add_argument('--limit', '-l', type=int, default=10,
                               help='Number of recent tasks to show')
    
    # Validate command
    subparsers.add_parser('validate', help='Run data validation checks')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'start':
        start_scheduler(background=args.background)
    elif args.command == 'stop':
        stop_scheduler()
    elif args.command == 'update':
        run_manual_update(force_refresh=args.force)
    elif args.command == 'status':
        check_status()
    elif args.command == 'history':
        view_task_history(limit=args.limit)
    elif args.command == 'validate':
        is_healthy = validate_data()
        sys.exit(0 if is_healthy else 1)


if __name__ == "__main__":
    main()