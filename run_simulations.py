import subprocess
import time
from datetime import datetime
import logging
import sys
import os

def setup_logging():
    """Setup logging configuration"""
    log_dir = 'simulation_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove old log files
    for file in os.listdir(log_dir):
        os.remove(os.path.join(log_dir, file))
    
    # Create new log file
    log_file = os.path.join(log_dir, 'simulation.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Simplified format
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def run_simulation(num_runs=1):
    """Run simulation with clean logging"""
    log_file = setup_logging()
    logging.info(f"Starting simulation suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        logging.info("\nSimulation Run")
        logging.info("-" * 40)
        
        # Set environment variables
        my_env = os.environ.copy()
        my_env["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging
        my_env["PYTHONUNBUFFERED"] = "1"      # Ensure output is not buffered
        
        # Run the simulation
        process = subprocess.Popen(
            ['python', 'testing.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,  # Line buffered
            env=my_env
        )
        
        # Process output in real-time with immediate flushing
        while process.poll() is None:  # Check if process is still running
            # Read from stdout and stderr
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()
            
            if stdout_line:
                clean_output = stdout_line.strip()
                logging.info(clean_output)
                sys.stdout.flush()
                
            if stderr_line:
                # Filter out TF/CUDA warnings
                if not any(x in stderr_line.lower() for x in ['tensorflow', 'cuda', 'cudnn']):
                    logging.error(stderr_line.strip())
                    sys.stdout.flush()
        
        # Get final return code
        returncode = process.wait()
        
        # Check for any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            for line in remaining_stdout.splitlines():
                if line:
                    logging.info(line.strip())
                    
        if remaining_stderr:
            error_lines = [line for line in remaining_stderr.splitlines() if line and 
                         not any(x in line.lower() for x in ['tensorflow', 'cuda', 'cudnn'])]
            if error_lines:
                logging.error("\nRemaining simulation errors:")
                for line in error_lines:
                    logging.error(line.strip())
                    sys.stdout.flush()
                return False
        
        return returncode == 0
            
    except KeyboardInterrupt:
        if 'process' in locals():
            try:
                process.terminate()  # Try graceful termination first
                try:
                    process.wait(timeout=3)  # Wait up to 3 seconds
                except subprocess.TimeoutExpired:
                    process.kill()  # Force kill if still running
                    process.wait()
            except Exception as e:
                logging.error(f"Error terminating process: {str(e)}")
        logging.warning("\nSimulation interrupted by user")
        sys.stdout.flush()
        return False
    except Exception as e:
        logging.error(f"\nUnexpected error: {str(e)}")
        sys.stdout.flush()
        return False
    finally:
        # Ensure process is terminated if it exists
        if 'process' in locals() and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=1)
            except Exception:
                try:
                    process.kill()
                    process.wait()
                except Exception as e:
                    logging.error(f"Failed to terminate process in cleanup: {str(e)}")

if __name__ == "__main__":
    try:
        success = run_simulation(num_runs=1)
        if success:
            logging.info("\nSimulation completed successfully!")
        else:
            logging.error("\nSimulation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("\nProgram interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Program failed with error: {str(e)}")
        sys.exit(1)