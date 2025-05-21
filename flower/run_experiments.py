import subprocess
import os

# Configuration
NUM_RUNS = 5  # Number of times to run the experiment

# Command to execute
# Assumes flwr is in the system PATH and the script is run from the workspace root.
# It also assumes your server and client apps are correctly specified in your project.
# By default, `flwr run` looks for a `pyproject.toml` or specific app modules.
# If your `flwr run` command needs more specific arguments (e.g., to point to your apps
# if they are not auto-discoverable with default names like server_app.py/client_app.py
# or specified in pyproject.toml), you need to add them here.
# For example: FLOWER_COMMAND = ["flwr", "run", "--server-app", "flower_normal.server_app:app", "--client-app", "flower_normal.client_app:app"]
FLOWER_COMMAND = ["flwr", "run"]

def run_single_experiment(run_number):
    """Runs a single Flower experiment and captures its output."""
    print(f"--- Starting Experiment Run {run_number + 1}/{NUM_RUNS} ---")
    
    # Define output files for this run
    output_filename = f"run_{run_number + 1}_output.txt"
    # Error file is no longer separate, stderr is redirected to stdout
    
    try:
        with open(output_filename, 'w') as outfile:
            # Execute the command, redirecting stderr to stdout
            process = subprocess.Popen(FLOWER_COMMAND, stdout=outfile, stderr=subprocess.STDOUT, text=True)
            process.wait() # Wait for the command to complete
            
        if process.returncode == 0:
            print(f"Experiment Run {run_number + 1} completed successfully. Output in {output_filename}")
        else:
            # If there was an error, its output will also be in output_filename
            print(f"Experiment Run {run_number + 1} failed with return code {process.returncode}. Check {output_filename} for details.")
            
    except FileNotFoundError:
        print(f"Error: The command 'flwr' was not found. Make sure Flower is installed and in your system's PATH.")
        print(f"Attempted command: {' '.join(FLOWER_COMMAND)}")
        # Write the error to the output file for this run as well, if it was opened
        try:
            with open(output_filename, 'a') as outfile: # append mode
                outfile.write(f"\nError: The command 'flwr' was not found.\nAttempted command: {' '.join(FLOWER_COMMAND)}")
        except UnboundLocalError: # outfile might not be defined if FileNotFoundError happened before `with open`
            pass
        return False 
    except Exception as e:
        print(f"An error occurred during run {run_number + 1}: {e}")
        # Write the error to the output file for this run
        try:
            with open(output_filename, 'a') as outfile: # append mode
                outfile.write(f"\nAn unexpected error occurred: {e}")
        except UnboundLocalError:
            pass
        return False
        
    return True # Indicate success or failure based on process return code

if __name__ == "__main__":
    successful_runs = 0
    failed_runs = 0
    command_not_found = False

    # Ensure an `output` directory exists for logs (optional, but good practice)
    if not os.path.exists("experiment_logs"):
        os.makedirs("experiment_logs")

    for i in range(NUM_RUNS):
        if not run_single_experiment(i):
            # Check if FileNotFoundError was the reason for failure
            # This check is a bit indirect; a more robust way might be to return specific error types from run_single_experiment
            is_file_not_found = False
            try:
                with open(f"run_{i + 1}_output.txt", 'r') as f_check:
                    if "Error: The command 'flwr' was not found." in f_check.read():
                        is_file_not_found = True
            except FileNotFoundError:
                 # If the output file itself doesn't exist, it's likely the command failed very early.
                is_file_not_found = True 
            
            if is_file_not_found:
                print("Stopping further attempts as 'flwr' command seems to be an issue.")
                command_not_found = True
                break # Exit the loop
            failed_runs += 1
        else:
            successful_runs += 1
        print("-----------------------------------------------------")
        
    print("\n--- Experiment Series Summary ---")
    if command_not_found:
        print(f"Could not execute 'flwr run'. Please check Flower installation and PATH.")
    else:
        print(f"Total runs attempted: {NUM_RUNS}")
        print(f"Successful runs: {successful_runs}")
        print(f"Failed runs: {failed_runs}")