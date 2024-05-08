import subprocess

# List of arguments to pass to the script
arguments_list = [['--use_actions', 'True'],]
# arguments_list = [['--use_actions', 'False'], ['--use_actions', 'True']]

# Path to the script to be executed
script_path = 'train.py'
# script_path = 'eval_pointmass.py'

for arguments in arguments_list:
    # Prepare the command
    command = ['python3', script_path] + arguments

    # Execute the script
    subprocess.run(command)