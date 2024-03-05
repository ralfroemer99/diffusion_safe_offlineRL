import subprocess
import numpy as np

# List of arguments to pass to the script
arguments_list = [['--use_actions', 'True', '--scale', 1],
                  ['--use_actions', 'True', '--scale', 1],
                  ['--use_actions', 'True', '--scale', 1]]
ranges = [np.logspace(-1, 4, 11), 
          ]

# arguments_list = [['--use_actions', 'False'], ['--use_actions', 'True']]

# Path to the script to be executed
# script_path = 'train.py'
script_path = 'eval_pointmass.py'

for idx, arguments in enumerate(arguments_list):
    for val in ranges[idx]:
        arguments[-1] = val
    
        # Prepare the command
        command = ['python3', script_path] + arguments

        # Execute the script
        subprocess.run(command)