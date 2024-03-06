import subprocess
import numpy as np

# List of arguments to pass to the script
arguments_list = [['--use_actions', 'True', '--scale', 1, 'batch_size', 1]]
scale_range = [np.logspace(-1, 4, 11), 
          ]
batch_size_range = [[1, 2, 4, 8]]

# arguments_list = [['--use_actions', 'False'], ['--use_actions', 'True']]

# Path to the script to be executed
# script_path = 'train.py'
script_paths = ['eval_pointmass.py', 'eval_quad2d.py']

for script_path in script_paths:
    for idx, arguments in enumerate(arguments_list):
        for val1 in scale_range[idx]:
            arguments[3] = str(val1)

            for val2 in batch_size_range[idx]:
                arguments[5] = str(val2)
                # Prepare the command
                command = ['python3', script_path] + arguments

                print(command)

                # Execute the script
                subprocess.run(command)
