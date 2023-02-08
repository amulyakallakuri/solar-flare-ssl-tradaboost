import subspace
import tradaboost


if __name__ == "__main__":
    code = int(input('''
    Which transfer learning code would you like to run?
    Enter 1 for TrAdaBoost and 2 for Subspace Alignment
    '''))
    if code == 1:
        tradaboost.execute()
    elif code == 2:
        subspace.execute()

# Upon running this file multiple times,
# it is observed that the Adapt Implementation for TrAdaBoost in tradaboost.py halts execution and does not proceed at times,
# even though the code in the .py file is the exact same as the code in the .ipynb file.
# The code in the tradaboost.py file is only a beautified version of the .ipynb file which adheres to the main ->
# -> get data from data file -> call to execution format required as per the guidelines.

# If the Adapt Implementation for TrAdaBoost on tradaboost.py (using this main.py) does not work,
# please refer to the .ipynb file for the executed results.