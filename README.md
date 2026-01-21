# Group2-AI-ML
    This is the repo for the first project of AI and Machine Learning, 2nd Bachelor Year of Computer Sciences, Maastricht University of Group 2. Our project consists of collecting data from an ML Agent environment and applying Machine Learning techniques to the collected data.

## Virtual Environment
    A SETUP.md is available in the repo to set the virtual environment.

    We used a Miniconda virtual environment to run the automation script, this doesn’t affect the results of our runs. Here is a list of instructions to use Miniconda :
    1. Install Miniconda on https://www.anaconda.com/docs/getting-started/miniconda/main
    2. In the wizard, uncheck "Register Miniconda3 as my default Python 3.12".
    3. Open the Anaconda Powershell Prompt
    4. Write the following command : conda create -n mlagents python=3.10.11 && conda activate mlagents
    5. Type "y"

    For future reference, every time you want to use this virtual environment, type "conda activate mlagents" in an Anaconda Powershell Prompt.
    You will know if you are in the environment if the (base) at the beginning of the prompt line is replaced by (mlagents).

## Sorter scripts
### Automation scripts to run trainings
    In the run_automation.py you can find under results\sorter\sorter_models\scripts\data, you can modify the details of each run you want to make, following the pattern you can find in the file. You can add more runs or delete some. The only rules are : 
    - Each run needs to have all the modified parameters of all the runs. Even if one run doesn’t change a certain parameter, it still needs to have it with its baseline value.
    - All runs need to have different names, otherwise the second and all runs after will not run.

    If you want to modify the number of environments: 
    Please uncomment line 140 and add a comma at the end of line 139

    If you want to modify other hyperparameters than buffer size, batch size, number of layers, hidden units, and number of environments:
    Please add after line 124 its “path” you can find in the Sorter_curriculum.yaml file in results/sorter, following the pattern of the other “config” lines, which you can find lines 121-124

    Once you’ve written all the runs you want :
    1. Open your terminal and set up the virtual environments (the steps to follow are in SETUP.md)
    2. Navigate using cd to the directory containing the script which is results\sorter\sorter_models\scripts\data
    3. Run the run_automation.py file with the command : python run_automation.py
    4. Let it run until you come back to being able to write in the terminal
    5. You can find your runs in results/sorter/YAMLchanged_Louise, please put your runs in the corresponding folder(s) or create new ones if needed.

### Preprocessing script
    You can find the preprocessing script PreProcessing.py in results\sorter\sorter_models\scripts\data. 
    Select the folder of the runs you want to check, copy its path in line 44, then run. You will see the anomalies in your terminal.

### Random Forest Model
    Navigate to the directory containing the model:
    results\sorter\sorter_models\scripts\model

    Then write this command in a virtual environment:

    You will be able to see the results in the temrinal

### Gradient Boosting Model
    To run the model, make sure the virtual environment is set up and activated (see SETUP.md).

    Navigate to the directory containing the model:
    results/sorter/sorter_models/scripts/model

    Run the script that trains and evaluates the Gradient Boosted Regression model:
    gbdt_model.py

    The script will load the preprocessed data, train the model, and output the prediction results and evaluation metrics in the terminal.

    
## Soccer Scripts
### Automation script to run trainings
    In the results/soccer folder, you can find the run_automation_soccer.py script. 
    Once mlagents is correctly installed on your device, you will need to make sure the build environment is correctly made. 
    You will need to first change the paths at the very top to match those on your device, most of which don’t need much changing. 
    This is to ensure that the results go to the correct folder and that the data can be interpreted properly. 
    You can then change the name of the master file on line 27 to a name of your choice. 
    Once everything is set up properly, you must open the Anaconda powershell, activate ML Agents, and then run the script using “python” followed by your path to the script.
    This will activate the automation script and keep running until all 27 unique runs are completed, while logging the information in a csv file.


### Prediction model
    Once the runs are over and you wish to test out the SVR prediction model, head over to SVR.py, found in the same folder under results/soccer, and change the file name at the very top to the path of the master file you previously renamed. 
    Once you’ve completed that, you may now run the model and it will give you the Model’s Accuracy (R^2 score) and the average prediction error (MAE). 
    ! Note that the SVR model needs at least 5 or so runs to be able to compare and give valid results.
