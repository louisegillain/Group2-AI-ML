# Environment Setup
## Cloning Repo
- Before doing anything, clone the repo :
    - Create a new Folder on your computer where you would like your repo to be (through File Explorer)
    - Copy the path of this folder
    - Open a command prompt and run :
        cd < path >, then type enter
    - Click on Code on the repo page and copy the URL
    - On your command Prompt run :
        git clone < URL >, then type enter
      
## Virtual Environment
The virtual environment is here to make sure we all have the same version of packages etc.
Considering everyone already installed Unity and Python :
  - Run the automation script (only the first time you open the project):
      Mac/Linux : bash setup.sh
      Windows (PowerShell) : .\setup.ps1
  - Activate the virtual environment (every time you work) :
      Mac/Linux : source venv/bin/activate
      Windows (PowerShell) : .\venv\Scripts\Activate.ps1
  - If requirements.txt has been updated run :
      pip install -r requirements.txt (Same on Bash and PowerShell)
    Right now at the beginning of the project, the requirements.txt file doesn't exist yet, we will fill it over time
  - When finished with working on code deactivate :
      deactivate (Same on Bash and PowerShell)
