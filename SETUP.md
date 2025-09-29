# Environment Setup
## Cloning Repo
- Before doing anything, clone the repo :  
    - Create a new Folder on your computer where you would like your repo to be (through File Explorer)  
    - Copy the path of this folder  
    - Open a command prompt and run :  
        cd "path", then type enter  
    - Click on Code on the repo page and copy the URL  
    - On your command Prompt run :  
        git clone "URL", then type enter  
      You may have to connect to git on your browser, just click on Authorize git-ecosystem and put your password in
      
## Virtual Environment
The virtual environment is here to make sure we all have the same version of packages etc.  
I don't have a Mac or Linux, so contact me if you have any trouble (Louise Gillain)  
Considering everyone already installed Unity and Python 3.10.11 :  
  - Run the automation script (only the first time you open the project):  
      Mac/Linux : bash setup.sh (if error try : pwsh -ExecutionPolicy Bypass -File ./setup.sh)  
      Windows (PowerShell) : .\setup.ps1 (if error try : powershell -ExecutionPolicy Bypass -File .\setup.ps1)  
  - Activate the virtual environment (every time you work) :  
      Mac/Linux : source venv/bin/activate (if error try : pwsh .\venv/Scripts/Activate.ps1)  
      Windows (PowerShell) : .\venv\Scripts\Activate.ps1 (if error try : powershell -ExecutionPolicy Bypass -File .\venv\Scripts\Activate.ps1)  
  - If requirements.txt has been updated run :  
      pip install -r requirements.txt (Same on Bash and PowerShell)  
    Right now at the beginning of the project, the requirements.txt file doesn't exist yet, we will fill it over time  
  - When finished with working on code deactivate :  
      deactivate (Same on Bash and PowerShell) or just close the Bash/PowerShell
