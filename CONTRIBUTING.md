# How to contribute to the project
## Version Control
- The default branch name is main (when pushing you don't need to write the whole repo name, just main)
- Feature branches : create a branch when adding a new feature, naming convention = "feature/'feature-name'"
- Bugfix branches : create a branch when fixing bugs, naming convention = "fix/'bug-name'"

- You can create a branch either directly on GitHub in the branch dropdown or by using the command :
    git checkout -b 'branch-name' (this creates the branch and goes to it directly)
- To upload your branch to GitHub :
    git push origin 'branch-name'
- To push updated branch :
    git add .
    git commit -m "Describe changes"
    git push origin 'branch-name'

- Go from one branch to another using :
    git checkout 'branch-name'
- Once your branch is finished, you can merge it to main by first opening a Pull Request (at the top of the GitHub repo page next to Issues)
  Then, after your request was accepted :
    git checkout main
    git pull origin main
