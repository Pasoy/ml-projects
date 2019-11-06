# Information / Maths
in [the notes file](NOTES.md)

# Installation Jupyter Notebook (on WSL - Ubuntu)

### Update and upgrade system and install Python
```
sudo apt update && upgrade

sudo apt install python3 python3-pip ipython3
```

### Install Jupyter
```
sudo pip3 install jupyter
```

### Open config file
```
nano ~/.bashrc
```

### Below **esac** type
```
alias jupyter-notebook="~/.local/bin/jupyter-notebook --no-browser"
```
Then save and exit (CTRL-X -> Y -> ENTER)

### Source file
```
source ~/.bashrc
```

## Start Jupyter Notebook
```
jupyter notebook
```
