# shape-detector

A simple university project for detecting shapes and colors using a camera with 
[Python 3](https://github.com/python/cpython) and [OpenCV](https://github.com/opencv/opencv).

## Setup
If you are running an Ubuntu based Linux distribution, use ```setup.sh``` to install python and automatically setup
a virtual environment with the necessary dependencies. If you want to set it up manually, you will need to install
the following dependencies yourself: 

#### Dependencies
The following dependencies were used during development:
- Python ```3.12.3```
- OpenCV ```5.0.0```

## Run
If the setup was done correct, you can use ```run.sh``` to execute the application form the root directory. For manual
execution (without a virtual environment), use ```src/main.py``` as the execution file:

```bash
# Run with script
bash run.sh

# Run manually
python src/main.py
```

#### Controls

- ```Q``` -- Quit application
- ```Tab``` -- Toggle UI
- ```0-9``` -- Switch camera index
- ```W``` -- Go up in UI selection
- ```S``` -- Go down in UI selection
- ```A``` -- Enter sub-menu
- ```D``` -- Go back to main menu

