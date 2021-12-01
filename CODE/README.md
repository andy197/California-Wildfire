## Purpose

Codebase for db, training models, and 
generating various artifacts from those actions to supplement the report and
poster. 
I want to try to get the data loads added to this if I have the time, but model
metrics for the report/poster are the primary goal.

## Files and Packages

### main.py
The main file that should be run to go through the full process of training the
models and generating the metrics for the report.

At the top of this file are some variables that should be populated with your
own credentials to connect to the database. I recommend you make a copy of this
file so that you don't accidentally check your credentials into the repo

Additionally, there are flags to turn off and on some of the features that are 
available. I added comments throughout to highlight what there features are.

### requirements.txt
The packages and versions I used during my development. I am using Python 3.8 
too. I had to make some manual corrections to the file when I exported my own
env, so let me know if something doesn't work for you.

You should be able to run:

```bash
pip install -r requirements.txt
``` 

I recommend you create a virtual environment before running the previous 
command. 

### Data
The Classes related to retrieving the training, testing, and eventually 
visualization data (data use to generate the data that feeds the 
visualization). Has some methods within to generate descriptive statistics
about the training data that will be saved within this folder.

### Models
The Classes related to the training of the models as well as their metrics
that are saved within this folder.

### Visualization
Will contain the Classes related to the generation of the visualization data.