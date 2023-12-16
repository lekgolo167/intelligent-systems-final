# Intelligent Systems Final Project on LiChess AI
### By: Matthew Crump
---
## Contents
1. [Overview](#overview)
2. [Dependancies](#dependancies)
2. [Modules](#modules)
3. [Running](#running)

---
## Overview
#### Files
  * ```lichess_db_processory.py``` This file will fetch a Lichess datbase file, decompress it, filter it for games that contain evaluations then convert the board into a bit board representation, saving it to a database.
  * ```train_lichess_model.py``` This file will take in a database and use it to trian a model.
  * ```evaluate_lichess_model.py``` This file will take in a model and then evaluate it compared to another database.
  * ```lichess_model.py``` Contains the database model definitions for the evaluate and training modules.
  * ```ai_verses.py``` Use a trained model to verse a random move AI to see how well the model performs.
---
## Dependancies
  Python 3.10 was used for this project
  * chess==1.10.0
  * ipython==8.9.0
  * numpy==1.24.4
  * peewee==3.17.0
  * pytorch_lightning==2.1.2
  * Requests==2.31.0
  * torch==2.0.1
  * zstandard==0.22.0
---
## Running
* #### Step 1: Fetching Data
  Use the following command to fetch a lichess database. The month and year must be specified in the commandline args. Refer to [lichess's database](https://database.lichess.org/) to know the database size before downloading.
```python3 .\lichess_db_processor.py -m Jan -y 2013```
* #### Step 2: Train the Model
  To traint he model, run the following Python file with the database specified from the previous step and then supply the name of the model to be saved.
  ```python .\train_lichess_model.py -db lichess_db_standard_rated_2013-01.sqlite -ckpt lichess-model-1```
* #### Step 3: Evaluate the model
  To evaluate the model, run the following Python file with the model's checkpoint file specified and a different database than the one used to train the model.
  ```python .\evaluate_lichess_model.py -db lichess_db_standard_rated_2013-03.sqlite -ckpt lichess-model-1```
  The ai_verses.py file may also be used to evaluate the model. The model name is hardcoded, change it before running. It simply pits the AI against a random move AI and tallies up the total matches won by each AI out of 100. The best I can do is get a model that wins 83 of 100 matches.
