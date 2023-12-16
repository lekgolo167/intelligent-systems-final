
import argparse
import os
import sys
import pytorch_lightning as pl
import time
from peewee import SqliteDatabase
from lichess_model import Evaluations, EvaluationModel
from datetime import datetime

# Command-line arguments parsing
parser = argparse.ArgumentParser(description='Convert month name to a 2-character number.')
parser.add_argument('-db', dest='db_name', required=True, action='store', type=str, help='Name of the input DB under data/SQL/')
parser.add_argument('-ckpt',dest='checkpoint_name', required=True, action='store', type=str, help='Name for the model\'s checkopoint file')
args = parser.parse_args()

# Paths to the SQLite database file and the model's checkpoint file
sqlite_file_path = os.path.join('data', 'SQL', args.db_name)
checkpoint_file_path = os.path.join('data', 'MODEL', f'{args.checkpoint_name}.ckpt')
# Check if the SQLite database file exists
if not os.path.exists(sqlite_file_path):
  print(f'Database for {sqlite_file_path}, does not exist')
  sys.exit(0)

# Connect to the SQLite database
db = SqliteDatabase(sqlite_file_path)
Evaluations._meta.database = db

if __name__ == '__main__':

  import multiprocessing
  multiprocessing.freeze_support()
  
  # Count the number of rows in the database
  db.connect()
  cursor = db.execute_sql('SELECT COUNT(*) FROM evaluations;')
  num_of_evals = cursor.fetchone()[0]
  
  print(f'{num_of_evals} evaluation scores avalaible for training')

  layer_count = 1
  batch_size = 512
  epochs = 1

  # Generate a unique version name based on the current timestamp and configuration
  date = str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
  version_name = f'{date}-batch_size-{batch_size}-layer_count-{layer_count}'
  # Set up PyTorch Lightning logger for TensorBoard
  logger = pl.loggers.TensorBoardLogger("lightning_logs", name="lichess", version=version_name)
  # Configure and train the PyTorch Lightning Trainer
  trainer = pl.Trainer(precision=16, max_epochs=epochs, logger=logger)
  # Initialize the EvaluationModel
  model = EvaluationModel(count=num_of_evals, layer_count=layer_count, batch_size=batch_size, learning_rate=1e-3)

  # Train the model
  trainer.fit(model)
  # Save the model checkpoint
  trainer.save_checkpoint(filepath=checkpoint_file_path)
  print(f'Model saved to: {checkpoint_file_path}')
