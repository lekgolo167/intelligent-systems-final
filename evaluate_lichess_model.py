import argparse
import os
import sys
import chess
import chess.svg
import torch
import torch.nn.functional as F
from lichess_model import Evaluations, EvaluationDataset, EvaluationModel
from random import randrange
from peewee import SqliteDatabase
import time
import statistics

parser = argparse.ArgumentParser(description='Convert month name to a 2-character number.')
parser.add_argument('-db', dest='db_name', required=True, action='store', type=str, help='Name of the input DB under data/SQL/')
parser.add_argument('-ckpt',dest='checkpoint_name', required=True, action='store', type=str, help='Name for the model\'s checkopoint file')

args = parser.parse_args()
sqlite_file_path = os.path.join('data', 'SQL', args.db_name)
if not os.path.exists(sqlite_file_path):
    print(f'Database for {sqlite_file_path}, does not exist')
    sys.exit(0)

checkpoint_file_path = os.path.join('data', 'MODEL', f'{args.checkpoint_name}.ckpt')

db = SqliteDatabase(sqlite_file_path)

Evaluations._meta.database = db

db.connect()
cursor = db.execute_sql('SELECT COUNT(*) FROM evaluations;')
num_of_evaluations = cursor.fetchone()[0]


dataset = EvaluationDataset(count=num_of_evaluations)

model = EvaluationModel.load_from_checkpoint(checkpoint_path=checkpoint_file_path)


def show_index(idx:int) -> None:
    eval = Evaluations.select().where(Evaluations.id == idx+1).get()
    batch = dataset[idx]
    x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
    y_hat = model(x)
    loss = F.l1_loss(y_hat, y)
    print(f'Evaluation score: {y.data[0]:.2f} Predicted score: {y_hat.data[0]:.2f} Loss {loss:.2f}')
    print(f'FEN: {eval.fen}')
    board = chess.Board(eval.fen)
    board_svg = chess.svg.board(board=board)
    with open("chess.svg", "w") as svg_file:
        svg_file.write(board_svg)
    os.startfile("chess.svg")

for i in range(5):
    idx = randrange(num_of_evaluations)
    show_index(idx)
    time.sleep(1)


def zero_loss(idx:int) -> float:
	eval = Evaluations.select().where(Evaluations.id == idx+1).get()
	y = torch.tensor(eval.evaluation)
	y_hat = torch.zeros_like(y)
	loss = F.l1_loss(y_hat, y)
	return loss


def model_loss(idx:int) -> float:
	batch = dataset[idx]
	x, y = torch.tensor(batch['binary']), torch.tensor(batch['eval'])
	y_hat = model(x)
	loss = F.l1_loss(y_hat, y)
	return loss


zero_losses = []
model_losses = []

for i in range(1000):
	idx = randrange(num_of_evaluations)
	zero_losses.append(zero_loss(idx))
	model_losses.append(model_loss(idx))

print(f'model zero average Loss {statistics.mean(zero_losses)}')
print(f'Model average Loss {statistics.mean(model_losses)}')