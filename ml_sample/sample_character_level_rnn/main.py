import math
import random
import time
from typing import Tuple

import torch
import torch.nn as nn

from load import load
from preprocess import line2tensor
from rnn import RNN
from util import ALL_LETTERS


N_HIDDEN = 128
N_ITERATOR = 5000
N_CONFUSION = 1000


def output2category(y: torch.Tensor, all_categories: list) -> Tuple[str, int]:
    _, top_i = y.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_training_example(all_categories: list, category_lines: dict) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
    category = all_categories[random.randint(0, len(all_categories)-1)]
    line = category_lines[category][random.randint(0, len(category_lines[category])-1)]
    category_tensor = torch.Tensor([all_categories.index(category)]).long()
    line_tensor = line2tensor(line)
    return category, line, category_tensor, line_tensor


def train(model: RNN, criterion: nn.NLLLoss, all_categories: list, category_lines: dict, learning_rate: float=0.05) -> float:
    train_loss = 0.0
    start_tm = time.time()

    for i in range(1, 1+N_ITERATOR):
        category, line, category_tensor, line_tensor = random_training_example(all_categories, category_lines)

        hidden = model.initHidden().cuda()
        model.zero_grad()
        for j in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[j].cuda(), hidden)
        
        loss = criterion(output, category_tensor.cuda())
        train_loss += loss.item()
        loss.backward()

        for param in model.parameters():
            param.data.add_(param.grad.data, alpha=-learning_rate)

        if i % 100 == 0:
            guess, _ = output2category(output, all_categories)
            correct = "ok" if guess == category else f"ng ({category})"
            tm = time.time() - start_tm
            m = math.floor(tm/60)
            s = int(tm % 60)
            print(f"{i:6d} {i/N_ITERATOR*100}% ({m}m {s}s) {loss.item():.4f} {line} {guess} {correct}")

    return train_loss


def evaluate(model: RNN, all_categories: list, category_lines: dict):
    confusion = torch.zeros(len(all_categories), len(all_categories))

    model.eval()
    with torch.no_grad():
        for i in range(N_CONFUSION):
            category, _, _, line_tensor = random_training_example(all_categories, category_lines)
            hidden = model.initHidden().cuda()

            for j in range(line_tensor.size()[0]):
                output, hidden = model(line_tensor[j].cuda(), hidden)
            _, guess_i = output2category(output, all_categories)
            category_i = all_categories.index(category)
            confusion[category_i][guess_i] += 1
        
        for i in range(len(all_categories)):
            confusion[i] = confusion[i] / confusion[i].sum()
    
    print("evaluate:")
    print(confusion)
    print()


def predict(model: RNN, all_categories: list, line: str):
    with torch.no_grad():
        tensor = line2tensor(line)
        hidden = model.initHidden().cuda()
        for i in range(tensor.size()[0]):
            output, hidden = model(tensor[i].cuda(), hidden)
        
        topv, topi = output.topk(1, 1, True)
        value = topv[0][0].item()
        category_i = topi[0][0].item()
        print(f"({value:.2f}) {all_categories[category_i]}")


def main():
    all_categories, category_lines = load("data/names/*.txt")

    model = RNN(len(ALL_LETTERS), N_HIDDEN, len(all_categories)).cuda()
    criterion = nn.NLLLoss()

    train(model, criterion, all_categories, category_lines)
    evaluate(model, all_categories, category_lines)

    predict(model, all_categories, "Satoshi")
    predict(model, all_categories, "Jackson")

    print("DONE")


if __name__ == "__main__":
    main()
