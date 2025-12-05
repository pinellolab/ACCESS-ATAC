import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import logging
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from model import MaxATACCNN
from model_v2 import AccNet
from utils import set_seed
from dataset import get_dataloader

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--metric_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    return parser.parse_args()


def train(model, dataloader, criterion, optimizer, device):
    model.train()

    train_loss = 0.0
    for x, y in dataloader:
        pred = model(x.to(device)).view(-1)
        loss = criterion(pred.float(), y.to(device).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(dataloader)

    return train_loss


def valid(model, dataloader, criterion, device):
    model.eval()

    valid_loss = 0.0
    for x, y in dataloader:
        pred = model(x.to(device)).view(-1)
        loss = criterion(pred.float(), y.to(device).float())

        valid_loss += loss.item() / len(dataloader)

    return valid_loss


def predict(model, dataloader, device):
    model.eval()

    preds = []
    with torch.no_grad():
        for x in dataloader:
            pred = model(x.to(device)).view(-1)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    return preds


def main():
    args = parse_args()

    set_seed(args.seed)

    logging.info("Loading input files")
    data = np.load(args.data)

    train_dataloader = get_dataloader(
        x=data['train_x'],
        y=data['train_y'],
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        train=True,
    )
    valid_dataloader = get_dataloader(
        x=data['valid_x'],
        y=data['valid_y'],
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        train=True,
    )

    test_dataloader = get_dataloader(
        x=data['test_x'],
        y=data['test_y'],
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        train=False,
    )

    # Setup model
    logging.info(f"Input channels: {data['train_x'].shape[2]}")
    model = AccNet(in_ch=data['train_x'].shape[2], n_blocks=3)

    logging.info(f"training data size: {len(data['train_x'])}")
    logging.info(f"validation data size: {len(data['valid_x'])}")
    logging.info(f"test data size: {len(data['test_x'])}")

    device = torch.device(f"cuda:{args.cuda}")
    model.to(device)

    # Setup loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=3e-04, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, "min", min_lr=1e-5, patience=2, factor=0.5)

    """ Train the model """
    logging.info("Training started")
    best_score = np.inf

    epochs, train_losses, valid_losses, best_scores = [], [], [], []
    for epoch in range(args.epochs):
        train_loss = train(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        valid_loss = valid(
            dataloader=valid_dataloader, model=model, criterion=criterion, device=device
        )

        # save model if find a better validation score
        if valid_loss < best_score:
            best_score = valid_loss
            state = {
                "state_dict": model.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "epoch": epoch,
            }
            torch.save(state, f"{args.model_dir}/{args.out_name}.pth")
            # Reset patience counter
            patience = 10
        else:
            # early stop
            patience -= 1
            if patience == 0:
                logging.info("Early stop!")
                break

        logging.info(
            f"epoch: {epoch}, train: {train_loss:.5f}, valid: {valid_loss:.5f}, best: {best_score:.5f}")
        scheduler.step(valid_loss)

        epochs.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        best_scores.append(best_score)

    df = pd.DataFrame(data={"epoch": epochs,
                            "train_loss": train_losses,
                            "valid_loss": valid_losses,
                            "best_loss": best_scores})

    df.to_csv(f"{args.log_dir}/{args.out_name}.csv", index=False)

    # plot training log
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, valid_losses, label="valid_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{args.log_dir}/{args.out_name}.png")
    plt.close()
    logging.info(f"Training finished")

    logging.info("Evaluating on test set")
    state = torch.load(f"{args.model_dir}/{args.out_name}.pth")
    model.load_state_dict(state["state_dict"])

    test_preds = predict(
        dataloader=test_dataloader, model=model, device=device
    )

    # save test true labels and predictions
    test_df = pd.DataFrame(data={
        "true": data['test_y'],
        "pred": test_preds,
    })
    test_df.to_csv(f"{args.pred_dir}/{args.out_name}.csv", index=False)

    # plot AUPR curve for test set
    precision, recall, _ = precision_recall_curve(data['test_y'], test_preds)
    aupr = auc(recall, precision)
    logging.info(f"AUPR: {aupr:.5f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUPR: {aupr:.5f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"{args.metric_dir}/{args.out_name}_aupr.png")
    plt.close()

if __name__ == "__main__":
    main()
