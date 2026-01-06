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
from model import ACCESSNet
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
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--valid_data", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--assay", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--metric_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    return parser.parse_args()


def train(model, dataloader, criterion, optimizer, device, assay: str = "seq"):
    model.train()

    train_loss = 0.0
    for data in dataloader:
        seq = data['seq']
        atac_signal = data['atac_signal']
        access_signal = data['access_signal']
        target = data['label']

        if assay == "seq":
            pred = model(seq.to(device)).view(-1)
        elif assay == "atac":
            pred = model(seq.to(device), atac_signal.to(device)).view(-1)
        elif assay == "access":
            pred = model(seq.to(device), access_signal.to(device)).view(-1)
        elif assay == "both":
            pred = model(seq.to(device),
                         atac_signal.to(device),
                         access_signal.to(device)).view(-1)
        else:
            raise ValueError(f"Unsupported assay type: {assay}")
        
        loss = criterion(pred.float(), target.to(device).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(dataloader)

    return train_loss


def valid(model, dataloader, criterion, device, assay: str = "seq"):
    model.eval()

    valid_loss = 0.0
    for data in dataloader:
        seq = data['seq']
        atac_signal = data['atac_signal']
        access_signal = data['access_signal']
        target = data['label']

        if assay == "seq":
            pred = model(seq.to(device)).view(-1)
        elif assay == "atac":
            pred = model(seq.to(device), atac_signal.to(device)).view(-1)
        elif assay == "access":
            pred = model(seq.to(device), access_signal.to(device)).view(-1)
        elif assay == "both":
            pred = model(seq.to(device),
                         atac_signal.to(device),
                         access_signal.to(device)).view(-1)
        else:
            raise ValueError(f"Unsupported assay type: {assay}")
        loss = criterion(pred.float(), target.to(device).float())

        valid_loss += loss.item() / len(dataloader)

    return valid_loss


def predict(model, dataloader, device, assay: str = "seq"):
    model.eval()

    preds = []
    with torch.no_grad():
        for data in dataloader:
            seq = data['seq']
            atac_signal = data['atac_signal']
            access_signal = data['access_signal']

            if assay == "seq":
                pred = model(seq.to(device)).view(-1)
            elif assay == "atac":
                pred = model(seq.to(device), atac_signal.to(device)).view(-1)
            elif assay == "access":
                pred = model(seq.to(device), access_signal.to(device)).view(-1)
            elif assay == "both":
                pred = model(seq.to(device),
                             atac_signal.to(device),
                             access_signal.to(device)).view(-1)
            else:
                raise ValueError(f"Unsupported assay type: {assay}")
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    return preds


def main():
    args = parse_args()

    set_seed(args.seed)

    logging.info("Loading input files")
    train_data = np.load(args.train_data)
    valid_data = np.load(args.valid_data)
    test_data = np.load(args.test_data)

    train_dataloader = get_dataloader(
        seq=train_data['seq'],
        atac_signal=train_data['signal_atac'],
        access_signal=train_data['signal_access'],
        label=train_data['label'],
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
    )
    valid_dataloader = get_dataloader(
        seq=valid_data['seq'],
        atac_signal=valid_data['signal_atac'],
        access_signal=valid_data['signal_access'],
        label=valid_data['label'],
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    test_dataloader = get_dataloader(
        seq=test_data['seq'],
        atac_signal=test_data['signal_atac'],
        access_signal=test_data['signal_access'],
        label=test_data['label'],
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
    )

    # Setup model
    model = ACCESSNet(peak_len=256)

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
            assay=args.assay
        )
        valid_loss = valid(
            dataloader=valid_dataloader, 
            model=model, 
            criterion=criterion, 
            device=device,
            assay=args.assay
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
        dataloader=test_dataloader, 
        model=model, 
        device=device,
        assay=args.assay
    )

    # save test true labels and predictions
    test_df = pd.DataFrame(data={
        "true": test_data['label'],
        "pred": test_preds,
    })
    test_df.to_csv(f"{args.pred_dir}/{args.out_name}.csv", index=False)

    # plot AUPR curve for test set
    precision, recall, _ = precision_recall_curve(test_data['label'], test_preds)
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
