import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm


def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_dir_path):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        train_losses = []

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            image, text, label = batch
            image = image.to(device)
            text = {k: v.to(device) for k, v in text.items()}
            label = label.to(device)

            output = model(text, image)
            loss = criterion(output, label)

            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_losses[-1]:.3f}')

        torch.save(model.state_dict(), f'{model_dir_path}/model_{epoch + 1}.pth')

        print(f'Epoch {epoch + 1} average loss: {np.mean(train_losses):.3f}')
        evaluate(model, train_loader, criterion, device, 'Train')
        evaluate(model, val_loader, criterion, device)


def evaluate(model, data_loader, criterion, device, kind='Evaluation'):
    with torch.no_grad():
        losses, preds, labels = [], [], []
        for batch in tqdm(data_loader):
            image, text, label = batch
            image = image.to(device)
            text = {k: v.to(device) for k, v in text.items()}
            label = label.to(device)

            out = model(text, image)
            loss = criterion(out, label)
            losses.append(loss.item())

            pred = np.argmax(torch.sigmoid(out).cpu().detach().numpy(), axis=1)

            preds.append(pred.tolist())
            labels.append(label.cpu().numpy())
        
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)

        metrics = {"loss": np.mean(losses)}
        metrics["macro_f1"] = f1_score(labels, preds, average="macro")
        metrics["micro_f1"] = f1_score(labels, preds, average="micro")
        metrics["accuracy"] = accuracy_score(labels, preds)
        metrics["precision"] = precision_score(labels, preds)
        metrics["recall"] = recall_score(labels, preds)

        print(
            f'{kind} loss: {metrics["loss"]:.3f}, macro f1: {metrics["macro_f1"]:.3f}, ' + \
            f'micro f1: {metrics["micro_f1"]:.3f}, accuracy: {metrics["accuracy"]:.3f}, ' + \
            f'precision: {metrics["precision"]:.3f}, recall: {metrics["recall"]:.3f}'
        )
