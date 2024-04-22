import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm


def train(model, train_loader, train_eval_loader, val_seen_loader, val_unseen_loader, optimizer, criterion, epochs, device, model_dir_path):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        train_losses = []

        for i, batch in tqdm(enumerate(train_loader)):
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
        evaluate(model, train_eval_loader, criterion, device, 'Train')
        evaluate(model, val_seen_loader, criterion, device, 'Validation Seen')
        evaluate(model, val_unseen_loader, criterion, device, 'Validation Unseen')


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
        metrics['roc_auc'] = roc_auc_score(labels, preds)

        print(
            f'{kind} loss: {metrics["loss"]:.3f}, macro f1: {metrics["macro_f1"]:.3f}, ' + \
            f'micro f1: {metrics["micro_f1"]:.3f}, accuracy: {metrics["accuracy"]:.3f}, ' + \
            f'precision: {metrics["precision"]:.3f}, recall: {metrics["recall"]:.3f}, roc_auc: {metrics["roc_auc"]:.3f}'
        )

def train_freeze(model, train_loader, val_loader, optimizer, criterion, epochs, device, model_dir_path):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        train_losses = []

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            image, text, label = batch
            output = model(text, image)
            # label_tensor = torch.nn.functional.one_hot(label).to(dtype=torch.float32) # if using BCEWithLogitsLoss with 2 output
            # label = label.reshape(-1, 1).to(dtype=torch.float32) # if using BCEWithLogitsLoss with 1 output

            loss = criterion(output['text_image'], label) + criterion(output['image'], label) + criterion(output['text'], label)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)

            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f'{model_dir_path}/model_freeze_{epoch + 1}.pth')
        print(f'Epoch {epoch + 1} average loss: {np.mean(train_losses):.3f}')
        evaluate_freeze(model, train_loader, criterion, device, 'Train')
        evaluate_freeze(model, val_loader, criterion, device)

def evaluate_freeze(model, data_loader, criterion, device, kind='Evaluation'):
    with torch.no_grad():
        losses, preds, labels = [], [], []
        for batch in tqdm(data_loader):
            image, text, label = batch

            out = model(text, image)
            # label_tensor = torch.nn.functional.one_hot(label).to(dtype=torch.float32) # if using BCEWithLogitsLoss with 2 output
            # label = label.reshape(-1, 1).to(dtype=torch.float32) # if using BCEWithLogitsLoss with 1 output

            loss = criterion(out['text_image'], label) + criterion(out['image'], label) + criterion(out['text'], label)
            losses.append(loss.item())

            # voting mechanism
            pred1 = np.argmax((torch.sigmoid(out['text_image'])).cpu().detach().numpy(), axis=1)
            pred2 = np.argmax((torch.sigmoid(out['image'])).cpu().detach().numpy(), axis=1)
            pred3 = np.argmax((torch.sigmoid(out['text'])).cpu().detach().numpy(), axis=1)
            pred = np.mean([pred1, pred2, pred3], axis=0)
            pred = np.round(pred)

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
        metrics["ROC-AUC"] = roc_auc_score(labels, preds)

        print(
            f'{kind} loss: {metrics["loss"]:.3f}, macro f1: {metrics["macro_f1"]:.3f}, ' + \
            f'micro f1: {metrics["micro_f1"]:.3f}, accuracy: {metrics["accuracy"]:.3f}, ' + \
            f'precision: {metrics["precision"]:.3f}, recall: {metrics["recall"]:.3f}, ' + \
            f'ROC-AUC: {metrics["ROC-AUC"]:.3f}'
        )
