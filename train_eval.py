import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from statistics import mode
from sklearn.metrics import accuracy_score, precision_score, recall_score

def grid_search(model, train_loader, val_loader, loss_fn, learning_rates, weight_decays, num_epochs=5):
    
    results = []

    # Perform the grid search over learning rates and weight decays
    for lr in learning_rates:
        for wd in weight_decays:
            print(f'Learning Rate: {lr} || Weight Decay: {wd}')
            # Instantiate a fresh copy of the model to avoid reusing old weights
            current_model = type(model)()  # Calls the model class constructor
            optimizer = torch.optim.Adam(current_model.parameters(), lr=lr, weight_decay=wd)

            # Train the model and collect results for each epoch
            train_result = train_model(current_model, train_loader, val_loader, loss_fn, optimizer, num_epochs)

            # Loop through all epochs and store the results
            for epoch in range(num_epochs):
                results.append({
                    'Epoch': epoch + 1,
                    'Learning Rate': lr,
                    'Weight Decay': wd,
                    'Training Loss': train_result['Training Loss'][epoch],
                    'Validation Loss': train_result['Validation Loss'][epoch],
                    'Validation Accuracy': train_result['Validation Accuracy'][epoch]
                })

    # Convert the results list to a DataFrame for better visualization
    df_results = pd.DataFrame(results)

    # Identify the best result based on validation accuracy (at the last epoch per combination)
    best_result = max(results, key=lambda x: x['Validation Accuracy'])

    print("Best result based on Validation Accuracy:")
    print(best_result)

    return {'All Results': results, 'Best Result': best_result}, df_results

def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=5):
    ''' Use function to train model. 
        Prints training loss, validation loss, and validation accuracy after every epoch. 
        Also returns all the losses and accuracies in lists
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    final_train_loss = []
    final_val_loss = []
    final_val_accuracy = []
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()  # Ensure the model is in training mode
        
        train_loss = 0
        train_size = 0

        # Wrap the DataLoader with tqdm for the progress bar, without overwriting the original DataLoader
        for images, labels, _ in tqdm(train_loader, desc=f"Training"):
            
            train_size += images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Calculate average loss over an epoch
        train_loss /= train_size

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
            
            val_loss = 0
            val_size = 0

            correct = 0
            total = 0
            # Again, use tqdm without overwriting the DataLoader
            for images, labels, _ in tqdm(val_loader, desc=f"Validation"):
                val_size += images.size(0)

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                #Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss /= val_size
            val_accuracy = 100 * correct / total
        
        final_train_loss.append(train_loss)
        final_val_loss.append(val_loss)
        final_val_accuracy.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    return {'Training Loss' : final_train_loss, 
            'Validation Loss' : final_val_loss, 
            'Validation Accuracy' : final_val_accuracy}

def train_model_with_early_stopping(model, train_loader, val_loader, loss_fn, optimizer, num_epochs=5, patience=3):
    ''' Use function to train model with early stopping.
        Prints training loss, validation loss, and validation accuracy after every epoch.
        Also returns all the losses and accuracies in lists.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    final_train_loss = []
    final_val_loss = []
    final_val_accuracy = []

    best_val_loss = float('inf')  # Initialize to infinity for the best validation loss
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()  # Ensure the model is in training mode
        
        train_loss = 0
        train_size = 0

        # Wrap the DataLoader with tqdm for the progress bar
        for images, labels, _ in tqdm(train_loader, desc=f"Training"):
            train_size += images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # Calculate average loss over an epoch
        train_loss /= train_size

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Turn off gradients for validation
            
            val_loss = 0
            val_size = 0

            correct = 0
            total = 0
            # Validation loop
            for images, labels, _ in tqdm(val_loader, desc=f"Validation"):
                val_size += images.size(0)
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_loss /= val_size
            val_accuracy = 100 * correct / total

        final_train_loss.append(train_loss)
        final_val_loss.append(val_loss)
        final_val_accuracy.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered. Training stopped at epoch {epoch + 1}.")
            break

    return {
        'Training Loss': final_train_loss,
        'Validation Loss': final_val_loss,
        'Validation Accuracy': final_val_accuracy
    }


def evaluate_model(model, data_loader):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    predictions = defaultdict(list)
    actual_labels = {}
    all_frame_preds = []
    all_frame_true = []

    with torch.no_grad():
        for images, labels, vids in data_loader:
            
            images = images.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()


            for vid, pred, label in zip(vids, predicted, labels):
                predictions[vid].append(pred)
                actual_labels[vid] = label
                all_frame_preds.append(pred)
                all_frame_true.append(label)

    # frame-by-frame accuracy
    frame_accuracy = accuracy_score(all_frame_true, all_frame_preds)

    # Get majority votes
    vid_true = []
    vid_pred = []
    for vid, preds in predictions.items():
        majority_vote = mode(preds)
        actual_label = actual_labels[vid]
        vid_pred.append(majority_vote)
        vid_true.append(actual_label)

    # Calculate majority vote accuracy, precision and recall
    majority_accuracy = accuracy_score(vid_true, vid_pred)
    majority_precision = precision_score(vid_true, vid_pred, average='macro')
    majority_recall = recall_score(vid_true, vid_pred, average='macro')

    return {
        'majority_accuracy': majority_accuracy,
        'majority_precision': majority_precision,
        'majority_recall': majority_recall,
        'frame_accuracy': frame_accuracy
    }

