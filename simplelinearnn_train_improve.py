import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_utils import LinearRegressionModel, DRPDataset, predicting, determine_input_dim
from torch.utils.data import DataLoader

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
import improvelib.utils as frm

# Model-specifc imports
from model_params_def import train_params

filepath = Path(__file__).resolve().parent 

def run(params):
    # ------------------------------------------------------
    # [Req] Build model path and data names for train and val sets
    # ------------------------------------------------------
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    train_data = torch.load(Path(params["input_dir"]) / train_data_fname)
    val_data = torch.load(Path(params["input_dir"]) / val_data_fname)

    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['val_batch'], shuffle=False)

    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
    else:
        device = torch.device("cpu")   # Use CPU
    # Prepare model and train settings
    # Model, Loss, and Optimizer

    input_dim = determine_input_dim(train_loader)
    print("input_dim", input_dim)
    model = LinearRegressionModel(input_dim=input_dim, dropout_prob=0.01).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])

    best_loss = np.inf
    early_stop_counter = 0
    early_stop = False
    # Training loop
    for epoch in range(params['epochs']):
        for batch_data, batch_labels in train_loader:
            # Convert numpy arrays to torch tensors
            #batch_data = torch.tensor(batch_data, dtype=torch.float32)
            #batch_labels = torch.tensor(batch_labels, dtype=torch.long)

            # Forward pass
            #print("Epoch:", epoch)
            #print("batch_data:", batch_data)
            #print("batch_lables:", batch_labels)
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.unsqueeze(1).to(device)
            outputs = model(batch_data)
            #print("outputs:", outputs)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad() # Clear gradients
            loss.backward()       # Compute gradients
            optimizer.step()      # Update weights

        # Validation
        val_true, val_pred = predicting(model, val_loader, device)
        val_loss = criterion(torch.tensor(val_pred), torch.tensor(val_true)).item()
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            torch.save(model, modelpath)
        else:
            early_stop_counter += 1
            if early_stop_counter >= params['patience']:
                early_stop = True
        print(f"Epoch [{epoch+1}/{params['epochs']}], Loss: {val_loss:.4f}, Early stop count: {early_stop_counter}/{params['patience']}")
        if early_stop:
            break

    if not early_stop:
        torch.save(model, modelpath)



    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    best_model = torch.load(modelpath, weights_only=False)
    best_model.to(device)

    # Compute predictions
    val_true, val_pred = predicting(best_model, val_loader, device)
    print("val_pred:", val_pred)
    print("val_true:", val_true)
    #val_pred = val_pred.numpy().flatten()
   
     # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=val_true, 
        y_pred=val_pred, 
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores


# [Req]
def main(args):
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="simplelinearnn_params.ini",
                                       additional_definitions=train_params)
    timer_train = frm.Timer()    
    val_scores = run(params)
    timer_train.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_train.json', 
                           extra_dict={"stage": "train"})
    print("\nFinished model training.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
    

