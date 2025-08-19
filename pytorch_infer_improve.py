import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_utils import predicting
from torch.utils.data import DataLoader

# [Req] IMPROVE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
import improvelib.utils as frm

# Model-specifc imports
from model_params_def import infer_params

filepath = Path(__file__).resolve().parent 

def run(params):
    # ------------------------------------------------------
    # [Req] Build model path and create data name for test set
    # ------------------------------------------------------
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"],
                                     model_file_format=params["model_file_format"],
                                     model_dir=params["input_model_dir"])
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    # ------------------------------------------------------
    # Load model input data (ML data)
    # ------------------------------------------------------
    test_data = torch.load(Path(params["input_data_dir"]) / test_data_fname)
 

    test_loader = DataLoader(test_data, batch_size=params['infer_batch'], shuffle=False)


    # ------------------------------------------------------
    # Prepare, train, and save model
    # ------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
    else:
        device = torch.device("cpu")   # Use CPU


    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    best_model = torch.load(modelpath, weights_only=False)
    best_model.to(device)

    # Compute predictions
    test_true, test_pred = predicting(best_model, test_loader, device)

   
     # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=test_true, 
        y_pred=test_pred, 
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=test_true, 
            y_pred=test_pred, 
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="pytorch_params.ini",
                                       additional_definitions=infer_params)
    timer_train = frm.Timer()    
    status = run(params)
    timer_train.save_timer(dir_to_save=params["output_dir"], 
                           filename='runtime_infer.json', 
                           extra_dict={"stage": "infer"})
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
    

