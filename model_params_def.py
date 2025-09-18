"""
Model-specific params
If no params are required by the model, then it should be an empty list.
"""

from improvelib.utils import str2bool

preprocess_params = []

train_params = [
    {"name": "model",
     "type": str,
     "default": "default",
     "help": "Model architecture to run. One of 'default', 'small', or 'large'."
    }, 
]

infer_params = []