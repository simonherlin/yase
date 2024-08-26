import torch
from models.segmentation.sfnet_lite import SFNetLite
from models.depth.midas import MiDaS
from utils import preprocess, postprocess, optimization, onnx_export

class Yase:
    def __init__(self, task="segmentation", model_name="SFNetLite", fp16=False, use_onnx=False, **kwargs):
        """
        Initialize the Yase class with the desired task and model.
        
        :param task: Type of task ('segmentation', 'depth').
        :param model_name: Name of the model to use (default is 'SFNetLite' for segmentation).
        :param fp16: Whether to use FP16 precision.
        :param use_onnx: Whether to export and use the ONNX model.
        :param kwargs: Additional arguments for model-specific settings.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fp16 = fp16
        self.use_onnx = use_onnx
        self.task = task
        
        # Initialize the model based on the task
        self.model = self._initialize_model(model_name, **kwargs)

        if self.use_onnx:
            self.model = onnx_export.export_model_to_onnx(self.model, self.device)

    def _initialize_model(self, model_name, **kwargs):
        """
        Internal method to initialize the appropriate model based on the task.
        
        :param model_name: Name of the model to initialize.
        :param kwargs: Additional arguments for model-specific settings.
        :return: Initialized model.
        """
        if self.task == "segmentation" and model_name == "SFNetLite":
            return SFNetLite(**kwargs).to(self.device)
        elif self.task == "depth" and model_name == "MiDaS":
            return MiDaS(**kwargs).to(self.device)
        else:
            raise ValueError(f"Unsupported task or model: {self.task} - {model_name}")

    def run_inference(self, input_data):
        """
        Run inference using the selected model.
        
        :param input_data: Input data for the model (e.g., image).
        :return: Model output.
        """
        preprocessed_input = preprocess(input_data, task=self.task)
        with torch.no_grad():
            if self.fp16:
                with torch.cuda.amp.autocast():
                    output = self.model(preprocessed_input)
            else:
                output = self.model(preprocessed_input)
        return postprocess(output, task=self.task)
