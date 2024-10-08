import torch
# from .models.segmentation.sfnet_lite import SFNetLite
from .models.depth.monodepth2_estimator import Monodepth2Estimator
from .utils import preprocessing, postprocessing, onnx_export

class Yase:
    def __init__(self, fp16=False, use_onnx=False, **kwargs):
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
        self.model = {}
        # Initialize the model based on the task
        self._initialize_models(**kwargs)

        if self.use_onnx:
            self.model = onnx_export.export_model_to_onnx(self.model, self.device)

    def _initialize_models(self, *kwargs):
        """
        Internal method to initialize the appropriate model based on the task.
        
        :param model_name: Name of the model to initialize.
        :param kwargs: Additional arguments for model-specific settings.
        :return: Initialized model.
        """
        self.model["depth"] = Monodepth2Estimator()
        # self.model['segmentation'] = 
        # if self.task == "segmentation" and model_name == "SFNetLite":
        #     return SFNetLite(**kwargs).to(self.device)
        # elif self.task == "depth" and model_name == "MiDaS":
        #     return MiDaS(**kwargs).to(self.device)
        # else:
        #     raise ValueError(f"Unsupported task or model: {self.task} - {model_name}")

    def run_inference(self, input_data):
        """
        Run inference using the selected model.
        
        :param input_data: Input data for the model (e.g., image).
        :return: Model output.
        """
        preprocessed_input = preprocessing(input_data, task=self.task)
        with torch.no_grad():
            if self.fp16:
                with torch.cuda.amp.autocast():
                    output = self.model(preprocessed_input)
            else:
                output = self.model(preprocessed_input)
        return postprocessing(output, task=self.task)
