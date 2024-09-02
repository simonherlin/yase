import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from ...utils.dowload_utils import download_monodepth_weight
from .networks import (ResnetEncoder, DepthDecoder)

class Monodepth2Estimator:
    def __init__(self):
        """
        Initializes the Monodepth2Estimator. Downloads model weights if they are not present.
        """
        self.weight_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'depth', 'monodepth2'))

        # Ensure that model weights are downloaded
        self._ensure_weights_downloaded()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        encoder_path = os.path.join(self.weight_path, 'encoder.pth')
        depth_decoder_path = os.path.join(self.weight_path, 'depth.pth')
        # # Load the models
        self.encoder = ResnetEncoder(18, False)
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)


        self.loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(self.loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()

        feed_width = 640
        feed_height = 192
        self.transform = transforms.Compose([
            transforms.Resize((feed_height, feed_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def _ensure_weights_downloaded(self):
        """
        Checks if the model weights are downloaded, and downloads them if necessary.
        """
        encoder_url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip"
        required_md5checksum = "a964b8356e08a02d009609d9e3928f7c"
        # Download files if they do not exist
        download_monodepth_weight(
            url=encoder_url,
            hash=required_md5checksum,
            dest_path=self.weight_path
        )

    def predict_depth(self, image_path):
        """
        Predicts the depth map for a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            depth (numpy.ndarray): Calculated depth map.
        """
        try:
            input_image = Image.open(image_path).convert('RGB')
            original_width, original_height = input_image.size

            input_image = self.transform(input_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.encoder(input_image)
                outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
            depth = 1 / disp_resized.squeeze().cpu().numpy()

            return depth

        except Exception as e:
            print(f"Error in predicting depth: {e}")
            return None
