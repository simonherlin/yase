import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from yase.utils.download_utils import download_file
import yase.models.networks as networks

class Monodepth2Estimator:
    def __init__(self, encoder_path=None, decoder_path=None):
        """
        Initializes the Monodepth2Estimator. Downloads model weights if they are not present.

        Args:
            encoder_path (str): Path to the encoder model file.
            decoder_path (str): Path to the decoder model file.
        """
        if encoder_path is None:
            encoder_path = os.path.join(os.path.dirname(__file__), '../weights/mono+stereo_640x192/encoder.pth')
        if decoder_path is None:
            decoder_path = os.path.join(os.path.dirname(__file__), '../weights/mono+stereo_640x192/depth.pth')

        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

        # Ensure that model weights are downloaded
        self._ensure_weights_downloaded()

        # Load the models
        self.encoder = networks.ResnetEncoder(18, False)
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.depth_decoder.load_state_dict(torch.load(self.decoder_path))

        self.encoder.eval()
        self.depth_decoder.eval()

    def _ensure_weights_downloaded(self):
        """
        Checks if the model weights are downloaded, and downloads them if necessary.
        """
        encoder_url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/models/mono+stereo_640x192/encoder.pth"
        decoder_url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/models/mono+stereo_640x192/depth.pth"

        # Download files if they do not exist
        download_file(encoder_url, self.encoder_path)
        download_file(decoder_url, self.decoder_path)

    def predict_depth(self, image_path):
        """
        Predicts the depth map for a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            depth (numpy.ndarray): Calculated depth map.
        """
        input_image = Image.open(image_path)
        original_width, original_height = input_image.size
        feed_width = 640
        feed_height = 192

        transform = transforms.Compose([
            transforms.Resize((feed_height, feed_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        input_image = transform(input_image).unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
        depth = 1 / disp_resized.squeeze().cpu().numpy()

        return depth
