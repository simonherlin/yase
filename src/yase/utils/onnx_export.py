import torch.onnx

def export_model_to_onnx(model, device, input_size=(1, 3, 1024, 512), onnx_file="model.onnx"):
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    torch.onnx.export(model, dummy_input, onnx_file, export_params=True, opset_version=11, do_constant_folding=True)
    return onnx_file
