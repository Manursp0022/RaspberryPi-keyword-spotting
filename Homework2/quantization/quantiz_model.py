import numpy as np
import onnx
import onnxruntime as ort
import os
import random
import torch
from msc_dataset_lab3 import MSCDataset

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    StaticQuantConfig,
    quantize,
)

CLASSES = ['stop','up']
calibration_ds = MSCDataset(root_dir='./data/msc-val', classes=CLASSES, transform=torch.nn.Identity())
frontend_float32_file = f'./model/Group2_frontend.onnx'
model_float32_file = f'./model/Group2_model_float32.onnx'
ort_frontend = ort.InferenceSession(frontend_float32_file)
ort_model = ort.InferenceSession(model_float32_file)

class DataReader(CalibrationDataReader):
    def __init__(self, dataset, frontend_path):
        self.dataset = dataset
        self.frontend_sess = ort.InferenceSession(frontend_path)
        self.enum_data = iter(self.dataset)
        self.frontend_input_name = self.frontend_sess.get_inputs()[0].name

    def get_next(self):
        sample = next(self.enum_data, None)
        if sample is None:
            return None

        inputs = sample['x'].numpy()  
        if inputs.ndim == 2:
            inputs = inputs.squeeze(0)
        inputs = np.expand_dims(inputs, 0) 

        mfcc_features = self.frontend_sess.run(None, {self.frontend_input_name: inputs})[0]

        mfcc_features = np.expand_dims(mfcc_features, 1) 

        return {'input': mfcc_features} 

    def rewind(self):
        self.enum_data = iter(self.dataset)

data_reader = DataReader(calibration_ds,frontend_float32_file)

conf = StaticQuantConfig(
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QDQ,
    calibrate_method=CalibrationMethod.MinMax ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=False,
)

model_int8_file = f'./model/Group2_model_INT8.onnx'
quantize(model_float32_file, model_int8_file, conf)


