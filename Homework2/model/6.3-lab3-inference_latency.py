from subprocess import Popen
from time import sleep

import numpy as np
import onnxruntime as ort
import pandas as pd

# Fix the CPU frequency to its maximum value (1.5 GHz)
Popen(
    'sudo sh -c "echo performance >'
    '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
    shell=True,
).wait()

x_test = np.random.normal(size=(1, 1, 16000)).astype(np.float32)

# Change the model name
MODEL_NAME = 'mobilenet_96'

frontend_file = f'{MODEL_NAME}_frontend.onnx'
model_file = f'{MODEL_NAME}_model_INT8.onnx'

sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 1
sess_opt.inter_op_num_threads = 1
sess_opt.enable_profiling = True

ort_frontend = ort.InferenceSession(frontend_file, sess_options=sess_opt)
# ORT profile file names use the timestamp.
# Sleep 1 minute to generate two different file names.
sleep(60)
ort_model = ort.InferenceSession(model_file, sess_options=sess_opt)

tot_latencies = []
for i in range(100):
    features = ort_frontend.run(None, {'input': x_test})[0]
    outputs = ort_model.run(None, {'input': features})[0]
    sleep(0.1)

frontend_profile = ort_frontend.end_profiling()
model_profile = ort_model.end_profiling()


def print_stats(profile_file):
    df = pd.read_json(profile_file)
    df_filtered = df[['name', 'dur']]
    name_order = df_filtered.drop_duplicates('name')['name']

    # Group by 'name' and calculate median/std 'dur'
    stats = (
        df_filtered.groupby('name')['dur']
        .agg(['median', 'std', 'min', 'max'])
        .reset_index()
    )
    stats['name'] = pd.Categorical(
        stats['name'], categories=name_order, ordered=True
    )
    stats = stats.sort_values('name')

    print(stats)


print('Frontend stats:')
print_stats(frontend_profile)
print()
print('Model stats:')
print_stats(model_profile)
