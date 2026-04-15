import sofa
import numpy as np

sofa_path = r"C:\Users\Admin\CIPIC\subject_003.sofa"
hrtf = sofa.Database.open(sofa_path)

source_positions = hrtf.Source.Position.get_values()
data_ir = hrtf.Data.IR.get_values()
sampling_rate = hrtf.Data.SamplingRate.get_values()

print("source_positions shape:", source_positions.shape)
print("data_ir shape:", data_ir.shape)
print("sampling_rate:", sampling_rate)

print("前5个方向坐标：")
print(source_positions[:5])
