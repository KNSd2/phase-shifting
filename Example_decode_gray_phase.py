from Phase_Gray_Class import *

width = 1920
height = 1080
step = 32
gamma = 1.0
output_dir = 'output_test'
black_thr = 5
white_thr = 40
filter_size = 0
input_prefix = r"sample_data\object1"
config_file = r"sample_data\object1\config.xml"

ps = PhaseShifting(width, height, step, gamma, output_dir, black_thr, white_thr, filter_size, input_prefix, config_file)
ps.generate()
ps.decode()