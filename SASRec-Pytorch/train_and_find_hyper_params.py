import os

hidden_units = [32, 64, 128, 256]
dropout_rates = [0.4, 0.5]
num_blocks = [2, 3]

for unit in hidden_units:
    for dropout_rate in dropout_rates:
        for num_block in num_blocks:
            tensorboard_log_dir = "params_search_tensorboard_logs/tensorboard_logs_units=" + str(unit) + "_dropout_rate=" + str(dropout_rate) + "_num_blocks=" + str(num_block)
            os.mkdir(tensorboard_log_dir)
            print("Training by hyperparams: " + "Unit=" + str(unit) + ", DropoutRate=" + str(dropout_rate) + ", NumBlocks=" + str(num_block) + " ...")
            os.system("python train.py" + " --hidden_units=" + str(unit) + " --dropout_rate=" + str(dropout_rate) + " --num_blocks=" + str(num_block) + " --tensorboard_log_dir=" + tensorboard_log_dir)
            print("="*80)