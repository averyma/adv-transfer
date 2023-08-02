import torch, sys, os, time

def checkpoint_save(state_dict, save_dir, checkpoint_name):
    checkpoint_path = os.path.join(save_dir, checkpoint_name)
    t0 = time.perf_counter()
    checkpoint = torch.save(state_dict, checkpoint_path)
    t1 = time.perf_counter()
    save_time = t1 - t0
    statinfo = os.stat(checkpoint_path)
    cmd = "echo \"{}\n{}\" > {}.stats".format(statinfo.st_size, save_time, checkpoint_path)
    os.system(cmd)
