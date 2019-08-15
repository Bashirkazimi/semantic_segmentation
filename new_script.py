import os


dirnames = ["dem", "slrm", "openneg", "openpos", "svf", "ld"]
main_path = "/notebooks/tmp/data/bmbp_data"  # "/media/kazimi/Data/data/bmbp_data"


for d in dirnames[:-1]:
    dx = "{}/{}/x4".format(main_path, d)
    dy = "{}/y4".format(main_path)
    dvx = "{}/validation/{}/x4".format(main_path, d)
    dvy = "{}/validation/y4".format(main_path, d)
    print(dx, dy, dvx, dvy)
    if d == "dem":
        cmd = "python3 train.py --train_image_dir={} --train_label_dir={} --validation_image_dir={} " \
              "--validation_label_dir={} --model_name={} --preprocess=True --wildcard=*.npy".format(dx, dy, dvx, dvy, d)
    else:
        cmd = "python3 train.py --train_image_dir={} --train_label_dir={} --validation_image_dir={} " \
              "--validation_label_dir={} --model_name={} --wildcard=*npy".format(dx, dy, dvx, dvy, d)
    os.system(cmd)
