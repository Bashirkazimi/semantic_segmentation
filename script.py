import os


dirnames = ["slrm", "openneg", "openpos", "svf", "ld", "dem"]
main_path = "/notebooks/tmp/data/bmbp_data"  # "/media/kazimi/Data/data/bmbp_data"


for d in dirnames[:-1]:
    dx = "{}/{}/x".format(main_path, d)
    dy = "{}/y".format(main_path)
    dvx = "{}/validation/{}/x".format(main_path, d)
    dvy = "{}/validation/y".format(main_path, d)
    print(dx, dy, dvx, dvy)
    cmd = "python3 train.py --train_image_dir={} --train_label_dir={} --validation_image_dir={} " \
          "--validation_label_dir={} --model_name={}".format(dx, dy, dvx, dvy, d)
    os.system(cmd)

for d in dirnames[-1:]:
    dx = "{}/x".format(main_path)
    dy = "{}/y".format(main_path)
    dvx = "{}/validation/x".format(main_path)
    dvy = "{}/validation/y".format(main_path)
    print(dx, dy, dvx, dvy)
    cmd = "python3 train.py --train_image_dir={} --train_label_dir={} --validation_image_dir={} " \
          "--validation_label_dir={} --model_name={} --preprocess=True".format(dx, dy, dvx, dvy, d)
    os.system(cmd)