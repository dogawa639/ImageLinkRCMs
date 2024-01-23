if __name__ == "__main__":
    import sys
    import configparser
    import os
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    from preprocessing.movie import YoLoV5Detect, ByteTrack

    CONFIG = "../../config/config_all.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    camera_data_dir = read_data["camera_datadir"]

    read_save = config["SAVE"]
    movie_dir = read_save["movie_dir"]

    YOLO = False
    BYTE = True
    CLEANING = True

    img_points = [
        np.array([[700, 520], [1530, 670], [730, 930], [1560, 410]], dtype=np.float32) / np.array(
        [[1920, 1080]], dtype=np.float32),
        np.array([[560, 920], [505, 495], [890, 450], [1430, 710]], dtype=np.float32) / np.array(
                [[1920, 1080]], dtype=np.float32),
        np.array([[1860, 625], [1560, 370], [670, 535], [1070, 335]], dtype=np.float32) / np.array(
            [[1920, 1080]], dtype=np.float32)
                  ]
    xy_points = [np.array([[-69266.30906479518, 93476.6843737138], [-69260.76750148823, 93480.99447847917],
                          [-69260.67954013556, 93476.06864445977], [-69268.68402048285, 93492.07760501398]],
                         dtype=np.float32),
                 np.array([[-69262.40005648049, 93411.84591414176], [-69253.18587790198, 93411.93366822123],
                           [-69253.09812377447, 93407.01943966746], [-69262.40005640057, 93406.49291518782]],
                          dtype=np.float32),
                 np.array([[-69262.40005648049, 93411.84591414176], [-69262.70432696717, 93427.99825516953],
                           [-69271.42596356821, 93411.48528991156], [-69271.77482905846, 93427.9982551752]],
                          dtype=np.float32)
                 ]
    idxs = [0, 1, 2, 3, 5, 7]

    if YOLO:
        for i, rel_dir in enumerate(["調査地点①", "調査地点④", "調査地点⑤"]):
            tmp_dir = os.path.join(camera_data_dir, rel_dir)
            for cur_dir, dirs, files in os.walk(tmp_dir):
                for file in files:
                    tmp_path = os.path.join(cur_dir, file)
                    if os.path.splitext(tmp_path)[1] == ".mp4":
                        print("start. ", tmp_path)
                        basename, ext = os.path.splitext(file)
                        tmp_out_dir = os.path.join(movie_dir, rel_dir, basename)
                        os.makedirs(tmp_out_dir, exist_ok=True)
                        datetime_str = basename[:-2]
                        start_time = datetime.datetime.strptime(datetime_str, "%Y%m%d-%H%M%S")
                        detection = YoLoV5Detect(start_time=start_time, img_points=img_points[i], xy_points=xy_points[i], batch_size=1, device="mps")
                        detection.detect(tmp_path, idxs, out_csv_dir=tmp_out_dir)
                        print("finish. ", tmp_path)
        print("Yolo finish.")

    if BYTE:
        for i, rel_dir in enumerate(["調査地点1", "調査地点4", "調査地点5"]):
            if i == 0 or i == 2:
                continue
            tmp_dir = os.path.join(movie_dir, rel_dir)
            for cur_dir, dirs, files in os.walk(tmp_dir):
                for file in files:
                    tmp_path = os.path.join(cur_dir, file)
                    if os.path.splitext(tmp_path)[1] == ".csv" and "yolov_out" in tmp_path:
                        print("start. ", tmp_path)
                        basename, ext = os.path.splitext(file)
                        dir_name = os.path.basename(cur_dir)
                        datetime_str = dir_name[:-2]
                        start_time = datetime.datetime.strptime(datetime_str, "%Y%m%d-%H%M%S")
                        byte_path = tmp_path.replace("yolov", "byte")
                        bytetrack = ByteTrack(3. / 25.)
                        bytetrack.track(tmp_path, byte_path)
                        detection = YoLoV5Detect(start_time=start_time, img_points=img_points[i], xy_points=xy_points[i], batch_size=1)
                        detection.mapping2xy(byte_path, byte_path.replace("byte", "2d_byte"), resolution=0.1)
                        print("finish. ", tmp_path)
        print("Byte finish.")

    if CLEANING:
        # target: 2d_byte_out_{num}.csv
        detection = YoLoV5Detect()
        cmap = plt.get_cmap("hsv")
        colors = [tuple((np.array(cmap(i / len(idxs))[2::-1])).astype(float)) for i in
                  range(len(idxs))]
        # remove previous files
        rm_dirs = []
        rm_files = []
        for i, rel_dir in enumerate(["調査地点1", "調査地点4", "調査地点5"]):
            if i == 0 or i == 2:
                continue
            tmp_dir = os.path.join(movie_dir, rel_dir)
            for cur_dir, dirs, files in os.walk(tmp_dir):
                for tmp in dirs:
                    if "cleaned" in tmp:
                        rm_dirs.append(os.path.join(cur_dir, tmp))
                for file in files:
                    if "cleaned" in file:
                        rm_files.append(os.path.join(cur_dir, file))
        for tmp in rm_dirs:
            shutil.rmtree(tmp)
        for tmp in rm_files:
            os.remove(tmp)

        for i, rel_dir in enumerate(["調査地点1", "調査地点4", "調査地点5"]):
            if i == 0 or i == 2:
                continue
            tmp_dir = os.path.join(movie_dir, rel_dir)
            for cur_dir, dirs, files in os.walk(tmp_dir):
                for file in files:
                    tmp_path = os.path.join(cur_dir, file)
                    if os.path.splitext(tmp_path)[1] == ".csv" and "2d_byte_out" in tmp_path:
                        print("start. ", tmp_path)
                        basename, ext = os.path.splitext(file)
                        dir_name = os.path.basename(cur_dir)
                        datetime_str = dir_name[:-2]
                        start_time = datetime.datetime.strptime(datetime_str, "%Y%m%d-%H%M%S")
                        cleaned_path = tmp_path.replace("2d_byte_out", "2d_byte_out_cleaned")
                        two_d_path = tmp_path.replace("2d_byte_out", "2d")
                        cleaned = detection.clean_trajectory(tmp_path, cleaned_path)
                        detection.write_traj_statistics(cleaned, out_csv_path=two_d_path, out_png_path=two_d_path.replace("csv", "png"))
                        print("finish. ", tmp_path)
                print(f"write trajctories for {cur_dir}.")
                obj_dfs = []
                c = []
                for j, idx in enumerate(idxs):
                    if f"2d_byte_out_{idx}.csv" in files:
                        obj_dfs.append(os.path.join(cur_dir, f"2d_byte_out_cleaned_{idx}.csv"))
                        c.append(colors[j])
                if len(obj_dfs) > 0:
                    detection.write_trajectory(obj_dfs, os.path.join(cur_dir, "2d_trajectory.png"), colors=c, clean=False)
