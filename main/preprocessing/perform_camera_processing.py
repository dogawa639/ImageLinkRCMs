if __name__ == "__main__":
    import sys
    sys.path.append("/home/ogawad/デスクトップ/Code/ImageLinkRCMs")
    import configparser
    import os
    import numpy as np
    import datetime
    from preprocessing.movie import YoLoV5Detect, ByteTrack

    img_points = np.array([[700, 520], [1530, 670], [730, 930], [1560, 410]], dtype=np.float32) / np.array(
        [[1920, 1080]], dtype=np.float32)
    xy_points = np.array([[-69266.30906479518, 93476.6843737138], [-69260.76750148823, 93480.99447847917],
                          [-69260.67954013556, 93476.06864445977], [-69268.68402048285, 93492.07760501398]],
                         dtype=np.float32)
    
    base_dir = "/home/ogawad/デスクトップ/Data/Matsuyama/カメラ調査/調査地点①/220928"
    out_dir = "/home/ogawad/デスクトップ/Code/ImageLinkRCMs/data/movie"

    for cur_dir, dirs, files in os.walk(base_dir):
        for file in files:
            basename, ext = os.path.splitext(file)
            if ext != ".mp4":
                continue
            print("start. ", file)
            tmp_out_dir = os.path.join(cur_dir, basename)
            os.makedirs(tmp_out_dir, exist_ok=True)
            datetime_str = basename[:-2]
            start_time = datetime.datetime.strptime(datetime_str, "%Y%m%d-%H%M%S")
            detection = YoLoV5Detect(start_time=start_time, img_points=img_points, xy_points=xy_points, batch_size=16,device="cuda:0")
            detection.detect(os.path.join(cur_dir, file), [0, 1, 2, 3, 5, 7], out_csv_dir=tmp_out_dir, out_mov_path=os.path.join(out_dir, file))
            print("finish. ", file)