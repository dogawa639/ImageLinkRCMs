if __name__ == "__main__":
    import configparser
    import os
    import numpy as np
    import datetime
    import time
    from preprocessing.movie import YoLoV5Detect, ByteTrack

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    camera_data_dir = read_data["camera_datadir"]

    read_save = config["SAVE"]
    movie_dir = read_save["movie_dir"]
    debug_dir = read_save["debug_dir"]

    CLIP = False
    DETECT = False
    TRACK = False
    RESULTMOVIE = False
    RESULTTRAJ = True

    start_time = datetime.datetime.strptime("2022-09-28 22:41:29", "%Y-%m-%d %H:%M:%S")
    img_points = np.array([[700, 520], [1530, 670], [730, 930], [1560, 410]], dtype=np.float32) / np.array(
        [[1920, 1080]], dtype=np.float32)
    xy_points = np.array([[-69266.30906479518, 93476.6843737138], [-69260.76750148823, 93480.99447847917],
                          [-69260.67954013556, 93476.06864445977], [-69268.68402048285, 93492.07760501398]],
                         dtype=np.float32)
    detection = YoLoV5Detect(start_time=start_time, img_points=img_points, xy_points=xy_points, batch_size=1,
                             device="mps")
    if CLIP:
        detection.get_clip(os.path.join(camera_data_dir, "調査地点①/220928/20220928-224129MA.mp4"),
                           os.path.join(movie_dir, "cap_1.png"), 0)
        detection.get_clip(os.path.join(camera_data_dir, "調査地点③/220928/20220928-112855MA.mp4"),
                           os.path.join(movie_dir, "cap_3.png"), 0)
        detection.get_clip(os.path.join(camera_data_dir, "調査地点④/220928/20220928-132455MA.mp4"),
                            os.path.join(movie_dir, "cap_4.png"), 0)
        detection.get_clip(os.path.join(camera_data_dir, "調査地点⑤/220928/20220927-121720MA.mp4"),
                            os.path.join(movie_dir, "cap_5.png"), 0)
    if DETECT:
        t1 = time.perf_counter()
        detection.detect(os.path.join(camera_data_dir, "調査地点①/220928/20220928-224129MA.mp4"), [0], out_csv_dir=movie_dir, out_mov_path=os.path.join(movie_dir, "people.mp4"))
        t2 = time.perf_counter()
        print(f"Detection time: {t2 - t1}")

    if TRACK:
        bytetrack = ByteTrack(3. / 25.)
        bytetrack.track(os.path.join(movie_dir, "yolov_out_0.csv"), os.path.join(movie_dir, "byte_out_0.csv"))

        detection.mapping2xy(os.path.join(movie_dir, "byte_out_0.csv"),
                             os.path.join(movie_dir, "peope_2d_byte.csv"),
                             os.path.join(movie_dir, "peope_topview_byte.mp4"), resolution=0.1)
    if RESULTMOVIE:
        detection.write_detection_result_movie(
            os.path.join(camera_data_dir, "調査地点①/220928/20220928-224129MA.mp4"),
            [os.path.join(movie_dir, "peope_2d_byte.csv")],
            os.path.join(movie_dir, "people_2d_total_byte.mp4"))

    if RESULTTRAJ:
        detection.write_traj_statistics(os.path.join(movie_dir, "peope_2d_byte.csv"), out_csv_path=os.path.join(movie_dir, "people_2d_byte_va.csv"), out_png_path=os.path.join(movie_dir, "people_2d_byte_statistics.png"))
        detection.write_trajectory([os.path.join(movie_dir, "peope_2d_byte.csv")], os.path.join(movie_dir, "people_2d_byte.png"), clean=True)