if __name__ == "__main__":
    import configparser
    import os
    import json

    from preprocessing.geo_util import *
    from utility_kalmanfilter import *

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    org_image_dir = read_data["org_image_dir"]
    image_data_path = read_data["image_data_path"]
    image_data_list = load_json(image_data_path)
    onehot_data_path = read_data["onehot_data_path"]

    MAPSEG = False
    GISSEG = True

    if MAPSEG:
        #print(image_data_list[1]["name"])
        #map_path = image_data_list[1]["path"]
        map_path = os.path.join(org_image_dir, "ohsaka_avi_lum.png")

        map_seg = MapSegmentation([map_path], max_class_num=3)
        png_data, one_hot = map_seg.convert_file(map_path)
        print(png_data.shape)
        print(one_hot.shape, one_hot.dtype)
        map_seg.write_colormap()
        map_seg.write_hist()

    if GISSEG:
        onehot_data = load_json(onehot_data_path)
        geo_path = onehot_data[0]["geo_path"]
        gis_seg = GISSegmentation(geo_path, "土地利用CD", 4, 18, [227734, 227760], [104849, 104865])
        print(gis_seg.raster_file)
        gis_seg.write_colormap()
        gis_seg.write_hist()



