if __name__ == "__main__":
    import os
    import json
    import configparser
    from preprocessing.network import *
    from preprocessing.pp import *

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    #nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path, mode="ped")
    nw_data = NetworkCNN(node_path, link_path, mode="ped")

    pp_data_dir = "/Users/dogawa/Desktop/Data/Matsuyama/R4/R4松山ｰ新宿観光PP調査（PP調査データ）/20220907"
    loc_path = os.path.join(pp_data_dir, "t_loc_data.csv")
    feeder_path = os.path.join(pp_data_dir, "t_locfeeder.csv")
    trip_path = os.path.join(pp_data_dir, "t_trip.csv")
    mode_code = 100
    PP.map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code, out_file="/Users/dogawa/Desktop/Git/GANs/debug/data/pp_ped.csv")
    pp_data = PP("/Users/dogawa/Desktop/Git/GANs/debug/data/pp_ped.csv", nw_data)
    pp_data.plot_path(trip_path, loc_path, org_trip_id=222315)

    pp_train_test = pp_data.split_into([0.8, 0.2])
    print(len(pp_data), len(pp_train_test[0]), len(pp_train_test[1]))

    print(pp_data.path_dict[1])
    print(pp_data.tids[1])
    pp_data.write_geo_file("/Users/dogawa/Desktop/Git/GANs/debug/data/pp_ped.geojson", driver="GeoJSON")