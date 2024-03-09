if __name__ == "__main__":
    import os
    import json
    import configparser
    from preprocessing.network import *
    from preprocessing.pp import *
    from utility import *

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]
    org_pp_dir = read_data["org_pp_dir"]
    pp_paths = json.loads(read_data["pp_path"])
    save_data = config["SAVE"]
    debug_dir = save_data["debug_dir"]

    MAPMATCHING = True
    PLOT = True

    #nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path, mode="ped")
    nw_data = NetworkCNN(node_path, link_path, mode="ped")

    pp_data_dir = org_pp_dir
    loc_path = os.path.join(pp_data_dir, "t_loc_data.csv")
    feeder_path = os.path.join(pp_data_dir, "t_locfeeder.csv")
    trip_path = os.path.join(pp_data_dir, "t_trip.csv")

    mode_code_walk = 420
    mode_code_car = {100, 300}
    if MAPMATCHING:
        nw_data.set_mode("ped")
        pp_df_walk_tmp = PP.map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code_walk,
                                         out_file=pp_paths[0])
        nw_data.set_mode("car")
        pp_df_car_tmp = PP.map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code_car,
                                        out_file=pp_paths[1])

    if PLOT:
        nw_data.set_mode("ped")
        pp_data = PP(pp_paths[0], nw_data)
        pp_data.plot_path(trip_path, loc_path, org_trip_id=15384)

        pp_train_test = pp_data.split_into([0.8, 0.2])
        print(len(pp_data), len(pp_train_test[0]), len(pp_train_test[1]))

        print(pp_data.path_dict[1])
        print(pp_data.tids[1])
        pp_data.write_geo_file(os.path.join(debug_dir, "data/pp_ped.geojson"), driver="GeoJSON")