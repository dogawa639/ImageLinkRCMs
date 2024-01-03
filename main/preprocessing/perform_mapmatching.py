if __name__ == "__main__":
    import os
    import json
    import configparser
    import numpy as np
    import pandas as pd

    from preprocessing.network import *
    from preprocessing.pp import *

    CONFIG = "../../config/config_all.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    org_pp_dir = read_data["org_pp_dir"]
    pp_path = json.loads(read_data["pp_path"])

    USEEXIST = False

    nw_data = NetworkCNN(node_path, link_path, mode="ped")

    pp_df_walk = None
    pp_df_car = None
    tid_walk = 0
    tid_car = 0
    # perform map-matching for subtree of pp_data_dir
    for root, dirs, files in os.walk(org_pp_dir):
        if "t_loc_data.csv" in files and "t_locfeeder.csv" in files and "t_trip.csv" in files:
            loc_path = os.path.join(root, "t_loc_data.csv")
            feeder_path = os.path.join(root, "t_locfeeder.csv")
            trip_path = os.path.join(root, "t_trip.csv")
            mode_code_walk = 100
            mode_code_car = {300, 310, 320}
            if not USEEXIST:
                nw_data.set_mode("ped")
                pp_df_walk_tmp = PP.map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code_walk,
                                out_file=os.path.join(root, "path_walk.csv"))
                nw_data.set_mode("car")
                pp_df_car_tmp = PP.map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code_car,
                                out_file=os.path.join(root, "path_car.csv"))
            else:
                pp_df_walk_tmp = pd.read_csv(os.path.join(root, "path_walk.csv"))
                pp_df_car_tmp = pd.read_csv(os.path.join(root, "path_car.csv"))

            pp_df_walk_tmp["ID"] = pp_df_walk_tmp["ID"] + tid_walk
            pp_df_car_tmp["ID"] = pp_df_car_tmp["ID"] + tid_car
            if len(pp_df_walk_tmp) > 0:
                tid_walk = pp_df_walk_tmp["ID"].max()
            if len(pp_df_car_tmp) > 0:
                tid_car = pp_df_car_tmp["ID"].max()

            if pp_df_walk is not None:
                pp_df_walk = pd.concat([pp_df_walk, pp_df_walk_tmp], axis=0)
            else:
                pp_df_walk = pp_df_walk_tmp
            if pp_df_car is not None:
                pp_df_car = pd.concat([pp_df_car, pp_df_car_tmp], axis=0)
            else:
                pp_df_car = pp_df_car_tmp

    pp_df_walk.to_csv(pp_path[0], index=False)
    pp_df_car.to_csv(pp_path[1], index=False)

    nw_data.set_mode("ped")
    pp_data = PP(pp_path[0], nw_data)
    pp_data.write_geo_file(pp_path[0].replace(".csv", ".geojson"), driver="GeoJSON")

    nw_data.set_mode("car")
    pp_data = PP(pp_path[1], nw_data)
    pp_data.write_geo_file(pp_path[1].replace(".csv", ".geojson"), driver="GeoJSON")