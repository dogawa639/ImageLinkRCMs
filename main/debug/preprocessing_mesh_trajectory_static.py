if __name__ == "__main__":
    import configparser
    import os

    from preprocessing.mesh import MeshNetwork
    from preprocessing.mesh_trajectory import MeshTrajStatic

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_geo = config["GEOGRAPHIC"]
    utm_num = int(read_geo["utm_num"])
    read_data = config["DATA"]
    data_dir = read_data["data_dir"]

    GEOFILE = False
    TRAJECTORY = True

    mnw_data = MeshNetwork((-69875.305966, 92588.464115, -66556.048456, 94636.347820), 16, 10, 0)
    if GEOFILE:
        mnw_data.write_geo_file(utm_num, os.path.join(data_dir, "mesh.geojson"))
    if TRAJECTORY:
        data_list = [os.path.join(data_dir, "walk_small.csv")]
        traj_data = MeshTrajStatic(data_list, mnw_data)

        print(traj_data.get_trip_nums())
        print(traj_data.get_aids())
        traj_data.show_traj(0, 1000)

        input("Press Enter to continue...")


