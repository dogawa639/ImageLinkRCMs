if __name__ == "__main__":
    import configparser
    import os

    from preprocessing.mesh import MeshNetwork
    from preprocessing.mesh_trajectory import MeshTraj

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    traj_dir = read_data["traj_dir"]

    mnw_data = MeshNetwork((-69280, 93470, -69255, 93495), 25, 25, 3)
    data_list = [os.path.join(traj_dir, "20220928-050000MA", "trajectory_0.csv"), os.path.join(traj_dir, "20220928-050000MA", "trajectory_0.csv")]

    traj_data = MeshTraj(data_list, mnw_data, time_resolution=1.0)

    print(len(traj_data))
    for i in range(3):
        traj_data.show_action(i)

    traj_data.show_actions(0)

