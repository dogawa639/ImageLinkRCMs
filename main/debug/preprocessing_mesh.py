if __name__ == "__main__":
    import numpy as np
    from preprocessing.mesh import *

    mesh = MeshNetwork((0, 0, 10, 20), 10, 20, 3)

    mesh.add_prop(np.array([(0, 0), (0, 1), (0, 2)]), np.array([0, 1, 2]))
    mesh.show_props()

    mesh.move_prop(np.array([(0, 0), (0, 1), (0, 2)]), np.array([(1, 0), (1, 1), (1, 2)]), np.array([0, 1, 2]))
    mesh.show_props()

