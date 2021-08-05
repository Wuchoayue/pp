import os


if __name__ == "__main__":
    for i in range(0, 10):
        os.system("./hiai_infer_florence -i ./input/input_{}/ -o ./output/output_{}/ -m ./model_shape/test_celu_shape_{}.om".format(i, i, i))
