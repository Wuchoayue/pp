import numpy as np

def compare(gt_path, ascend_out_path, dtype = np.float32):

    gt_data = np.fromfile(gt_path, dtype)
    ascend_out_data = np.fromfile(ascend_out_path, dtype)

    diff = np.abs(ascend_out_data - gt_data)
    rela_diff = (diff)/(gt_data)
    max_diff_idx = np.argmax(diff)
    eps = 1e-3

    error_count = np.sum(rela_diff>eps)
    error_rate = error_count / (gt_data.shape[0])

    print("error rate: {:.2f}% = {}/{}".format(error_rate*100,error_count,gt_data.shape[0]))

    if error_count == 0:
        print("Compare Success!")
    else:
        print("Compare Failed!")

    print("Max diff: ground truth:{}, ascend output:{}".format(gt_data[max_diff_idx],ascend_out_data[max_diff_idx]))


if __name__ == "__main__":

    for i in range(0, 10):
        print("test {}".format(i))

        gt_path = "./truth/test_celu_shape_gt_{}.bin".format(i)
        ascend_out_path = "./output/output_{}/out_{}.bin".format(i, i)
        compare(gt_path, ascend_out_path)
