import argparse
import numpy as np
import os
import struct


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
        fid.write(byte_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--depth_map", help="path to depth map", type=str, required=True
    )
    # parser.add_argument(
    #     "-n", "--normal_map", help="path to normal map", type=str, required=True
    # )
    parser.add_argument(
        "--min_depth_percentile",
        help="minimum visualization depth percentile",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--max_depth_percentile",
        help="maximum visualization depth percentile",
        type=float,
        default=95,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError(
            "min_depth_percentile should be less than or equal "
            "to the max_depth_perceintile."
        )

    # Read depth and normal maps corresponding to the same image.
    if not os.path.exists(args.depth_map):
        raise FileNotFoundError("File not found: {}".format(args.depth_map))

    # if not os.path.exists(args.normal_map):
    #     raise FileNotFoundError("File not found: {}".format(args.normal_map))

    depth_map = read_array(args.depth_map)
    # normal_map = read_array(args.normal_map)

    min_depth, max_depth = np.percentile(
        depth_map, [args.min_depth_percentile, args.max_depth_percentile]
    )
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    import pylab as plt

    # Visualize the depth map.
    plt.figure()
    plt.imshow(depth_map)
    plt.title("depth map")

    # # Visualize the normal map.
    # plt.figure()
    # plt.imshow(normal_map)
    # plt.title("normal map")

    plt.show()


if __name__ == "__main__":
    main()
