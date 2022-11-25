#!/usr/bin/env python
'''
You can download a baseline ReID model and sample input from:
https://github.com/ReID-Team/ReID_extra_testdata

Authors of samples and Youtu ReID baseline:
        Xing Sun <winfredsun@tencent.com>
        Feng Zheng <zhengf@sustech.edu.cn>
        Xinyang Jiang <sevjiang@tencent.com>
        Fufu Yu <fufuyu@tencent.com>
        Enwei Zhang <miyozhang@tencent.com>

Copyright (C) 2020-2021, Tencent.
Copyright (C) 2020-2021, SUSTech.
'''
import argparse
import os.path
import numpy as np
import cv2 as cv

backends = (cv.dnn.DNN_BACKEND_DEFAULT,
            cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
            cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_BACKEND_VKCOM,
            cv.dnn.DNN_BACKEND_CUDA)

targets = (cv.dnn.DNN_TARGET_CPU,
           cv.dnn.DNN_TARGET_OPENCL,
           cv.dnn.DNN_TARGET_OPENCL_FP16,
           cv.dnn.DNN_TARGET_MYRIAD,
           cv.dnn.DNN_TARGET_HDDL,
           cv.dnn.DNN_TARGET_VULKAN,
           cv.dnn.DNN_TARGET_CUDA,
           cv.dnn.DNN_TARGET_CUDA_FP16)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess(images, height, width):
    """
    Create 4-dimensional blob from image
    :param images: input image
    :param height: the height of the resized input image
    :param width: the width of the resized input image
    """
    img_list = []
    for image in images:
        image = cv.resize(image, (width, height))
        img_list.append(image[:, :, ::-1])

    images = np.array(img_list)
    images = (images / 255.0 - MEAN) / STD

    input = cv.dnn.blobFromImages(images.astype(np.float32), ddepth=cv.CV_32F)
    return input


def extract_feature(img_dir, model_path, batch_size=32, resize_h=384, resize_w=128, backend=cv.dnn.DNN_BACKEND_OPENCV,
                    target=cv.dnn.DNN_TARGET_CPU):
    """
    Extract features from images in a target directory
    :param img_dir: the input image directory
    :param model_path: path to ReID model
    :param batch_size: the batch size for each network inference iteration
    :param resize_h: the height of the input image
    :param resize_w: the width of the input image
    :param backend: name of computation backend
    :param target: name of computation target
    """
    feat_list = []
    path_list = os.listdir(img_dir)
    path_list = [os.path.join(img_dir, img_name) for img_name in path_list]
    count = 0

    for i in range(0, len(path_list), batch_size):
        print('Feature Extraction for images in', img_dir, 'Batch:', count, '/', len(path_list))
        batch = path_list[i: min(i + batch_size, len(path_list))]
        imgs = read_data(batch)
        inputs = preprocess(imgs, resize_h, resize_w)

        feat = run_net(inputs, model_path, backend, target)

        feat_list.append(feat)
        count += batch_size

    print('Feature Extraction for images in', img_dir, 'complete')
    feats = np.concatenate(feat_list, axis=0)
    return feats, path_list


def run_net(inputs, model_path, backend=cv.dnn.DNN_BACKEND_OPENCV, target=cv.dnn.DNN_TARGET_CPU):
    """
    Forword propagation for a batch of images.
    :param inputs: input batch of images
    :param model_path: path to ReID model
    :param backend: name of computation backend
    :param target: name of computation target
    """
    net = cv.dnn.readNet(model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    net.setInput(inputs)
    out = net.forward()
    out = np.reshape(out, (out.shape[0], out.shape[1]))
    return out


def read_data(path_list):
    """
    Read all images from a directory into a list
    :param path_list: the list of image path
    """
    img_list = []
    for img_path in path_list:
        img = cv.imread(img_path)
        if img is None:
            continue
        img_list.append(img)
    return img_list


def normalize(nparray, order=2, axis=0):
    """
    Normalize a N-D numpy array along the specified axis.
    :param nparray: the array of vectors to be normalized
    :param order: order of the norm
    :param axis: the axis of x along which to compute the vector norms
    """
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def similarity(array1, array2):
    """
    Compute the euclidean or cosine distance of all pairs.
    :param  array1: numpy array with shape [m1, n]
    :param  array2: numpy array with shape [m2, n]
    Returns:
      numpy array with shape [m1, m2]
    """
    array1 = normalize(array1, axis=1)
    array2 = normalize(array2, axis=1)
    dist = np.matmul(array1, array2.T)
    return dist


def topk(query_feat, gallery_feat, topk=5):
    """
    Return the index of top K gallery images most similar to the query images
    :param query_feat: array of feature vectors of query images
    :param gallery_feat: array of feature vectors of gallery images
    :param topk: number of gallery images to return
    """
    sim = similarity(query_feat, gallery_feat)
    index = np.argsort(-sim, axis=1)
    return [i[0:int(topk)] for i in index]


def drawRankList(query_name, gallery_list, output_size=(128, 384)):
    """
    Draw the rank list
    :param query_name: path of the query image
    :param gallery_list: path of the gallery image
    :param output_size: the output size of each image in the rank list
    """

    def addBorder(im, color):
        bordersize = 5
        border = cv.copyMakeBorder(
            im,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv.BORDER_CONSTANT,
            value=color
        )
        return border

    query_img = cv.imread(query_name)
    query_img = cv.resize(query_img, output_size)
    query_img = addBorder(query_img, [0, 0, 0])
    cv.putText(query_img, 'Query', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)

    gallery_img_list = []
    for i, gallery_name in enumerate(gallery_list):
        gallery_img = cv.imread(gallery_name)
        gallery_img = cv.resize(gallery_img, output_size)
        gallery_img = addBorder(gallery_img, [255, 255, 255])
        cv.putText(gallery_img, 'G%02d' % i, (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)
        gallery_img_list.append(gallery_img)
    ret = np.concatenate([query_img] + gallery_img_list, axis=1)
    return ret


def visualization(topk_idx, query_names, gallery_names, output_dir='vis'):
    """
    Visualize the retrieval results with the person ReID model
    :param topk_idx: the index of ranked gallery images for each query image
    :param query_names: the list of paths of query images
    :param gallery_names: the list of paths of gallery images
    :param output_dir: the path to save the visualize results
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i, idx in enumerate(topk_idx):
        query_name = query_names[i]
        topk_names = [gallery_names[j] for j in idx]
        vis_img = drawRankList(query_name, topk_names)
        output_path = os.path.join(output_dir, '%03d_%s' % (i, os.path.basename(query_name)))
        cv.imwrite(output_path, vis_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to run human parsing using JPPNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--query_dir', '-q', default='query', help='Path to query image.')
    parser.add_argument('--gallery_dir', '-g', default='gallery', help='Path to gallery directory.')
    parser.add_argument('--resize_h', default=256, help='The height of the input for model inference.')
    parser.add_argument('--resize_w', default=128, help='The width of the input for model inference')
    parser.add_argument('--model', '-m', default='reid.onnx', help='Path to pb model.')
    parser.add_argument('--visualization_dir', default='vis', help='Path for the visualization results')
    parser.add_argument('--topk', default=10, help='Number of images visualized in the rank list')
    parser.add_argument('--batchsize', default=32, help='The batch size of each inference')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA backend" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: NCS2 VPU, '
                             '%d: HDDL VPU, '
                             '%d: Vulkan, '
                             '%d: CUDA, '
                             '%d: CUDA FP16'
                             % targets)
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model):
        raise OSError("Model not exist")

    query_feat, query_names = extract_feature(args.query_dir, args.model, args.batchsize, args.resize_h, args.resize_w,
                                              args.backend, args.target)
    gallery_feat, gallery_names = extract_feature(args.gallery_dir, args.model, args.batchsize, args.resize_h,
                                                  args.resize_w, args.backend, args.target)

    topk_idx = topk(query_feat, gallery_feat, min(args.topk, len(gallery_names)))
    visualization(topk_idx, query_names, gallery_names, output_dir=args.visualization_dir)

import cv2

#img = cv2.imread("../../Pictures/Madeline.png", cv2.IMREAD_COLOR)
#cv2.imshow("Madeline", img)
#cv2.waitKey(0)
#cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
#if not cap.isOpened():
#    raise IOError("Cannot open webcam")

#while True:
#    ret, frame = cap.read()
#    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#    cv2.imshow('Input', frame)

#    c = cv2.waitKey(1)
#    if c == 27:
#        break

vid_capture = cv2.VideoCapture('../../Downloads/Biden.mov')

if (vid_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(5)
    print('Frames per second : ', fps, 'FPS')

    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_count = vid_capture.get(7)
    print('Frame count : ', frame_count)

play = True
current = 0
while vid_capture.isOpened():
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    if play:
        ret, frame = vid_capture.read()
        current += 1
        print(current/fps)
    if ret:
        cv2.imshow('Frame', frame)
        # 20 is in milliseconds, try to increase the value, say 50 and observe
        key = cv2.waitKey(20)
        if key == 2:
            current = max(0, current-fps*5)
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, current)
        if key == 3:
            current = min(frame_count, current+fps*5)
            vid_capture.set(cv2.CAP_PROP_POS_FRAMES, current)
        if key == 32:
            play = not play
        if key == 27:
            break
    else:
        break

# Release the video capture object
vid_capture.release()

#cap.release()
cv2.destroyAllWindows()

#!/usr/bin/env python
'''
You can download a baseline ReID model and sample input from:
https://github.com/ReID-Team/ReID_extra_testdata

Authors of samples and Youtu ReID baseline:
        Xing Sun <winfredsun@tencent.com>
        Feng Zheng <zhengf@sustech.edu.cn>
        Xinyang Jiang <sevjiang@tencent.com>
        Fufu Yu <fufuyu@tencent.com>
        Enwei Zhang <miyozhang@tencent.com>

Copyright (C) 2020-2021, Tencent.
Copyright (C) 2020-2021, SUSTech.
'''
import argparse
import os.path
import numpy as np
import cv2 as cv

backends = (cv.dnn.DNN_BACKEND_DEFAULT,
            cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
            cv.dnn.DNN_BACKEND_OPENCV,
            cv.dnn.DNN_BACKEND_VKCOM,
            cv.dnn.DNN_BACKEND_CUDA)

targets = (cv.dnn.DNN_TARGET_CPU,
           cv.dnn.DNN_TARGET_OPENCL,
           cv.dnn.DNN_TARGET_OPENCL_FP16,
           cv.dnn.DNN_TARGET_MYRIAD,
           cv.dnn.DNN_TARGET_HDDL,
           cv.dnn.DNN_TARGET_VULKAN,
           cv.dnn.DNN_TARGET_CUDA,
           cv.dnn.DNN_TARGET_CUDA_FP16)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def preprocess(images, height, width):
    """
    Create 4-dimensional blob from image
    :param images: input image
    :param height: the height of the resized input image
    :param width: the width of the resized input image
    """
    img_list = []
    for image in images:
        image = cv.resize(image, (width, height))
        img_list.append(image[:, :, ::-1])

    images = np.array(img_list)
    images = (images / 255.0 - MEAN) / STD

    input = cv.dnn.blobFromImages(images.astype(np.float32), ddepth=cv.CV_32F)
    return input


def extract_feature(vid, model_path, batch_size=32, resize_h=384, resize_w=128, backend=cv.dnn.DNN_BACKEND_OPENCV,
                    target=cv.dnn.DNN_TARGET_CPU):
    """
    Extract features from images in a target directory
    :param vid: the input video
    :param model_path: path to ReID model
    :param batch_size: the batch size for each network inference iteration
    :param resize_h: the height of the input image
    :param resize_w: the width of the input image
    :param backend: name of computation backend
    :param target: name of computation target
    """
    feat_list = []
    count = 0

    for i in range(0, int(vid.get(cv.CAP_PROP_FRAME_COUNT)), batch_size):
        print('Feature Extraction for frames in Batch:', count, '/', vid.get(cv.CAP_PROP_FRAME_COUNT))
        batch = []
        for j in range(0, batch_size, 1):
            ret, frame = vid.read()
            if not ret:
                break
            batch.append(frame)
        inputs = preprocess(batch, resize_h, resize_w)

        feat = run_net(inputs, model_path, backend, target)

        feat_list.append(feat)
        count += batch_size

    print('Feature Extraction for frames complete')
    feats = np.concatenate(feat_list, axis=0)
    return feats


def run_net(inputs, model_path, backend=cv.dnn.DNN_BACKEND_OPENCV, target=cv.dnn.DNN_TARGET_CPU):
    """
    Forword propagation for a batch of images.
    :param inputs: input batch of images
    :param model_path: path to ReID model
    :param backend: name of computation backend
    :param target: name of computation target
    """
    net = cv.dnn.readNet(model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    net.setInput(inputs)
    out = net.forward()
    out = np.reshape(out, (out.shape[0], out.shape[1]))
    return out


def read_data(path_list):
    """
    Read all images from a directory into a list
    :param path_list: the list of image path
    """
    img_list = []
    for img_path in path_list:
        img = cv.imread(img_path)
        if img is None:
            continue
        img_list.append(img)
    return img_list


def normalize(nparray, order=2, axis=0):
    """
    Normalize a N-D numpy array along the specified axis.
    :param nparray: the array of vectors to be normalized
    :param order: order of the norm
    :param axis: the axis of x along which to compute the vector norms
    """
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def similarity(array):
    """
    Compute the euclidean or cosine distance of all pairs.
    :param  array1: numpy array with shape [m1, n]
    :param  array2: numpy array with shape [m2, n]
    Returns:
      numpy array with shape [m1, m2]
    """
    array = normalize(array, axis=1)
    dist = np.matmul(array, array.T)
    return dist


def topk(frame_feat, topk=5):
    """
    Return the index of top K gallery images most similar to the query frames
    :param frame_feat: array of feature vectors of frames
    :param topk: number of gallery images to return
    """
    sim = similarity(frame_feat)
    index = np.argsort(-sim, axis=1)
    return [i[0:int(topk)] for i in index]


def drawRankList(query, gallery, output_size=(128, 384)):
    """
    Draw the rank list
    :param query: query image
    :param gallery: gallery images
    :param output_size: the output size of each image in the rank list
    """

    def addBorder(im, color):
        bordersize = 5
        border = cv.copyMakeBorder(
            im,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv.BORDER_CONSTANT,
            value=color
        )
        return border

    query = cv.resize(query, output_size)
    query = addBorder(query, [0, 0, 0])
    cv.putText(query, 'Query', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)

    gallery_img_list = []
    for i, gallery_img in enumerate(gallery):
        gallery_img = cv.resize(gallery_img, output_size)
        gallery_img = addBorder(gallery_img, [255, 255, 255])
        cv.putText(gallery_img, 'G%02d' % i, (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)
        gallery_img_list.append(gallery_img)
    ret = np.concatenate([query] + gallery_img_list, axis=1)
    return ret


def visualization(topk_idx, vid, batch_size):
    """
    Visualize the retrieval results with the person ReID model
    :param topk_idx: the index of ranked gallery images for each query image
    :param vid: the input video
    :param batch_size: the size of each batch
    """
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    writer = None
    for i, idx in enumerate(topk_idx):
        vid.set(cv.CAP_PROP_POS_FRAMES, i)
        _, query = vid.read()
        topk_frames = []
        for j in idx:
            vid.set(cv.CAP_PROP_POS_FRAMES, j)
            _, frame = vid.read()
            topk_frames.append(frame)
        result = drawRankList(query, topk_frames)
        size = (int(query.shape[1]), int(result.shape[0]*query.shape[1]/result.shape[1]))
        result = cv.resize(result, size, interpolation=cv.INTER_LINEAR)
        result = np.vstack((query, result))
        if writer is None:
            writer = cv.VideoWriter('result.avi', fourcc, vid.get(cv.CAP_PROP_FPS), (result.shape[1], result.shape[0]))
        writer.write(result)
        if i % batch_size == 0:
            print('Visualization of Results for frames in Batch:', i, '/', vid.get(cv.CAP_PROP_FRAME_COUNT))

    print('Visualization of Results for frames complete')
    writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use this script to run human parsing using JPPNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--query_dir', '-q', default='query', help='Path to query image.')
    parser.add_argument('--gallery_dir', '-g', default='gallery', help='Path to gallery directory.')
    parser.add_argument('--resize_h', default=256, help='The height of the input for model inference.')
    parser.add_argument('--resize_w', default=128, help='The width of the input for model inference')
    parser.add_argument('--model', '-m', default='reid.onnx', help='Path to pb model.')
    parser.add_argument('--visualization_dir', default='vis', help='Path for the visualization results')
    parser.add_argument('--topk', default=10, help='Number of images visualized in the rank list')
    parser.add_argument('--batchsize', default=32, help='The batch size of each inference')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA backend" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: NCS2 VPU, '
                             '%d: HDDL VPU, '
                             '%d: Vulkan, '
                             '%d: CUDA, '
                             '%d: CUDA FP16'
                             % targets)
    args, _ = parser.parse_known_args()

    if not os.path.isfile(args.model):
        raise OSError("Model not exist")

    vid = cv.VideoCapture('../../Downloads/Biden.mov')

    if not vid.isOpened():
        print("Error opening the video file")

    fps = vid.get(cv.CAP_PROP_FPS)
    print('Frames per second : ', fps, 'FPS')
    frame_count = vid.get(cv.CAP_PROP_FRAME_COUNT)
    print('Frame count : ', frame_count)

    play = True
    current = 0

    frame_feat = extract_feature(vid, args.model, args.batchsize, args.resize_h, args.resize_w,
                                              args.backend, args.target)
    #gallery_feat, gallery_names = extract_feature(args.gallery_dir, args.model, args.batchsize, args.resize_h,
                                                  #args.resize_w, args.backend, args.target)

    topk_idx = topk(frame_feat, min(args.topk, frame_count))
    visualization(topk_idx, vid, args.batchsize)
