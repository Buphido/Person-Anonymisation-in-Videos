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
        image = cv.imread(image)
        image = cv.resize(image, (width, height))
        img_list.append(image[:, :, ::-1])

    images = np.array(img_list)
    images = (images / 255.0 - MEAN) / STD

    input = cv.dnn.blobFromImages(images.astype(np.float32), ddepth=cv.CV_32F)
    return input


def write_frames(path, vid, batch_size, step):
    """
    Write frames into directory as images for easier and faster retrieval
    :param path: path to directory
    :param vid: the input video
    :param batch_size: size of batch
    :param step: step size between frames
    """
    count = 0
    for i in range(0, int(vid.get(cv.CAP_PROP_FRAME_COUNT)), int(batch_size * step)):
        print('Writing for frames into', path, 'Batch:', count, '/',
              int(vid.get(cv.CAP_PROP_FRAME_COUNT) / step))
        for j in range(0, batch_size, 1):
            vid.set(cv.CAP_PROP_POS_FRAMES, step * (count + j))
            ret, frame = vid.read()
            if not ret:
                break
            nmr = int(np.log10(int(vid.get(cv.CAP_PROP_FRAME_COUNT)/step))) - int(np.log10(max(1, count + j)))
            zeros = ''
            for k in range(0, nmr, 1):
                zeros += str(0)
            cv.imwrite(path + '/' + path + zeros + str(count + j) + '.jpg', frame)
        count += batch_size
    print('Writing for frames into', path, 'complete')



def extract_feature(source, vid, model_path, step, batch_size=32, resize_h=384, resize_w=128, backend=cv.dnn.DNN_BACKEND_OPENCV,
                    target=cv.dnn.DNN_TARGET_CPU):
    """
    Extract features from images in a target directory
    :param source: string of input material
    :param vid: the input video
    :param model_path: path to ReID model
    :param step: step size of video
    :param batch_size: the batch size for each network inference iteration
    :param resize_h: the height of the input image
    :param resize_w: the width of the input image
    :param backend: name of computation backend
    :param target: name of computation target
    """
    feat_list = []
    batch = None
    names = []
    count = 0
    exist = True
    if not os.path.isdir(source):
        exist = False
        os.mkdir(source)
    for i in range(0, int(vid.get(cv.CAP_PROP_FRAME_COUNT)), batch_size*step):
        print('Feature Extraction for frames in', source, 'Batch:', count, '/',
              int(vid.get(cv.CAP_PROP_FRAME_COUNT) / step))
        batch = []
        for j in range(0, batch_size, 1):
            frame = None
            if not exist:
                vid.set(cv.CAP_PROP_POS_FRAMES, step * (count + j))
                ret, frame = vid.read()
                if not ret:
                    break
            nmr = int(np.log10(int(vid.get(cv.CAP_PROP_FRAME_COUNT) / step))) - int(np.log10(max(1, count + j)))
            zeros = ''
            for k in range(0, nmr, 1):
                zeros += str(0)
            path = source + '/' + source + zeros + str(count + j) + '.jpg'
            if not exist:
                cv.imwrite(path, frame)
            if not os.path.exists(path):
                break
            batch.append(path)
            names.append(path)
        inputs = preprocess(batch, resize_h, resize_w)

        feat = run_net(inputs, model_path, backend, target)
        feat_list.append(feat)
        count += batch_size

    print('Feature Extraction for frames in', source, 'complete')
    feats = np.concatenate(feat_list, axis=0)
    return feats, names


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


def topk(query_feat, gallery_feat,  topk=5):
    """
    Return the index of top K gallery images most similar to the query frames
    :param query_feat: array of feature vectors of query
    :param gallery_feat: array of feature vectors of gallery
    :param topk: number of gallery images to return
    """
    sim = similarity(query_feat, gallery_feat)
    index = np.argsort(-sim, axis=1)
    return [i[:int(topk)] for i in index]


def drawRankList(query_name, gallery_list, output_size=(128, 384)):
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

    query = cv.imread(query_name)
    query = cv.resize(query, output_size)
    query = addBorder(query, [0, 0, 0])
    cv.putText(query, 'Query', (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)

    gallery_img_list = []
    for i, gallery_name in enumerate(gallery_list):
        gallery_img = cv.imread(gallery_name)
        gallery_img = cv.resize(gallery_img, output_size)
        gallery_img = addBorder(gallery_img, [255, 255, 255])
        cv.putText(gallery_img, 'G%02d' % i, (10, 30), cv.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2)
        gallery_img_list.append(gallery_img)
    ret = np.concatenate([query] + gallery_img_list, axis=1)
    return ret


def visualization(topk_idx, query_names, gallery_names, batch_size, step_query, step_gallery):
    """
    Visualize the retrieval results with the person ReID model
    :param topk_idx: the index of ranked gallery images for each query image
    :param query_names: filepaths of query images
    :param gallery_names: filepaths of gallery images
    :param batch_size: the size of each batch
    :param step_query: step size of query frames
    :param step_gallery: step size of gallery frames
    """
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    writer = None
    for i, idx in enumerate(topk_idx):
        query_name = query_names[i]
        topk_names = [gallery_names[j] for j in idx]
        result = drawRankList(query_name, topk_names)
        query = cv.imread(query_name)
        size = (int(query.shape[1]), int(result.shape[0]*query.shape[1]/result.shape[1]))
        result = cv.resize(result, size, interpolation=cv.INTER_LINEAR)
        result = np.vstack((query, result))
        if writer is None:
            writer = cv.VideoWriter('result.avi', fourcc, vid.get(cv.CAP_PROP_FPS), (result.shape[1], result.shape[0]))
        writer.write(result)
        if i % batch_size == 0:
            print('Visualization of Results for frames in Batch:', i, '/', len(query_names))

    print('Visualization of Results for frames complete')
    writer.release()


if __name__ == '__main__':
    if not os.path.exists('result.avi'):
        parser = argparse.ArgumentParser(description='Use this script to run human parsing using JPPNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--query_dir', '-q', default='query', help='Path to query image.')
        parser.add_argument('--gallery_dir', '-g', default='gallery', help='Path to gallery directory.')
        parser.add_argument('--resize_h', default=256, help='The height of the input for model inference.')
        parser.add_argument('--resize_w', default=128, help='The width of the input for model inference')
        parser.add_argument('--model', '-m', default='reid.onnx', help='Path to pb model.')
        parser.add_argument('--visualization_dir', default='vis', help='Path for the visualization results')
        parser.add_argument('--topk', default=10, help='Number of images visualized in the rank list')
        parser.add_argument('--batchsize', default=64, help='The batch size of each inference')
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

        vid = cv.VideoCapture('../../Downloads/People.mov')

        if not vid.isOpened():
            print("Error opening the video file")

        fps = vid.get(cv.CAP_PROP_FPS)
        print('Frames per second : ', fps, 'FPS')
        frame_count = vid.get(cv.CAP_PROP_FRAME_COUNT)
        print('Frame count : ', frame_count)

        step_query = 2
        time = 0.5
        step_gallery = int(time * fps)
        query_feat, query_names = extract_feature('Query', vid, args.model, step_query, args.batchsize, args.resize_h,
                                                  args.resize_w, args.backend, args.target)
        gallery_feat, gallery_names = extract_feature('Gallery', vid, args.model, step_gallery, args.batchsize,
                                                      args.resize_h, args.resize_w, args.backend, args.target)

        topk_idx = topk(query_feat, gallery_feat, min(args.topk, len(gallery_names)))
        visualization(topk_idx, query_names, gallery_names, args.batchsize, step_query, step_gallery)

    vid_capture = cv.VideoCapture('result.avi')

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
        ret, frame = vid_capture.read()
        if play:
            current += 1
            print(current/fps)
        else:
            vid_capture.set(cv.CAP_PROP_POS_FRAMES, current)
        if ret:
            cv.imshow('Frame', frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv.waitKey(int(2000/fps))
            if key == 2:
                current = max(0, current-fps*1)
                vid_capture.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 44:
                current = max(0, current-1)
                vid_capture.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 3:
                current = min(frame_count, current+fps*1)
                vid_capture.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 46:
                current = min(frame_count, current+1)
                vid_capture.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 32:
                play = not play
            if key == 27:
                break
        else:
            vid_capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            current = 0
            print(current / fps)

    # Release the video capture object
    vid_capture.release()
    cv.destroyAllWindows()