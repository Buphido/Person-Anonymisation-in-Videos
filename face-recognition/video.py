import argparse
import os.path
import numpy as np
import cv2 as cv
import pixellib
import tensorflow as tf
import imgaug
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg

from images import Images
from filters import filters
from progress import Progress
from train import set


if __name__ == '__main__':
    #vid = cv.VideoCapture('../../Downloads/People.mov')
    #change_bg = alter_bg(model_type="pb")
    #change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
    if False:
        seg = instance_segmentation()
        seg.load_model("mask_rcnn_coco.h5")
        vid = cv.VideoCapture('airport.mov')

        if not vid.isOpened():
            print("Error opening the video file")

        fps = vid.get(cv.CAP_PROP_FPS)
        print('Frames per second : ', fps, 'FPS')
        frame_count = vid.get(cv.CAP_PROP_FRAME_COUNT)
        print('Frame count : ', frame_count)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
        #out_canny = cv.VideoWriter('canny_box.mp4', fourcc, fps, (width, height))
        out_blur = cv.VideoWriter('blur_shape.mp4', fourcc, fps, (width, height))
        out_blur_bb = cv.VideoWriter('blur_box.mp4', fourcc, fps, (width, height))
        #out_rep = cv.VideoWriter('replace_box.mp4', fourcc, fps, (width, height))
        #out_cust = cv.VideoWriter('custom_box.mp4', fourcc, fps, (width, height))
        set(filters.CUSTOM_BLUR, [3.456302388796793, -8.851734782864286, 7.864385222319630])
        prog = Progress('Processing Video:', n=frame_count)
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            cv.imwrite('frame.jpg', frame)
            #new_canny = filters(Images(rgb=np.copy(frame)), filters.CANNY)
            new_blur = filters(Images(rgb=np.copy(frame)), filters.BLUR)
            #new_rep = filters(Images(rgb=np.copy(frame)), filters.REPLACE)
            #new_cust = filters(Images(rgb=np.copy(frame)), filters.CUSTOM_BLUR)
            target_classes = seg.select_target_classes(person=True)
            a, b = seg.segmentImage('frame.jpg', segment_target_classes=target_classes, show_bboxes=True)
            segments = a['masks']
            locs = a['rois']
            #locs[:, [1, 3]] = locs[:, [3, 1]]
            mask_bb = np.zeros(frame.shape[:2], dtype=bool)
            mask = np.zeros(frame.shape[:2], dtype=bool)
            for loc in locs:
                (y1, x1, y2, x2) = loc
                mask_bb[y1:y2, x1:x2] = True
            for segment in y:
                mask |= segment
            #new_canny = filters(Images(rgb=np.copy(frame)), filters.CANNY, locs=locs)
            #new_blur = filters(Images(rgb=np.copy(frame)), filters.BLUR, locs=locs)
            #new_rep = filters(Images(rgb=np.copy(frame)), filters.REPLACE, locs=locs)
            #new_cust = filters(Images(rgb=np.copy(frame)), filters.CUSTOM_BLUR, locs=locs)
            final_blur_bb = np.copy(frame)
            final_blur = np.copy(frame)
            #final_canny, final_blur, final_rep, final_cust = np.copy(frame), np.copy(frame), np.copy(frame), np.copy(frame)
            #final_canny[mask] = new_canny.rgb[mask]
            final_blur[mask] = new_blur.rgb[mask]
            final_blur_bb[mask_bb] = new_blur.rgb[mask_bb]
            #final_rep[mask] = new_rep.rgb[mask]
            #final_cust[mask] = new_cust.rgb[mask]
            #out_canny.write(final_canny)
            out_blur.write(final_blur)
            out_blur_bb.write(final_blur_bb)
            #out_rep.write(final_rep)
            #out_cust.write(final_cust)

        # Display the cropped frame on the screen
            cv.imshow('frame', np.vstack((final_blur, final_blur_bb)))#np.vstack((np.hstack((final_canny, final_blur)), np.hstack((final_rep, final_cust)))))
            cv.waitKey(1)
            prog.update()
        prog.quit()
        #out_canny.release()
        out_blur.release()
        out_blur_bb.release()
        #out_rep.release()
        #out_cust.release()
        cv.destroyAllWindows()
        cv.waitKey(1)
    #target_classes = seg.select_target_classes(person=True)
    #segmask, vid_capture = seg.process_video("airport.mov", frames_per_second=vid_capture.get(5), segment_target_classes=target_classes,
                      #show_bboxes=True)
    vid = cv.VideoCapture('replace_box.mp4')

    if (vid.isOpened() == False):
        print("Error opening the video file")
    # Read fps and frame count
    else:
        # Get frame rate information
        fps = vid.get(cv.CAP_PROP_FPS)
        print('Frames per second : ', fps, 'FPS')

        # Get frame count
        frame_count = vid.get(cv.CAP_PROP_FRAME_COUNT)
        print('Frame count : ', frame_count)

    play = True
    current = 0
    while vid.isOpened():
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid.read()
        #cv.imwrite('frame.jpg', frame)
        #seg = instance_segmentation()
        #seg.load_model("mask_rcnn_coco.h5")
        #target_classes = seg.select_target_classes(person=True)
        #seg.process_video("airport2.mov", frames_per_second=fps, segment_target_classes=target_classes, show_bboxes=True)
        #segmask, frame = seg.segmentImage("frame.jpg", segment_target_classes=target_classes, show_bboxes=True)
        #print("segmask:" + str(segmask['rois'].shape))

        if play:
            current += 1
            print(current/fps)
        else:
            vid.set(cv.CAP_PROP_POS_FRAMES, current)
            print(current / fps)
        if ret:
            cv.imshow('Frame', frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            #print("pre")
            key = cv.waitKey(1)#int(1000/fps))
            #print("post")
            if key == 2:
                current = max(0, current-fps*1)
                vid.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 44:
                current = max(0, current-1)
                vid.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 3:
                current = min(frame_count, current+fps*1)
                vid.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 46:
                current = min(frame_count, current+1)
                vid.set(cv.CAP_PROP_POS_FRAMES, current)
                print(current / fps)
            if key == 32:
                play = not play
            if key == 27:
                break
        else:
            vid.set(cv.CAP_PROP_POS_FRAMES, 0)
            current = 0
            print(current / fps)

    # Release the video capture object
    vid.release()
    cv.destroyAllWindows()