import ffmpeg
import argparse
import os, sys
import json
import cv2
import numpy as np
import time

from subprocess import call

# Rectangle corners starting and engind coorinates

TERMINATE_SEARCH = False

refPt = []
cropping = False


def generate_info_box(video_width, video_heihgt, img):
    ret_img = img
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1
    font_color = (255, 100, 100)
    font_thickness = 1
    j = 0  # line index
    # this text will be printed on the image from each video
    text_to_print = ["Video Resolution: Width: {0} Height: {1}".format(video_width, video_heihgt),
                     "Click on a point and drag to draw a rectangle",
                     "Press r button to see the Region of Interest (ROI) cropped",
                     "Press 'esc' exit or go to the next video in the video dir",
                     "Last selected rectanlge will be saved in a json file"]
    for line in text_to_print:
        text_size = cv2.getTextSize(line, font, 1, font_thickness)[0]
        gap = text_size[1] + 5
        y = int((img.shape[0] + text_size[1]) / 8) + j * gap
        x = 0  # for center alignment => int((img.shape[1] - textsize[0]) / 2)
        j += 1
        ret_img = cv2.putText(img, line, (x, y), font, font_size, font_color, font_thickness, lineType=cv2.LINE_AA)

    return ret_img


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    global last_end_xy
    global last_start_xy

    clone = img.copy()
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", img)

        print("--------------------------------------")
        if len(refPt) == 2:
            # cropping we always need start_point in 1st quadrant and end_point in 3rd quadrant
            # quadrant are define top left rotated anti clockwise
            print("Points Selected:{0}".format(refPt))
            # P1 in 1st quadrant and P2 in 3rd quadrant
            if refPt[0][1] < refPt[1][1] and refPt[0][0] < refPt[1][0]:
                print("P1: Q1 and P2: Q3")
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                last_start_xy = refPt[0]
                last_end_xy = refPt[1]
                isCropAreaSet = True

            # P1 in 2nd quadrant and P2 in 4th quadrant
            # refPt[0][1] > refPt[1][1] and refPt[0][0] < refPt[1][0]
            # refPt[0][1] < refPt[1][1] and refPt[0][0] > refPt[1][0]
            elif refPt[0][1] > refPt[1][1] and refPt[0][0] < refPt[1][0]:
                print("P1: Q2 and P2: Q4")
                roi = clone[refPt[1][1]:refPt[0][1], refPt[0][0]:refPt[1][0]]
                last_start_xy = [(refPt[0][0], refPt[1][1])]
                last_end_xy = [(refPt[1][0], refPt[0][1])]
                isCropAreaSet = True

            # P1 in 3rd quadrant and P2 in 1st quadrant
            elif refPt[0][1] > refPt[1][1] and refPt[0][0] > refPt[1][0]:
                print("P1: Q3 and P2: Q1")
                roi = clone[refPt[1][1]:refPt[0][1], refPt[1][0]:refPt[0][0]]
                last_start_xy = refPt[1]
                last_end_xy = refPt[0]
                isCropAreaSet = True

            # P1 in 4th quadrant and P2 in 2nd quadrant
            elif refPt[0][1] < refPt[1][1] and refPt[0][0] > refPt[1][0]:
                print("P1:Q4 and P2: Q2")
                roi = clone[refPt[0][1]:refPt[1][1], refPt[1][0]:refPt[0][0]]
                last_start_xy = [(refPt[1][0], refPt[0][1])]
                last_end_xy = [(refPt[0][0], refPt[1][1])]
                isCropAreaSet = True
            else:
                print("WARNNING: Points are not making a valid rectanlge")
                isCropAreaSet = False

            cv2.imshow("ROI", roi)

            # Keep only up to two points
            refPt.clear()


def get_cropping_area(video_path, video_width, video_heihgt, nb_frames):
    global img, refPt
    global TERMINATE_SEARCH
    isCropAreaSet = False
    global last_end_xy
    global last_start_xy

    # TODO: optima frame number: search less
    if len(nb_frames) <= 3:
        frame_nb = np.floor(int(nb_frames) // 100 + 1)
    elif len(nb_frames) <= 4:
        frame_nb = np.floor(int(nb_frames) // 100 + 1)
    elif len(nb_frames) > 4:
        frame_nb = np.floor(int(nb_frames) // 100 + 1)

    # Opens the Video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Trouble opening the video ")
    else:
        print("Rendering video frames... ")

    i = 1
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Something went wrong while cap.read() ... ")
            break
        else:
            i += 1

        if i == frame_nb:
            print("Get {0}th_nd frame of the video".format(i))
            cv2.namedWindow(winname='image')
            # Generate information box on the image
            img = generate_info_box(video_width, video_heihgt, img)


            while True:
                print("DEBUG:1 Number of ref Points:{0}".format(len(refPt)))
                cv2.imshow('image', img)
                clone = img.copy()
                cv2.setMouseCallback('image', click_and_crop)

                # wait for a key to be pressed to exit or refresh
                key = cv2.waitKey(0) & 0xff
                #print("DEBUG:2 Number of ref Points:{0}".format(len(refPt)))

                # Exit if ESC pressed
                # if the 'r' key is pressed, reset the cropping region
                # if q or Q is pressed terminate the batch search
                if key == ord("r"):
                    img = clone.copy()
                if key == 27:
                    print("Pressed 'Esc', next video in the video corpus")
                    break
                if key == ord('q') or key == ord('Q'):
                    print("Pressed Q, Force termination")
                    TERMINATE_SEARCH = True
                    break
               # print("DEBUG:3 Number of ref Points:{0}".format(len(refPt)))

            break  # break cap.isOpened() loop

    if last_start_xy and last_end_xy:
        crop_width = int(last_end_xy[0]-last_start_xy[0] )
        crop_height = int(last_end_xy[1]-last_start_xy[1])
        # crop_width = width if last_end_x == -1 else last_end_x
        # x_ = last_start_xy[0] if last_start_xy[0] < last_end_xy[0] else last_end_xy[0]
        # y_ = last_start_xy[1] if last_start_xy[1] < last_end_xy[0] else last_end_xy[1]
        crop_area = {
            'x': last_start_xy[0],
            'y': last_start_xy[1],
            'width': np.abs(crop_width),
            'height': np.abs(crop_height),
        }
    else:
        crop_area = {
            'x': 0,
            'y': 0,
            'width': video_width,
            'height': video_heihgt,
        }

    # clear
    cap.release()
    cv2.destroyAllWindows()
    refPt.clear()
    print("crop_area {}, isCropAreaSet: {}".format(crop_area,isCropAreaSet))
    return isCropAreaSet, crop_area


def create_output_json(dir, video_format, out_dir, json_name):
    global refPt
    out_json = {}
    for file in os.listdir(dir):
        if file.endswith(video_format):

            try:
                # Metadata
                meta_dict = ffmpeg.probe('{0}/{1}'.format(dir, file))
                nb_frames = meta_dict["streams"][0]["nb_frames"]
                width = meta_dict["streams"][0]["width"]
                height = meta_dict["streams"][0]["height"]

                print("video file name: {0}, resolution: {1}x{2}, number of frame: {3} ".
                      format(file, width, height, nb_frames))

                # get cropping data
                isCropAreaSet, crop_area = get_cropping_area('{0}/{1}'.format(dir, file), width, height, nb_frames)

                # This flag is set inside get_cropping_area if the user press q or Q button instead of esc
                # assume this is a force exit therefore not saving the current status
                if TERMINATE_SEARCH:
                    break

                json_array = []
                if crop_area:
                    print("crop-area is selected, inserting this to output json")
                    metadata = {
                        'nb_frames': nb_frames,
                        "isCropAreaSet":isCropAreaSet,
                        'v_width': width,
                        'v_height': height,
                    }
                    json_array.append({"crop_area": crop_area, "metadata": metadata})
                    out_json["{0}".format(file)] = json_array
                else:
                    print("crop area is not selected")

            except ffmpeg.Error as e:
                print("Error: {0}".format(e.stderr), file=sys.stderr)
                sys.exit(1)

    with open("{0}/{1}_{2}.json".format(out_dir, json_name, time.strftime("%Y%m%d-%H%M%S")), 'w') as f:
        json.dump(out_json, f)


# def click_event(event, x, y, flags, params):
#     # checking for left mouse clicks
#     global last_end_x, last_end_y, last_start_y, last_start_x
#     #draw
#     font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     font_color = (255, 0, 0)
#     font_scale = 1
#     font_thick = 1
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # displaying the start rec coordinates
#         print("Starting Point: (x:{0}, y:{1})".format(x,y))
#         last_start_x,last_start_y = x, y
#         cv2.putText(img, str(x) + ',' + str(y), (x, y), font, font_scale, font_color, font_thick, cv2.LINE_AA)
#         cv2.imshow('image', img)
#
#     if event == cv2.EVENT_RBUTTONDOWN:
#         # display the rectangle end crd
#         print("End Point: (x:{0}, y:{1})".format(x,y))
#         cv2.putText(img, str(x) + ',' + str(y), (x, y), font, font_scale, font_color, font_thick, cv2.LINE_AA)
#         cv2.imshow('image', img)
#
#     if (last_end_x != -1) and (last_end_y != -1) and\
#             (last_start_x != -1) and (last_start_y != -1):
#         start_point = (last_start_x, last_start_y)
#         end_point = (last_end_x, last_end_y)
#         color = (255, 0, 0)
#         thickness = 1
#         cv2.rectangle(img, start_point, end_point, color, thickness)
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Access metadeta and creat a json with crop dimenssion')
    parser.add_argument('--dir',
                        help='Path to the input videofiles',
                        required=True)
    parser.add_argument('--out',
                        help='Path for the output json including cropping area',
                        required=True)
    parser.add_argument('--f',
                        help='for example .m4v or .mp4',
                        required=True)
    parser.add_argument('--oname',
                        help='Name of the json output',
                        default="cropped_dimenssion",
                        required=False)

    args = parser.parse_args()

    create_output_json(args.dir, args.f, args.out, args.oname)
