import ffmpeg
import argparse
import os, sys
import json
import cv2
import numpy as np
import time
import logging
import datetime

TERMINATE_SEARCH = False

refPt = []
cropping = False
last_start_xy = 0
last_end_xy = 0


def generate_info_box(video_width, video_heihgt, img):
    ret_img = img
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1
    font_color = (255, 100, 100)
    font_thickness = 1
    j = 0  # line index
    # this text will be printed on the image from each video
    text_to_print = ["Video Resolution: Width: {0} Height: {1}".format(video_width, video_heihgt),
                     "---------------------------------------",
                     "Click on a point and drag to draw a rectangle",
                     "Press 'R': Refresh the image and ROI",
                     "Press 'ESC': go to next video",
                     "Press Q: Force termination, end directory search",
                     "Last selected rectanlge will be saved in a json file"]
    for line in text_to_print:
        text_size = cv2.getTextSize(line, font, 1, font_thickness)[0]
        gap = text_size[1] + 5
        y = int((img.shape[0] + text_size[1]) / 8) + j * gap
        x = 0  # for center alignment => int((img.shape[1] - textsize[0]) / 2)
        j += 1
        ret_img = cv2.putText(img, line, (x, y), font, font_size, font_color, font_thickness, lineType=cv2.LINE_AA)

    return ret_img


def get_cropping_area(video_path: str, video_width: str, video_heihgt: str, nb_frames: str, output_path: str) -> object:
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
        logging.debug("Trouble opening the video ")
    else:
        logging.debug("Video successfully opened ")

    i = 1
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            logging.debug("Something went wrong while cap.read() ... ")
            break
        else:
            i += 1

        if i == frame_nb:
            logging.debug("Frame number: {0}".format(i))
            cv2.namedWindow(winname='image')
            # Generate information box on the image
            img = generate_info_box(video_width, video_heihgt, img)

            while True:
                # print("DEBUG:1 Number of ref Points:{0}".format(len(refPt)))
                cv2.imshow('image', img)
                clone = img.copy()
                cv2.setMouseCallback('image', click_and_crop)

                # wait for a key to be pressed to exit or refresh
                key = cv2.waitKey(0) & 0xff
                # print("DEBUG:2 Number of ref Points:{0}".format(len(refPt)))

                # Exit if ESC pressed
                # if the 'r' key is pressed, reset the cropping region
                # if q or Q is pressed terminate the batch search
                if key == ord("r"):
                    logging.info("Pressed R: refreshing image")
                    print("Pressed R: refreshing image")
                    img = clone.copy()
                if key == 27:
                    logging.info("Pressed 'Esc': Next video in the video corpus or finish")
                    print("Pressed 'Esc': Next video in the video corpus or finish")
                    crop_image= '{0}/{1}.jpg'.format(output_path, os.path.basename(video_path))
                    cv2.imwrite(crop_image, img)
                    break
                if key == ord('q') or key == ord('Q'):
                    logging.info("Pressed Q: Force termination, end directory search")
                    print("Pressed Q: Force termination, end directory search")
                    TERMINATE_SEARCH = True
                    break
            # print("DEBUG:3 Number of ref Points:{0}".format(len(refPt)))

            break  # break cap.isOpened() loop

    if last_start_xy and last_end_xy:
        isCropAreaSet = True
        crop_width = int(np.abs(last_end_xy[0] - last_start_xy[0]))
        crop_height = int(np.abs(last_end_xy[1] - last_start_xy[1]))
        # crop_width = width if last_end_x == -1 else last_end_x
        # x_ = last_start_xy[0] if last_start_xy[0] < last_end_xy[0] else last_end_xy[0]
        # y_ = last_start_xy[1] if last_start_xy[1] < last_end_xy[0] else last_end_xy[1]
        crop_area = {
            'x': last_start_xy[0],
            'y': last_start_xy[1],
            'width': crop_width,
            'height': crop_height,
        }
    else:
        isCropAreaSet = False
        crop_area = {
            'x': 0,
            'y': 0,
            'width': 0,
            'height': 0,
        }

    # clear
    cap.release()
    cv2.destroyAllWindows()
    refPt.clear()
    logging.info("crop_area {}, isCropAreaSet: {}".format(crop_area, isCropAreaSet))
    print("crop_area {}, isCropAreaSet: {}".format(crop_area, isCropAreaSet))
    return isCropAreaSet, crop_area


def get_video_metadata(file_path):
    logging.debug("Extracting metadata")
    metadata = {}
    meta_dict = ffmpeg.probe(file_path)
    logging.debug(meta_dict["streams"][0])
    nb_frames = meta_dict["streams"][0]["nb_frames"]
    width = meta_dict["streams"][0]["width"]
    height = meta_dict["streams"][0]["height"]
    r_frames = meta_dict["streams"][0]["r_frame_rate"]

    metadata = {
        'nb_frames': nb_frames,
        'width': width,
        'height': height,
        'r_frames': r_frames
    }
    return metadata


# call back function
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    global last_end_xy
    global last_start_xy
    roi = None

    #keep a copy of the image for refreshing
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
            logging.debug("Points Selected:{0}".format(refPt))
            # P1 in 1st quadrant and P2 in 3rd quadrant
            if refPt[0][1] < refPt[1][1] and refPt[0][0] < refPt[1][0]:
                logging.debug("P1: Q1 and P2: Q3")
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                last_start_xy = refPt[0]
                last_end_xy = refPt[1]


            # P1 in 2nd quadrant and P2 in 4th quadrant
            # refPt[0][1] > refPt[1][1] and refPt[0][0] < refPt[1][0]
            # refPt[0][1] < refPt[1][1] and refPt[0][0] > refPt[1][0]
            elif refPt[0][1] > refPt[1][1] and refPt[0][0] < refPt[1][0]:
                logging.debug("P1: Q2 and P2: Q4")
                roi = clone[refPt[1][1]:refPt[0][1], refPt[0][0]:refPt[1][0]]
                last_start_xy = [(refPt[0][0], refPt[1][1])]
                last_end_xy = [(refPt[1][0], refPt[0][1])]


            # P1 in 3rd quadrant and P2 in 1st quadrant
            elif refPt[0][1] > refPt[1][1] and refPt[0][0] > refPt[1][0]:
                logging.debug("P1: Q3 and P2: Q1")
                roi = clone[refPt[1][1]:refPt[0][1], refPt[1][0]:refPt[0][0]]
                last_start_xy = refPt[1]
                last_end_xy = refPt[0]


            # P1 in 4th quadrant and P2 in 2nd quadrant
            elif refPt[0][1] < refPt[1][1] and refPt[0][0] > refPt[1][0]:
                logging.debug("P1:Q4 and P2: Q2")
                roi = clone[refPt[0][1]:refPt[1][1], refPt[1][0]:refPt[0][0]]
                last_start_xy = [(refPt[1][0], refPt[0][1])]
                last_end_xy = [(refPt[0][0], refPt[1][1])]

            else:
                logging.warning("Points are not making a valid rectanlge")
                return


            cv2.imshow("ROI", roi)

            # Keep only up to two points
            refPt.clear()


def generate_crop_area_many_videos(dir, video_format, out_json, output_path):

    for file in os.listdir(dir):
        # if file.endswith(video_format):
        #
        #     try:
        #         # Metadata
        #         # get_video_metadata()
        #         meta_dict = ffmpeg.probe('{0}/{1}'.format(dir, file))
        #         logging.debug(meta_dict["streams"][0])
        #         nb_frames = meta_dict["streams"][0]["nb_frames"]
        #         width = meta_dict["streams"][0]["width"]
        #         height = meta_dict["streams"][0]["height"]
        #         r_frames = meta_dict["streams"][0]["r_frame_rate"]
        #
        #         logging.info("video file name: {0}, resolution: {1}x{2}, number of frame: {3}, frame per second {4} ".
        #                      format(file, width, height, nb_frames, r_frames))
        #         print("video file name: {0}, resolution: {1}x{2}, number of frame: {3}, frame per second {4} ".
        #               format(file, width, height, nb_frames, r_frames))
        #
        #         # get cropping data
        #         isCropAreaSet, crop_area = get_cropping_area('{0}/{1}'.format(dir, file), width, height, nb_frames)
        #
        #         # This flag is set inside get_cropping_area if the user press q or Q button instead of esc
        #         # assume this is a force exit therefore not saving the current status
        #         if TERMINATE_SEARCH:
        #             break
        #
        #         json_array = []
        #         if crop_area:
        #             logging.debug("Creating a json ...")
        #             metadata = {
        #                 'nb_frames': nb_frames,
        #                 "isCropAreaSet": isCropAreaSet,
        #                 'v_width': width,
        #                 'v_height': height,
        #                 'r_frame_rate': r_frames
        #             }
        #             json_array.append({"crop_area": crop_area, "metadata": metadata})
        #             out_json["{0}".format(file)] = json_array
        #         else:
        #             logging.info("crop area is not selected")
        #
        #     except ffmpeg.Error as e:
        #         logging.error("Error: {0}".format(e.stderr), file=sys.stderr)
        #         sys.exit(1)

    # create json
        file_path = '{0}/{1}'.format(dir, file)
        out_json = generate_crop_area_one_video(file_path,video_format,out_json,output_path)

    return out_json



def generate_crop_area_one_video( file_path, video_format, out_json, output_path):
    global refPt

    if os.path.basename(file_path).endswith(video_format):
        try:
            # Metadata
            # get_video_metadata()
            meta_data = get_video_metadata(file_path)

            logging.info("video file name: {0}, resolution: {1}x{2}, number of frame: {3}, frame per second {4} ".
                         format(os.path.basename(file_path), meta_data["width"],meta_data["height"],meta_data["nb_frames"],meta_data["r_frames"]))
            print("video file name: {0}, resolution: {1}x{2}, number of frame: {3}, frame per second {4} ".
                  format(os.path.basename(file_path), meta_data["width"],meta_data["height"],meta_data["nb_frames"],meta_data["r_frames"]))
            # get cropping data
            isCropAreaSet, crop_area = get_cropping_area(file_path, meta_data["width"],meta_data["height"],meta_data["nb_frames"],output_path)

            json_array = []
            if crop_area:
                logging.debug("Cropping area {} is created for {}".format(crop_area, file_path))
                json_array.append({"crop_area": crop_area, "metadata": meta_data, "isCropAreaSet":isCropAreaSet})
                out_json["{0}".format(os.path.basename(file_path))] = json_array
            else:
                logging.info("crop area is not selected")

        except ffmpeg.Error as e:
            logging.error("Error: {0}".format(e.stderr), file=sys.stderr)
            sys.exit(1)

    return out_json




if __name__ == '__main__':
    #initiate the logger
    logging.basicConfig(filename='log\edit_video_bounds_batch{}.log'.format(datetime.datetime.now().strftime('%d_%m_%Y')),
                        level=logging.DEBUG)
    logging.info("=============== Start at {} ==================".format(datetime.datetime.now()))

    parser = argparse.ArgumentParser(description='Access metadeta and creat a json with crop dimenssion')
    parser.add_argument('--dirOrfile',
                        help='Path to a video file or directory containing many video files. If is a directory program will'
                             'iterate over each video',
                        required=True)
    parser.add_argument('--out',
                        help='Path for the output json including cropping area',
                        required=True)
    parser.add_argument('--file_format',
                        help='Filter the file in the directory, for example .m4v or .mp4',
                        required=False)
    parser.add_argument('--oname',
                        help='Name of the json output',
                        default="cropped_dimension",
                        required=False)

    args = parser.parse_args()

    #create json
    output_json = {}

    if os.path.isfile(args.dirOrfile):
        logging.info("{} is a file".format(args.dirOrfile))
        out_json = generate_crop_area_one_video(args.dirOrfile, args.file_format, output_json, args.out)
    elif os.path.isdir(args.dirOrfile):
        logging.info("{} is a directory".format(args.dirOrfile))
        out_json = generate_crop_area_many_videos(args.dirOrfile, args.file_format, output_json, args.out)
    else:
        logging.error("invalid path {}".format(args.dirOrfile))



    # dump json
    output_fname = args.oname + "_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
    out_directory = args.out + "/" + output_fname
    with open(out_directory, 'w') as f:
        json.dump(out_json, f)
        logging.info("{0} created at {1}.json".format(output_fname, out_directory))



    logging.info("=============== End at {} ==================".format(datetime.datetime.now()))
