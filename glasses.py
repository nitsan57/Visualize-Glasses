import cv2
import numpy as np

# class Complex:
#     def __init__(self, realpart, imagpart):
#         self.r = realpart
#         self.i = imagpart


def cut_image(img):
    min_x = np.shape(img)[1]
    min_y= np.shape(img)[0]
    max_x = 0
    max_y = 0
    for y, a_x in enumerate(img[:,:,3]):
        for x, a in enumerate(a_x):
            if a !=0:
                min_x = min(x, min_x)
                max_x = max(x, max_x)
                min_y = min(y, min_y)
                max_y = max(y, max_y)
    if(min_x != max_x and min_y != max_y):
        return img[min_y:max_y,min_x:max_x,]
    return img


def rotateImage_from_existing_matrix(image, angle, M=None, nW=None, nH=None):
    return rotateImage(image, angle, M, nW, nH)

def rotateImage(image, angle, M=None, nW=None, nH=None):
    if M is not None:
        res = cv2.warpAffine(image, M, (nW, nH))
        return res, M
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -(angle*1.03), 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    res = cv2.warpAffine(image, M, (nW, nH))
    return res, M

def load_models():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    return face_cascade, eye_cascade

def load_resources_pre_process():
    orig_glasses = cv2.imread("glasses3.png",cv2.IMREAD_UNCHANGED)
    orig_glasses = cut_image(orig_glasses)
    orig_glasses_alpha_channel = np.repeat(orig_glasses[:,:,3][:, :, np.newaxis], 3, axis=2)
    orig_glasses = orig_glasses[:,:,:3]
    orig_glasses = cv2.medianBlur(orig_glasses, 5)
    orig_glasses_alpha_channel = cv2.normalize(orig_glasses_alpha_channel, None, alpha=0, beta=1,
                                          norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)
    orig_glasses = cv2.normalize(orig_glasses, None, alpha=0, beta=1,
                                          norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_32F)
    return orig_glasses, orig_glasses_alpha_channel

def check_if_image_is_good(left_eye, glasses_width, glasses_height, video):
    if (left_eye[0] + glasses_width) > video.shape[1] or left_eye[0] < 0 or (left_eye[1] + glasses_height) > \
            video.shape[0] or left_eye[1] < 0:
        x = cv2.putText(video, 'You are out of the view!',
                        (40, 40),
                        20,
                        1,
                        (0, 0, 0))
        cv2.imshow('img', x)
        k = cv2.waitKey(30) & 0xff
        return False
    return True

def pre_process(left_eye, right_eye, orig_glasses, orig_glasses_alpha_channel):
    glasses_width = int(abs(left_eye[0] - (right_eye[0] + right_eye[2])))
    glasses_width = int(glasses_width + 0.1 * glasses_width)
    last_good_frame_info = np.copy(left_eye), np.copy(right_eye)
    left_eye[0] -= int((0.1 * glasses_width) / 2)
    right_eye[0] += int((0.1 * glasses_width) / 2)
    r = glasses_width / (np.shape(orig_glasses)[1])
    glasses_height = int(r * np.shape(orig_glasses)[0])
    glasses = cv2.resize(orig_glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
    glasses_alpha_channel = cv2.resize(orig_glasses_alpha_channel, (glasses_width, glasses_height),
                                       interpolation=cv2.INTER_AREA)
    eye_rotation_angle = np.rad2deg(np.arctan2(right_eye[1] - left_eye[1], abs(left_eye[0] - (right_eye[0]))))
    glasses = glasses * glasses_alpha_channel
    return glasses, glasses_alpha_channel, eye_rotation_angle, last_good_frame_info


def alignImages(im1, im2):
    im1Gray = cv2.cvtColor(im1 * 255, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    im2Gray = cv2.cvtColor(im2 * 255, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1Gray, None)
    kp2, des2 = sift.detectAndCompute(im2Gray, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    GOOD_MATCH_PERCENT = 0.15
    matches = bf.match(des1, des2) #, k=2)
    matches.sort(key=lambda x: x.distance, reverse=False)
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    MIN_MATCH_COUNT = 10
    good = matches
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # matches = matches[:MIN_MATCH_COUNT]
    # src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[0]]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # Find homography
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=1)
        return h
    return None

def calc_transformation(eq_image, new_image,glasses, eq_eyes, new_eyes):
    # h = alignImages(eq_image, new_image)
    l_ee, r_ee = eq_eyes
    l_e, r_e = new_eyes
    src = np.array(np.array([[l_ee[0], l_ee[1]], [l_ee[0] +l_ee[2], l_ee[1]],[l_ee[0], l_ee[1]+l_ee[3]],[l_ee[0]+l_ee[2], l_ee[1]+l_ee[3]], [r_ee[0], r_ee[1]],[r_ee[0] + r_ee[2], r_ee[1]],[r_ee[0], r_ee[1] + r_ee[3]],[r_ee[0] + r_ee[2], r_ee[1]+ r_ee[3]]]))
    dst = np.array(np.array([[l_e[0], l_e[1]], [l_e[0] + l_e[2], l_e[1]], [l_e[0], l_e[1] + l_e[3]],
                             [l_e[0] + l_e[2], l_e[1] + l_e[3]], [r_e[0], r_e[1]],
                             [r_e[0] + r_e[2], r_e[1]], [r_e[0], r_e[1] + r_e[3]],
                             [r_e[0] + r_e[2], r_e[1] + r_e[3]]]))
    print(src)
    print(dst)
    print("###")
    h, mask = cv2.findHomography(src, dst)
    if h is None:
        return
    # Use homography
    height, width, channels = new_image.shape
    im1Reg = cv2.warpPerspective(new_image, h, (width, height))

    cv2.imshow("Img1", eq_image)
    cv2.imshow("Img2", im1Reg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def not_enough_eyes_detected(last_good_frame, video):
    if last_good_frame != []:
        cv2.imshow('img', last_good_frame)
        last_good_frame = []
    else:
        x = cv2.putText(video, 'No eyes detected! or too much(are two persons there?)!',
                        (40, 40),
                        20,
                        1,
                        (0, 0, 0))
        cv2.imshow('img', x)
        k = cv2.waitKey(30) & 0xff

def calc_eyes(eyes, init_phase, last_good_frame_info):
    left_eye, right_eye = eyes[-eyes[:, 0].argsort()]
    if not init_phase:
        left_avg = np.array(last_good_frame_info[0])
        right_avg = np.array(last_good_frame_info[1])
        NOT_MOVE_TRESHOLD = 4
        if not (np.any(left_eye[:2] < left_avg[:2] - NOT_MOVE_TRESHOLD) or np.any(
                    left_eye[:2] > left_avg[:2] + NOT_MOVE_TRESHOLD)):
            left_eye = ((left_avg * 5 + left_eye) / 6).astype(np.int)
            right_eye = ((right_avg * 5 + right_eye) / 6).astype(np.int)
    return left_eye, right_eye


def main():
    init_phase = True
    face_cascade, eye_cascade = load_models()
    cap = cv2.VideoCapture(0)
    orig_glasses, orig_glasses_alpha_channel = load_resources_pre_process()
    last_good_frame = []
    i = 0
    min_angle = 0.03
    last_good_frame_info = ()  # left, right
    EQ_IMAGE = None
    eq_eyes = None
    while 1:
        ret, video = cap.read()
        gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        video = cv2.normalize(video, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_32F)
        eyes = eye_cascade.detectMultiScale(gray, minNeighbors=2)
        if (len(eyes) != 2):  # temp to ignore one eye or less detection
            not_enough_eyes_detected(last_good_frame, video)
            continue
        left_eye, right_eye = calc_eyes(eyes, init_phase, last_good_frame_info)
        glasses, glasses_alpha_channel, eye_rotation_angle, last_good_frame_info = pre_process(left_eye, right_eye, orig_glasses, orig_glasses_alpha_channel)
        # if EQ_IMAGE is not None:
        #     calc_transformation(EQ_IMAGE, video, glasses, eq_eyes, (left_eye, right_eye))
        if abs(eye_rotation_angle) < min_angle:
            EQ_IMAGE = np.copy(video)
            eq_eyes = np.copy((left_eye, right_eye))
            min_angle = abs(eye_rotation_angle)
        glasses, M = rotateImage(glasses, eye_rotation_angle)
        glasses_width = glasses.shape[1]
        glasses_height = glasses.shape[0]
        glasses_alpha_channel, M = rotateImage_from_existing_matrix(glasses_alpha_channel, eye_rotation_angle, M, glasses_width ,glasses_height)
        if not check_if_image_is_good(left_eye, glasses_width, glasses_height, video):
            continue
        relative_eye = left_eye
        if eye_rotation_angle < 0:
            relative_eye = right_eye
        face_glasses_area = video[relative_eye[1]:relative_eye[1] + glasses_height, left_eye[0]:left_eye[0] + glasses_width]
        face_mask = face_glasses_area * (1-glasses_alpha_channel)
        video[relative_eye[1]:relative_eye[1] + glasses_height, left_eye[0]:left_eye[0] + glasses_width] = face_mask + glasses
        last_good_frame = video
        cv2.imshow('img',video)
        if i == 1 and init_phase:
            init_phase = False
        i += 1
        i = i % 4
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#
#
#
#
#
# #
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
# #



# from tkinter import *
# from PIL import Image
# from PIL import ImageTk
# import tkinter.filedialog as tkFileDialog
# import cv2
#
#
# def select_image():
#     # grab a reference to the image panels
#     global panelA, panelB
#
#     # open a file chooser dialog and allow the user to select an input
#     # image
#     path = tkFileDialog.askopenfilename()
#
#     # ensure a file path was selected
#     if len(path) > 0:
#         # load the image from disk, convert it to grayscale, and detect
#         # edges in it
#         image = cv2.imread(path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         edged = cv2.Canny(gray, 50, 100)
#
#         #  represents images in BGR order; however PIL represents
#         # images in RGB order, so we need to swap the channels
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # convert the images to PIL format...
#         image = Image.fromarray(image)
#         edged = Image.fromarray(edged)
#
#         # ...and then to ImageTk format
#         image = ImageTk.PhotoImage(image)
#         edged = ImageTk.PhotoImage(edged)
#
#         # if the panels are None, initialize them
#         if panelA is None or panelB is None:
#             # the first panel will store our original image
#             panelA = Label(image=image)
#             panelA.image = image
#             panelA.pack(side="left", padx=10, pady=10)
#
#             # while the second panel will store the edge map
#             panelB = Label(image=edged)
#             panelB.image = edged
#             panelB.pack(side="right", padx=10, pady=10)
#
#         # otherwise, update the image panels
#         else:
#             # update the pannels
#             panelA.configure(image=image)
#             panelB.configure(image=edged)
#             panelA.image = image
#             panelB.image = edged
#
#
# # initialize the window toolkit along with the two image panels
# root = Tk()
# panelA = None
# panelB = None
#
# # create a button, then when pressed, will trigger a file chooser
# # dialog and allow the user to select an input image; then add the
# # button the GUI
# btn = Button(root, text="Select an image", command=select_image)
# btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
#
# # kick off the GUI
# root.mainloop()