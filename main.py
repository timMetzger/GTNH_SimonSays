import time

import cv2
import mss
import numpy as np
import pyautogui


#TODO: Dialog boxs

class BoundingBoxWidget(object):
    def __init__(self):
        # Grab a screenshot to get bounding box from
        pyautogui.screenshot(r"C:\Users\timme\Desktop\Programming\GTNH_SimonSays_Cheat\test.png")
        self.original_image = cv2.imread('test.png')
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]

        # Record ending (x,y) coordintes on left mouse button release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            print('top left: {}, bottom right: {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            print('x,y,w,h : ({}, {}, {}, {})'.format(self.image_coordinates[0][0], self.image_coordinates[0][1], self.image_coordinates[1][0] - self.image_coordinates[0][0], self.image_coordinates[1][1] - self.image_coordinates[0][1]))

            # Draw rectangle
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image", self.clone)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()

    def show_image(self):
        return self.clone

    def get_bounds(self):
        return {"top":self.image_coordinates[0][0],"left":self.image_coordinates[0][1],
                "width":self.image_coordinates[1][0] - self.image_coordinates[0][0],
                "height":self.image_coordinates[1][1] - self.image_coordinates[0][1]}

class BoxLocationsWidget(object):
    def __init__(self):
        self.count = 0
        self.original_image = cv2.imread('test.png')
        self.clone = self.original_image.copy()

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)

        # Bounding box reference points
        self.box_coordinates = {}

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.count += 1
            self.box_coordinates[self.count] = (x,y)


            # Draw circle at event
            cv2.circle(self.clone,(x,y),25,(36,255,12), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()
            self.count = 0
            self.box_coordinates = {}

    def get_box_bounds(self):
        return self.box_coordinates

    def show_image(self):
        return self.clone

def get_bounding_box():
    boundingbox_widget = BoundingBoxWidget()
    while True:
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    return boundingbox_widget.get_bounds()

def get_selection_boxs():
    boundingbox_widget = BoxLocationsWidget()
    while True:
        cv2.imshow('image', boundingbox_widget.show_image())
        key = cv2.waitKey(1)

        # Close program with keyboard 'q'
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    return boundingbox_widget.get_box_bounds()

def main():
    print("WELCOME")
    print("-------")
    print("Bounding Box Procedure")
    print("*Hold left click at the start of the box and release for the end of the box")
    print("*Right Click to clear")
    print("*Press q when satisfied")
    print("-----------------------")

    processing_area = get_bounding_box()
    window_bbox = {"top":0,"left":0,"width":3400,"height":1440}

    y0 = processing_area["left"]
    y1 = processing_area["height"]
    x0 = processing_area["top"]
    x1 = processing_area["width"]

    print("Box Locations Procedure")
    print("-----------------------")
    print("Left click locations of boxs (this is where the computer will click)")
    print("Right click if you make an error to clear")
    print("Press q when satisfied")
    print("-----------------------------------------")

    box_coords = get_selection_boxs()

    print("PROCESSING START")
    print("----------------")

    sensitivity = 10


    with mss.mss() as sct:
        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(window_bbox))

            # Blocking out the white question mark
            cv2.circle(img,(3440//2,1440//2),150,(0,0,0),-1)

            # Cropping to processing area
            crop = img[y0:y0+y1,x0:x0+x1]


            lower_white = np.array([0,0,255-sensitivity])
            upper_white = np.array([255,sensitivity,255])

            # Changing to hsv color space
            hsvImg = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsvImg,lower_white,upper_white)


            # Display the picture
            cv2.imshow("OpenCV/Numpy normal", white_mask)

            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.desqtroyAllWindows()
                break




if __name__ == "__main__":
    main()