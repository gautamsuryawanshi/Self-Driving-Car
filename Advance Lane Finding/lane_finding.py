# -*- coding: utf-8 -*-


import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

"""Camera Calibration:
Used Chessboard images to calibrate camera and found out calibration matrix and distortion coefficents.
"""

def calibrateCamera():
    nx = 9 #enter the number of inside corners in x
    ny = 6 #enter the number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('./camera_cal/calibration*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert to grayscale
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
    
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

ret, mtx, dist, rvecs, tvecs = calibrateCamera()

"""Undistort Image function which undistorts a image using calibration matrix and distortion coefficents"""

def undistort_img(img):
  dst = cv2.undistort(img, mtx, dist, None, mtx)
  return dst


"""Color and Gradient thresholding to get a binary thresholded image"""

def gradient_and_color_thresholding(img):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
  abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
  scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

  # Threshold x gradient
  thresh_min = 20
  thresh_max = 100
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

  # Convert to HLS color space and separate the S channel
  # Note: img is the undistorted image
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  s_channel = hls[:,:,2]

  # Threshold color channel
  s_thresh_min = 170
  s_thresh_max = 255
  s_binary = np.zeros_like(s_channel)
  s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

  # Stack each channel to view their individual contributions in green and blue respectively
  # This returns a stack of the two binary images, whose components you can see as different colors
  color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

  # Combine the two binary thresholds
  combined_binary = np.zeros_like(sxbinary)
  combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

  return combined_binary



"""Application of Perspective transform to get a Birds-Eye View"""

lanewidth=300
h=720
w=1280
# define src points through getting a plane in the image
src = np.float32([(575,464),(707,464), (258,682), (1049,682)])
dst = np.float32([(lanewidth,0),(w-lanewidth,0),(lanewidth,h),(w-lanewidth,h)])

def get_camera_matrix():
    return cv2.getPerspectiveTransform(src, dst)
    
def get_camera_matrix_inv():
    return cv2.getPerspectiveTransform(dst, src)


def perspective_transform(img):
  h,w=img.shape[:2]
  img_birds_eye = cv2.warpPerspective(img, get_camera_matrix(), (w,h), flags=cv2.INTER_LINEAR)
  return img_birds_eye



"""Lane Boundary Detection using sliding window method"""

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    

    return  ploty,left_fit,right_fit,leftx, lefty, rightx, righty,left_fitx, right_fitx,out_img



def warp_back(img,warped,left_fitx,right_fitx,ploty):
  # Create an image to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))


  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

  # Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, get_camera_matrix_inv(), (warped.shape[1], warped.shape[0])) 

  
  # Combine the result with the original image
  result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

  return result

def get_road_curvature(binary,left_fit,right_fit):
  ym = 30/720 # meters per pixel in y dimension
  xm = 3.7/700 # meters per pixel in x dimension
    
  ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0] )
  leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty +left_fit[2]
  rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
  car_position= binary.shape[1]/2
    
    # Fit new polynomials to x,y 
  left_fit_cr = np.polyfit(ploty*ym, leftx*xm, 2)
  right_fit_cr = np.polyfit(ploty*ym, rightx*xm, 2)
    
    # Calculate the new radius of curvature
  y_eval=np.max(ploty)
    
  left_curve_radius = ((1 + (2*left_fit_cr[0]*y_eval*ym + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curve_radius = ((1 + (2*right_fit_cr[0]*y_eval*ym + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
  left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
  right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
  actual_position= (left_lane_bottom+ right_lane_bottom)/2
    
  distance= (car_position - actual_position)* xm
    
  return (left_curve_radius + right_curve_radius)/2, distance



def process_image(img):
  img = undistort_img(img)                                                        
  undist = gradient_and_color_thresholding(img)                                   
  binary_warped = perspective_transform(undist)                                  
  ploty,left_fit,right_fit,leftx, lefty, rightx, righty,left_fitx, right_fitx,out_img = fit_polynomial(binary_warped)              
  warped = binary_warped.copy()
  result = warp_back(img,warped,left_fitx,right_fitx,ploty)
  return result


'''  
cap = cv2.VideoCapture('project_video.mp4')
while (cap.isOpened()):
    ret, frame = cap.read()
    result = process_image(frame)
    cv2.imshow('result', result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''


from moviepy.editor import VideoFileClip
output = 'project_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)
