import cv2
import numpy as np

cap = cv2.VideoCapture(0)
imgTarget = cv2.imread("Images/edm_image.png")
myVid = cv2.VideoCapture("Images/IMG_4010(1).mp4")

detection = False
frameCounter = 0

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))
orb = cv2.ORB_create(nfeatures=1500)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
des1 = np.float32(des1)

while True:
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()

    try:
        kp2, des2 = orb.detectAndCompute(imgWebcam, None)
        des2 = np.float32(des2)

        if detection == False:
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        else:
            if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
                myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frameCounter = 0
            success, imgVideo = myVid.read()
            imgVideo = cv2.resize(imgVideo, (wT, hT))

        bf = cv2.BFMatcher()

        if des2 is not None:  # Check if des2 is not empty
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            print(len(good))
            if len(good) > 30:
                detection = True
                srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)

                pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

                imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

                maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
                cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
                maskInv = cv2.bitwise_not(maskNew)
                imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
                imgAug = cv2.bitwise_or(imgWarp, imgAug)

    except Exception as e:
        pass

    cv2.imshow("imgStacked", imgAug)
    frameCounter += 1

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
