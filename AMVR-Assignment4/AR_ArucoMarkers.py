import cv2
import numpy as np



def find_and_warp(frame, source, cornerIds):

    imgH, imgW, c = frame.shape
    srcH, srcW, e = source.shape
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

    if len(markerCorners) != 4:
        markerIds = np.array([])
    else:
        markerIds.flatten()

    refPts = []
    for i in cornerIds:
        j = np.squeeze(np.where(markerIds == i))

        if j.size == 0:
            continue
        else:
            j = j[0]

        markerCorners = np.array(markerCorners)
        corner = np.squeeze(markerCorners[j])
        refPts.append(corner)

    if len(refPts) != 4:
        return None

    (refPtTL, refPtTR, refPtBL, refPtBR) = np.array(refPts)
    print("TL", refPtTL)
    print("TR", refPtTR)
    print("BR", refPtBR)
    print("BL", refPtBL)
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    srcMat = np.array([[0, 0],[srcW, 0],[srcW, srcH], [0, srcH]])
    (H, _) = cv2.findHomography(srcMat, dstMat)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))
    mask = np.zeros((imgH, imgW), dtype="uint8")

    cv2.fillConvexPoly(mask, dstMat.astype("int32"), 255)
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)

    warpedMultiplied = cv2.multiply(warped.astype("float"),
                                    maskScaled)
    imageMultiplied = cv2.multiply(frame.astype(float),
                                   1.0 - maskScaled)

    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")
    return output




image_to_display = cv2.imread('Images/image.jpg')
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
videostream = cv2.VideoCapture(0)

while True:
    ret, frame = videostream.read()
    warped = find_and_warp(
        frame, image_to_display,  # Use the image to display
        cornerIds=(0, 1, 2, 3),  # Adjust cornerIds to match your marker IDs
    )
    if warped is not None:
        frame = warped
    cv2.imshow('beu', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

