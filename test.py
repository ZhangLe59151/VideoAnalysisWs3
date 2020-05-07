from tensorflow.keras.layers import AveragePooling2D 
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import SGD 
from collections import deque 
import numpy as np 
import pickle 
import cv2 
import os
import matplotlib.pyplot as plt

vs               = cv2.VideoCapture('27342.MP4')
(grabbed,frame1) = vs.read()
(grabbed,frame2) = vs.read() 
plt.imshow(frame1)
plt.axis('off')
plt.imshow(frame2)
plt.axis('off')
fps             = vs.get(cv2.CAP_PROP_FPS)

ptsLbl      = ['Nose',          # 0
               'Neck',          # 1
               'R:Shoulder',    # 2
               'R:Elbow',       # 3
               'R:Wrist',       # 4
               'L:Shoulder',    # 5
               'L:Elbow',       # 6
               'L:Wrist',       # 7
               'R:Hip',         # 8
               'R:Knee',        # 9
               'R:Ankle',       # 10
               'L:Hip',         # 11
               'L:Knee',        # 12
               'L:Ankle',       # 13
               'R:Eye',         # 14
               'L:Eye',         # 15
               'R:Ear',         # 16
               'L:Ear']         # 17
                                # 18 (background)

links       = [[1,2],       # Neck to R:Shoulder
               [1,5],       # Neck to L:Shoulder
               [2,3],       # R:Shoulder to R:Elbow
               [3,4],       # R:Elbow to R:Wrist
               [5,6],       # L:Shoulder to L:Elbow 
               [6,7],       # L:Elbow to L:Wrist
               [1,8],       # Neck to R:Hip
               [8,9],       # R:Hip to R:Knee
               [9,10],      # R:Knee to R:Ankle
               [1,11],      # Neck to L:Hip
               [11,12],     # L:Hip to L:Knee
               [12,13],     # L:Knee to L:Ankle
               [1,0],       # Neck to Nose
               [0,14],      # Nose to R:Eye
               [14,16],     # R:Eye to L:Ear
               [0,15],      # Nose to L:Eye
               [15,17],     # L:Eye to L:Ear
               [2,17],      # R:Shoulder to L:Ear
               [5,16]]      # L:Shoulder to R:Ear

                            # In total there are 19 pairs in 'links'


                            # Pairs of channel that corresponds to the
                            # pairs in 'links'                            
                            # For example, the PAFs of link [1,2] are
                            # located at channel [31,32] 
pafCh       = [[31,32],     # Neck to R:Shoulder
               [39,40],     # Neck to L:Shoulder
               [33,34],     # R:Shoulder to R:Elbow
               [35,36],     # R:Elbow to R:Wrist
               [41,42],     # L:Shoulder to L:Elbow 
               [43,44],     # L:Elbow to L:Wrist
               [19,20],     # Neck to R:Hip
               [21,22],     # R:Hip to R:Knee
               [23,24],     # R:Knee to R:Ankle
               [25,26],     # Neck to L:Hip
               [27,28],     # L:Hip to L:Knee
               [29,30],     # L:Knee to L:Ankle
               [47,48],     # Neck to Nose
               [49,50],     # Nose to R:Eye
               [53,54],     # R:Eye to L:Ear
               [51,52],     # Nose to L:Eye
               [55,56],     # L:Eye to L:Ear
               [37,38],     # R:Shoulder to L:Ear
               [45,46]]     # L:Shoulder to R:Ear

colours     = [[0,100,255],
               [0,100,255],
               [0,255,255],
               [0,100,255],
               [0,255,255],
               [0,100,255],
               [0,255,0],
               [255,200,100],
               [255,0,255],
               [0,255,0],
               [255,200,100],
               [255,0,255],
               [0,0,255],
               [255,0,0],
               [200,200,0],
               [255,0,0],
               [200,200,0],
               [0,0,0]]

ptColours   = [[0,100,255],     # 0
               [0,100,255],     # 1
               [0,100,255],     # 2
               [0,255,0],       # 3
               [0,255,255],     # 4
               [0,100,255],     # 5
               [0,255,0],       # 6
               [0,255,255],     # 7
               [255,0,255],     # 8
               [0,0,255],       # 9
               [255,0,0],       # 10
               [255,0,255],     # 11
               [0,0,255],       # 12
               [255,0,0],       # 13
               [0,0,0],         # 14
               [0,0,0],         # 15
               [200,200,0],     # 16
               [200,200,0]]     # 17

def searchPts(prMap,
              prThres=0.1):
    blur        = cv2.GaussianBlur(prMap,
                                   (3,3),
                                   0,           # Set SigmaX to 0
                                   0)           # Set SigmaY to 0, so that the SigmaX and SigmaY
                                                # are computed from kernel size
    mask        = np.uint8(blur > prThres)
    pts         = []
    
                                                # find all the blobs in the mask
    if cv2.__version__ == '3.4.2':
        (_,ctrs,_)  = cv2.findContours(mask,
                                       cv2.RETR_TREE,           # contour retrieval modes
                                       cv2.CHAIN_APPROX_SIMPLE) # contounr approximation methods
    else:
                                                # for opencv version 4.0 and above
        (ctrs,_)    = cv2.findContours(mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
                                                # Go through each blob and find the maximum point
                                                # in the blob
    for ctr in ctrs:
        blobs       = np.zeros(mask.shape)
        blobs       = cv2.fillConvexPoly(blobs,
                                         ctr,
                                         1)
        blob        = blur*blobs
        (_,maxVal,
         _,maxLoc)  = cv2.minMaxLoc(blob)       # The output of 'maxLoc' is (x,y)
        pts.append(maxLoc + (prMap[maxLoc[1],maxLoc[0]],))
                                                # The output of 'pts' is a list
                                                # for each item in the list is a tuple of
                                                # (x,y,probability)
    return (pts,mask)


def getAllPoints(cfMaps,
                 imgWidth,
                 imgHeight,
                 cfThres=0.1,
                 numOfKeyPts=18):
    ptGrp       = []
    ptList      = np.zeros((0,3))
    idx         = 0
    
    for keyPts in range(numOfKeyPts):
        prMap       = cfMaps[0,keyPts,:,:]
        prMap       = cv2.resize(prMap,
                                 (imgWidth,imgHeight))
        (pts,_)     = searchPts(prMap,
                                prThres=cfThres)
        ptWithId    = []
        for pt in range(len(pts)):
            ptWithId.append(pts[pt] + (idx,))
            ptList  = np.vstack([ptList,
                                 pts[pt]])
            idx     = idx+1
            
        ptGrp.append(ptWithId)
    
    return (ptGrp,ptList)

def getAllLinks(pafs,
                links,
                linkCh,
                ptGrp,
                imgSize,
                numOfPtsForLine=10,
                pafThres=0.1,
                cfThres=0.7):

    linksWithPairs  = []
    linksNoPairs    = []
    (H,W)           = imgSize

    for l in range(len(linkCh)):
                                            # Extract the two matrix for 
                                            # a single part affinity field that
                                            # involved in a link
        paf0        = output[0,pafCh[l][0],:,:]
        paf1        = output[0,pafCh[l][1],:,:]
        paf0        = cv2.resize(paf0,      # resize the matrix to the image
                                 (W,H))     # original size
        paf1        = cv2.resize(paf1,
                                 (W,H))
        
        linkPt0     = links[l][0]           # The first point in the link
        linkPt1     = links[l][1]           # The second point in the link
        
        pts0        = ptGrp[linkPt0]        # Get all the points identified as the first point in the link
        pts1        = ptGrp[linkPt1]        # Get all the points identified as the second point in the link
        
        numOfPts0   = len(pts0)
        numOfPts1   = len(pts1)
        
        if (numOfPts0 != 0 and numOfPts1 !=0):
            inPairs     = np.zeros((0,3))
            
            for i in range(numOfPts0):
                maxScore    = -1
                maxj        = -1
                linkLocated = 0
                
                for j in range(numOfPts1):
                                                        # e.g. pts0[i] --> (600,363,0.85,2)
                                                        # e.g. pts1[j] --> (230,242,0.77,5)
                    dij         = np.subtract(pts1[j][:2],pts0[i][:2])
                                                        # e.g. dij --> [-370,-121]
                    norm        = np.linalg.norm(dij)
                                                        # e.g. norm --> 389.28
                    if norm:
                        dij     = dij/norm              # e.g. dij --> [-0.95,-0.31]
                                                        # a unit vector
                                                        
                                                        # Generate 10 points, starting from,
                                                        # e.g. (600,363) to (230,242)
                                                        # Note: the notation for each point is (x,y)
                                                        # the output is a list that has 10 tuples of size 2
                    linePts     = list(zip(np.linspace(pts0[i][0],
                                                       pts1[j][0],
                                                       num=numOfPtsForLine),
                                           np.linspace(pts0[i][1],
                                                       pts1[j][1],
                                                       num=numOfPtsForLine)))        
                    pafVectors  = []
                    
                                                        # Get the vector (a 2 values tuple) from
                                                        # PAF for each point in the line
                    for k in range(len(linePts)):                            
                        vector  = [paf0[int(round(linePts[k][1])),
                                        int(round(linePts[k][0]))],
                                   paf1[int(round(linePts[k][1])),
                                        int(round(linePts[k][0]))]]
                        pafVectors.append(vector)       # pafVectors is a list of 10 tuple,
                                                        # each tuple is a size of 2
                    
                    pafScore    = np.dot(pafVectors,dij)    # perform dot product between a vector and dij, for all vectors
                                                            # This is to get the magnitude of each vector in the
                                                            # direction of the unit vector
                    avgPafScore = sum(pafScore)/len(pafScore)  
                    ptsBydThres = np.where(pafScore > pafThres)     # points with score that exceeds 'pafThres'
                                                                    # the output is a tuple, with a numpy array inside
                    ptsBydThres = ptsBydThres[0]                    # this is required to extract the numpy array 
                    
                    if (len(ptsBydThres)/numOfPtsForLine) > cfThres:
                        if avgPafScore > maxScore:
                            maxj        = j
                            maxScore    = avgPafScore
                            linkLocated = 1
                if linkLocated:
                    inPairs     = np.append(inPairs,
                                            [[pts0[i][3],pts1[maxj][3],maxScore]],
                                            axis=0)
            if inPairs.shape[0] == 0:           # When there is no pair in this link
                linksNoPairs.append(l)
                linksWithPairs.append([])
            else:
                linksWithPairs.append(inPairs)
        else:
            linksNoPairs.append(l)
            linksWithPairs.append([])
            
    return (linksWithPairs,linksNoPairs)

def getPersons(linksWithPairs,
               linksNoPairs,
               selLinks,                # sometimes we dont need all the links, just the selected links
               ptList):
    persons         = -1*np.ones((0,19))# each row in this matrix will represent a person
                                        # a person is denoted by 18 possible points (columns)
                                        # and the last column is to the total score for the person
                                        
                                        # the minus 1 is needed to make sure, when there is no value
                                        # filled in a cell, the value of the cell is -1, instead of 1
    for k in range(len(selLinks)):
        if k not in linksNoPairs:
                                        # we start the identification of the belonging of links
                                        # to a person by going through link by link
                                        # For a particular link, say from Neck to Right shoulder,
                                        # we look at all the links of this type
            jointsOf0   = linksWithPairs[k][:,0]    # we take all the first joints (the numbers identified with the joints) in this type of link
            jointsOf1   = linksWithPairs[k][:,1]    # we take all the second joints (the numbers identified with the joints) in this type of link
            (jointIdxOf0,
             jointIdxOf1)=np.array(selLinks[k])

            for i in range(len(jointsOf0)):
                                        # We go through all the first joints in that type of link
                exist   = 0
                personId= -1
                
                for j in range(len(persons)):
                                        # For each first joint, we search through the persons we have
                    if persons[j][jointIdxOf0]  == jointsOf0[i]:
                                        # If the joint is already identified with a person
                                        # we identify the id of the person, and inform its
                                        # existence
                        personId= j
                        exist   = 1
                        break
                        
                if exist:
                                    # If the joint is already with a person,
                                    # add the second joint of that link to the person
                    persons[personId][jointIdxOf1]  = jointsOf1[i]
                                    # Get the score for the joint 
                                    # and get the score for that link, sum them
                                    # with the score in the last column for that person
                    ptScoreOf1                      = ptList[jointsOf1[i].astype(int),2]
                    linkScore                       = linksWithPairs[k][i][2]
                    persons[personId][-1]           +=ptScoreOf1 + linkScore

                else:
                                    # If that joint is not with any person,
                                    # add the first joint to the person,
                                    # add the second joint to the person,
                                    # get the score for both joints, get the link score,
                                    # sum the three scores and put to the last column
                    newPerson       = -1*np.ones(19)
                    newPerson[jointIdxOf0]  = jointsOf0[i]
                    newPerson[jointIdxOf1]  = jointsOf1[i]
                    
                    ptScoreOf0              = ptList[jointsOf0[i].astype(int),2]
                    ptScoreOf1              = ptList[jointsOf1[i].astype(int),2]
                    linkScore               = linksWithPairs[k][i][2]
                    newPerson[-1]           = ptScoreOf0+ptScoreOf1+linkScore
                    
                    persons                 = np.vstack([persons,newPerson])
    return persons

def drawSkeleton(image,
                 selLinks,
                 persons,
                 ptList,
                 lineColours,
                 ptColours,
                 pltThres=4):
    
    for l in range(len(persons)):
        if persons[l,-1] > pltThres:
            for i in range(len(selLinks)):
                link    = persons[l,np.array(selLinks[i])]
                link    = link.astype(int)
                
                if -1 in link:
                    continue
                
                pt0     = np.int32(ptList[link[0],:2])
                pt1     = np.int32(ptList[link[1],:2])
                
                cv2.line(image,
                         (pt0[0],pt0[1]),
                         (pt1[0],pt1[1]),
                         lineColours[i],
                         3,
                         cv2.LINE_AA)
            
            pts     = persons[l]
            for j in range(len(pts)-1):
                if pts[j] == -1:
                    continue
                #print(pts[j])
                pt  = np.int32(ptList[int(pts[j]),:2])
                
                cv2.circle(image,
                           (pt[0],pt[1]),
                           5,
                           [255,255,255],
                           -1,
                           cv2.LINE_AA)
                cv2.circle(image,
                           (pt[0],pt[1]),
                           5,
                           ptColours[j],
                           1,
                           cv2.LINE_AA)

    
    return image



prototxt    = "pose_deploy_linevec.prototxt"
caffemodel  = "pose_iter_440000.caffemodel"

# vs      = cv2.VideoCapture(videopath)
outpath = 'xxxx.MP4'
writer  = None
(W, H)  = (None, None)
while True: 
  (grabbed, frame)     = vs.read() 
  if not grabbed: 
    break 
  if W is None or H is None: 
    (H, W)  = frame.shape[:2]
  output      = frame.copy()
  sklImg   = cv2.cvtColor(img.output, cv2.COLOR_BGR2RGB)
  net         = cv2.dnn.readNetFromCaffe(prototxt, caffemodel) 
  iptH        = 368
  iptW        = int((iptH/H)*W)
  blob        = cv2.dnn.blobFromImage(image=output, scalefactor=1.0/255, size=(iptW,iptH), mean=(0,0,0), swapRB=False, crop=False) 
  net.setInput(blob)
  output      = net.forward() 
  (ptGrp,ptList) = (ptGrp,ptList)  = getAllPoints(cfMaps=output, imgWidth=W, imgHeight=H)
  (linksWithPairs, linksNoPairs)      = getAllLinks(output, links, pafCh, ptGrp, (H,W))
  persons     = getPersons(linksWithPairs, linksNoPairs, links[0:17], ptList)
  skel     = drawSkeleton(sklImg, links[0:17], persons, ptList, colours, ptColours)
  if writer is None:
    fourcc = cv2.VideoWriter_fourcc(*"X264") 
    writer = cv2.VideoWriter(outpath, fourcc, fps, (W, H), True)
  writer.write(output)         
writer.release()
vs.release()