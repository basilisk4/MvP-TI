import cv2
import numpy as np

def segment(samPredictor,frame,bboxes):
    """Segments objects given their bounding boxes in a cv2 image with SAM 

    Parameters
    ----------
    samPredictor : segment_anything.predictor.SamPredictor
        The segment anything (SAM) predictor
    frame : numpy.ndarray
        A cv2 image 
    bboxes : list of numpy.ndarray
        A List containing the bounding boxes of all objects to segment
    Returns
    -------
    numpy.ndarray
        The segmented frame
    """
    h, w, c = frame.shape
    out_frame=np.zeros([h, w],dtype = np.uint8)
    InferImage= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for bbox in bboxes: 
        # create 1024x1024px window around bbox center for optimal prediction with SAM
        center=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]
        #compute top left corner with awareness for frame boundaries
        tl=np.array([round(center[0])-512,round(center[1])-512])
        if tl[0] < 0: tl[0]=0
        if tl[1] < 0: tl[1]=0
        if tl[0] > w-1024: tl[0]=w-1024
        if tl[1] > h-1024: tl[1]=h-1024

        Crop = InferImage[tl[1]:tl[1]+1023,tl[0]:tl[0]+1023]
        inputBox=[bbox[0]-tl[0],bbox[1]-tl[1],bbox[2]-tl[0],bbox[3]-tl[1]]

        samPredictor.set_image(Crop)
        masks, scores, logits = samPredictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(inputBox),
            multimask_output=False,  # Only return the most confident mask
        )
        mask_bw = ((masks[0]) * 255).astype(np.uint8)
        # Find all connected components (white areas)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bw, connectivity=4)
        # Identify the largest white area (ignoring the background, which is label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # Create a mask for the largest component
        mask_isolated = (labels == largest_label).astype(np.uint8) * 255
        # pad to original size
        pad_tb=[tl[1],h-(tl[1]+1023)]
        pad_lr=[tl[0],w-(tl[0]+1023)]
        mask_bw=np.pad(mask_isolated, (pad_tb,pad_lr), constant_values=((0,0),(0,0)))
        #merge masks
        out_frame=np.maximum(out_frame,mask_bw)
        return out_frame