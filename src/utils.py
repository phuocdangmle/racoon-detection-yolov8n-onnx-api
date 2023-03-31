import numpy as np


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def NMS(bboxes, probs, iou_threshold=0.3):
    sorted_indices = np.argsort(probs)[::-1]
    
    keep_bboxes = []
    
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_bboxes.append(box_id)
        
        IoUs = IoU(bboxes[box_id, :], bboxes[sorted_indices[1: ], :])
        
        keep_indices = np.where(IoUs < iou_threshold)[0] + 1
        
        sorted_indices = sorted_indices[keep_indices]
        
    return keep_bboxes
    

def IoU(box, boxes):
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])
    
    intersection_area = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
        
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    
    boxes_width = (boxes[:, 2] - boxes[:, 0])
    boxes_height = (boxes[:, 3] - boxes[:, 1])    
    boxes_area = boxes_width * boxes_height
    
    union_area = boxes_area + box_area - intersection_area
    
    IoU = intersection_area / union_area
    
    return IoU