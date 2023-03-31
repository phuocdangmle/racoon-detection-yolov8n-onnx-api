import onnxruntime
import numpy as np
import cv2
from PIL import Image
import uuid

from .utils import IoU, NMS, xywh2xyxy


class RacoonDetection():
    def __init__(self, model_path):
        self.classes_name = ['Racoon']
        self.ort_session = self.load_model(model_path)
        self.input_shape = (1, 3, 640, 640)
        
    
    def load_model(self, model_path):
        ort_session = onnxruntime.InferenceSession(model_path)
        return ort_session
    
    
    def detect(self, img, threshold=0.8):
        img_h, img_w = img.shape[: 2]
        resized_img = cv2.resize(img, self.input_shape[2:])
        
        rescaled_img = resized_img / 255.0
        transposed_img = rescaled_img.transpose(2, 0, 1)
        imgs = np.expand_dims(transposed_img, axis=0).astype(np.float32)

        predicts = self.ort_session.run(None, {"images": imgs})[0]
        
        predicts = np.squeeze(predicts)
        predicts = predicts.T
        
        probs = np.max(predicts[:, 4: ], axis=1)
        predicts = predicts[probs > threshold]
        probs = probs[probs > threshold]
        
        class_ids = np.argmax(predicts[:, 4: ], axis=1)
        
        bboxes = predicts[:, : 4]
        
        rescale_h = img_h / self.input_shape[2]
        rescale_w = img_w / self.input_shape[3]
        rescale = np.array([rescale_w, rescale_h, rescale_w, rescale_h])
        bboxes = (bboxes * rescale)
        bboxes = xywh2xyxy(bboxes).astype(np.int32)
        
        keep_bboxes = NMS(bboxes, probs)
        bboxes = bboxes[keep_bboxes]
        probs = probs[keep_bboxes]
        class_ids = class_ids[keep_bboxes]
        
        results = []
        for bbox, prob, class_id in zip(bboxes, probs, class_ids):
            result = {
                "bbox": list([int(i) for i in bbox]),
                "prob": float(prob),
                "class": self.classes_name[class_id]
            }
            results.append(result)
        return results
    
    
    def draw_box_detect(self, img, threshold=0.8):
        results = self.detect(img, threshold)
        
        for result in results:
            bbox = result["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        
        return img
        
        
    
    def __call__(self, img, threshold=0.8):
        return self.detect(img, threshold)
        
        
        
if __name__ == "__main__":
    # racoon_detection = RacoonDetection("src/racoon_detection.onnx")

    # img = cv2.imread("img.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # print(racoon_detection.detect(img))
    
    # print(racoon_detection(img))
    
    # print(racoon_detection.draw_box_detect(img))

    # # Test IoU
    # box = np.array([511, 41, 577, 76])
    # boxes = np.array([[544, 59, 610, 94]])

    # print(IoU(box, boxes)) # -> [0.13821138]
    
    pass