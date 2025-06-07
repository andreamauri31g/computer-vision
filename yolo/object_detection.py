import cv2
import numpy as np

SIZE = (320, 320)
MIN_P = 0.6

def load_yolo():
    net = cv2.dnn.readNet('yolo/yolov3.weights', 'yolo/yolov3.cfg')

    out_layers_names = net.getUnconnectedOutLayersNames()
    names = []
    with open('yolo/coco.names', 'r') as f:
        names = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(names), 3)).astype(int)

    return net, out_layers_names, names, colors

def object_detection(img, net, out_layers_names):
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255, size=SIZE)
    net.setInput(blob)
    output = net.forward(out_layers_names)

    return output

def get_boxes(output, width, height):
    boxes = []
    ids = []
    ps = []

    for i in output:
        for j in i:
            scores = j[5:]
            id = np.argmax(scores)
            p = scores[id]

            if p >= MIN_P:
                center_x = int(j[0] * width)
                center_y = int(j[1] * height)
                w = int(j[2] * width)
                h = int(j[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append((x, y, w, h))
                ids.append(id)
                ps.append(p)

    return boxes, ids, ps

def non_max_suppression(boxes, ps, min_p, threshold):
    boxes_max = []
    b_ids = cv2.dnn.NMSBoxes(boxes, ps, min_p, threshold)

    if len(b_ids) == 0:
        return boxes_max

    for i in b_ids:
        index = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        boxes_max.append(boxes[index])

    return boxes_max

def draw_results(img, boxes, ids, ps=None):
    for i in range(len(boxes)):
        box = boxes[i]
        x, y, w, h = box
        id = ids[i]
        color = colors[id].tolist()
        label = names[id]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        if (ps != None):
            label += f" {ps[i] * 100:.2f}%"

        cv2.putText(img, label, (x, h), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)

    return img

net, out_layers_names, names, colors = load_yolo()
img_path = input("Img path: ")
img = cv2.imread(img_path)
output = object_detection(img, net, out_layers_names)
boxes, ids, ps = get_boxes(output, img.shape[0], img.shape[1])
boxes = non_max_suppression(boxes, ps, MIN_P, 0.3)
img = draw_results(img, boxes, ids, ps=ps)
cv2.imshow("img", img)
cv2.waitKey(0)
