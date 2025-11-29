from ultralytics import YOLO
import cv2
import os

def predict_image(model_path, img_path, label_path, threshold=0.5):
    # Load trained model
    model = YOLO(model_path)

    # Load image
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    img_predict = img.copy()

    # Run prediction
    results = model.predict(img_predict)[0]

    # Draw predictions
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(img_predict, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(img_predict, results.names[int(class_id)].upper(),
                        (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show predictions
    cv2.imshow("Predictions", img_predict)

    # Draw ground truth
    img_truth = img.copy()
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        values = line.split()
        label = int(values[0])
        x, y, w, h = map(float, values[1:])
        x1 = int((x - w/2) * W)
        y1 = int((y - h/2) * H)
        x2 = int((x + w/2) * W)
        y2 = int((y + h/2) * H)
        label_name = results.names[label].upper()
        cv2.rectangle(img_truth, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img_truth, label_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Ground Truth", img_truth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example paths (adjust image filename as needed)
    model_path = os.path.join('project', 'Card-Model', 'weights', 'best.pt')
    img_path = os.path.join('dataset', 'test', 'images', '001783412_jpg.rf.71fa67222c66ff7f2f6ef5201d82b8e7.jpg')
    label_path = os.path.join('dataset', 'test', 'labels', '001783412_jpg.rf.71fa67222c66ff7f2f6ef5201d82b8e7.txt')

    predict_image(model_path, img_path, label_path)
