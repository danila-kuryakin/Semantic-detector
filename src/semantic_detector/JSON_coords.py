import cv2
import json

i = 0
m = 0


def save_coord(x, y):
    global i

    if i == 0:
        data = {"values": [{i: {"x": x, "y": y}}]}
        with open("data_file_" + str(m) + ".json", "w") as write_file:
            json.dump(data, write_file)
    else:
        data = ({i: {"x": x, "y": y}},)
        with open("data_file_" + str(m) + ".json") as write_file:
            data_j = json.load(write_file)

        data_j["values"] += list(data)

        with open("data_file_" + str(m) + ".json", 'w') as write_file:
            json.dump(data_j, write_file, indent=4)
    i = i + 1


# display coordinates
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        # displaying on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    0.7, (255, 0, 0), 2)
        cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image', img)
        # save to json file
        save_coord(x, y)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        global m
        global i
        i = 0
        m = m + 1
        print("data_file_" + str(m) + ".json will be created")


if __name__ == "__main__":
    global a
    img = cv2.imread('../resources/dataset/BirdView/008---lean/west_1.jpg', 1)
    cv2.imshow('image', img)

    # waiting click
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
