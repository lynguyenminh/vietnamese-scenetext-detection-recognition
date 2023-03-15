def convert_craft_to_rectangle(list_polygon):
            top_left, top_right, bottom_left, bottom_right = list_polygon
            x_coords = [top_left[0], top_right[0], bottom_right[0], bottom_left[0]]
            y_coords = [top_left[1], top_right[1], bottom_right[1], bottom_left[1]]
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            return list(map(int, [min_x, min_y, max_x, max_y]))

def convert_xyxy2xywh(list_coor):
    x = list_coor[0]
    y = list_coor[1]
    w = list_coor[2] - list_coor[0]
    h = list_coor[3] - list_coor[1]
    return [x, y, w, h]
def convert_xywh2xyxy(list_coor):
    x1 = list_coor[0]
    y1 = list_coor[1]
    x2 = list_coor[0] + list_coor[2]
    y2 = list_coor[1] + list_coor[3]
    return [x1, y1, x2, y2]