# coding: utf-8

import cv2 as cv
import numpy as np
import math as m
from omrtemplate import get_template_rois
from pyzbar.pyzbar import decode
import re


def get_testset_answers(filename):
    question_dictionary_re = re.compile(r"^\s*questiondictionary")
    question_variants = []
    randans_dictionary_re = re.compile(r"^\s*randomizedanswersdictionary")
    answers_variants = []
    corr_dictonary_re = re.compile(r"^\s*correctiondictionary")
    answer_scores = []
    r_c_func_re = re.compile(r"c\(\s*(\w+\s*(,\s*\w+\s*)+)\s*\)")
    glob_list_container_re = re.compile(r"list\(\s*(.*)\s*\)")
    list_container_re = re.compile(r"list\(\s*(c\(\s*\w+\s*(,\s*\w+\s*)*\s*\)\s*(,\s*c\(\s*\w+\s*(,\s*\w+\s*)*\s*\))*)\s*\)")

    r_script = open(filename, "r")
    line = r_script.readline()
    while line:
        if re.match(question_dictionary_re, line) or \
                re.match(randans_dictionary_re, line) or \
                re.match(corr_dictonary_re, line):
            start_line = line.strip()
            data = start_line
            balance = data.count('(') - data.count(')')
            line = r_script.readline()
            while balance != 0 and line:
                data += line.strip()
                balance = data.count('(') - data.count(')')
                line = r_script.readline()
            content_match = re.search(glob_list_container_re, data)
            if content_match:
                list_content = content_match.group(1)
                if re.match(question_dictionary_re, start_line):
                    for p in re.finditer(r_c_func_re, list_content):
                        question_variants.append((*(int(x.strip()) for x in p.group(1).split(",")),))
                elif re.match(corr_dictonary_re, start_line):
                    for p in re.finditer(r_c_func_re, list_content):
                        answer_scores.append((*(float(x.strip()) for x in p.group(1).split(",")),))
                elif re.match(randans_dictionary_re, start_line):
                    for p in re.finditer(list_container_re, list_content):
                        answers = []
                        for q in re.finditer(r_c_func_re, p.group(1)):
                            answers.append((*(int(x.strip()) for x in q.group(1).split(",")),))
                        answers_variants.append(answers)
        line = r_script.readline()

    return question_variants, answers_variants, answer_scores


def process_answer_sheet(answer_img,
                         template_filename,
                         contour_eps=0.02,
                         compactness_eps=0.1,
                         area_eps=0.7,
                         closeness_eps=1.5,
                         clahe_cliplimit=4.0,
                         blur_ksize=3,
                         laplacian_ksize=5,
                         skip_adaptive_thresh=True,
                         thresh_block_size=33,
                         thresh_bias=3):
    # Convert to gray tones
    img_gray = cv.cvtColor(answer_img, cv.COLOR_BGR2GRAY)

    # Decode the QR-code
    qr_code = decode(img_gray)[0]
    qrdata = qr_code.data.decode('utf-8').split()
    department_id = qrdata[0]
    course_id = qrdata[1]
    exam_date = qrdata[2]
    variants_count = int(qrdata[3])
    questions_count = int(qrdata[4])
    mx = 0
    my = 0
    for p in qr_code.polygon:
        mx = mx + p.x
        my = my + p.y

    mx = mx / 4
    my = my / 4

    height = img_gray.shape[0]
    width = img_gray.shape[1]
    img_corners = {(0, 0): 1, (width, 0): 0, (0, height): 2, (width, height): 3}
    min_dist = m.sqrt(height ** 2 + width ** 2)
    rotation_count = 0
    for k in img_corners:
        dist = m.sqrt((mx - k[0]) ** 2 + (my - k[1]) ** 2)
        if min_dist > dist:
            min_dist = dist
            rotation_count = img_corners[k]

    # Rotate the image such that the page is correctly oriented
    for c in range(rotation_count):
        img_gray = cv.transpose(img_gray)
        img_gray = cv.flip(img_gray, flipCode=1)

    qr_code = decode(img_gray)[0]
    qr_vertices = np.zeros((4, 1, 2), dtype=np.int32)
    for idx in range(4):
        p = qr_code.polygon[idx]
        qr_vertices[idx][0][0] = p.x
        qr_vertices[idx][0][1] = p.y

    # get template informations from the svg file
    page_dims, qr_dims, bounding_rect, answer_bubbles = get_template_rois(template_filename)

    # crop the page to edges, in relation to qr shape
    qr_norms = [np.linalg.norm(d) for d in qr_vertices]
    qr_vertices = np.roll(qr_vertices, -np.argmin(qr_norms), axis=0)
    qr_deltas = qr_vertices - np.roll(qr_vertices, 1, axis=0)
    qr_lengths = [int(np.linalg.norm(d)) for d in qr_deltas]
    new_qr_width = max(qr_lengths)
    new_qr_height = int(qr_dims['height'] * new_qr_width / qr_dims['width'])
    qr_x = qr_vertices[0][0][0]
    qr_y = qr_vertices[0][0][1]
    new_qr_vertices = np.array([
        [qr_x, qr_y],
        [qr_x, qr_y + new_qr_height - 1],
        [qr_x + new_qr_width - 1, qr_y + new_qr_height - 1],
        [qr_x + new_qr_width - 1, qr_y]], dtype="float32")
    reshaped_qr_vertices = np.float32(np.reshape(qr_vertices, (4, 2)))
    # compute the perspective transform matrix and then apply it
    perspective_tm = cv.getPerspectiveTransform(reshaped_qr_vertices, new_qr_vertices)
    img_qr_box = cv.warpPerspective(img_gray,
                                    perspective_tm,
                                    (int(img_gray.shape[1] * 1.1), int(img_gray.shape[0] * 1.1)))
    marg_lx = bounding_rect['x'] / 3
    marg_ly = bounding_rect['y'] / 3
    # marg_rx = (page_dims['width'] - (bounding_rect['x'] + bounding_rect['width'])) / 3
    # marg_ry = (page_dims['height'] - (bounding_rect['y'] + bounding_rect['height'])) / 3
    new_page_xul = int(new_qr_vertices[0][0] - (qr_dims['x'] + marg_lx) * new_qr_width / qr_dims['width'] * 1.2)
    if new_page_xul < 0:
        new_page_xul = 0
    new_page_yul = int(new_qr_vertices[0][1] - (qr_dims['y'] + marg_ly) * new_qr_height / qr_dims['height'])
    if new_page_yul < 0:
        new_page_yul = 0
    new_page_xlr = int(new_qr_vertices[0][0] +
                       (page_dims['width'] - qr_dims['x']) * new_qr_width / qr_dims['width'])
    if new_page_xlr > img_qr_box.shape[1]:
        new_page_xlr = img_qr_box.shape[1]
    new_page_ylr = int(new_qr_vertices[0][1] +
                       (page_dims['height'] - qr_dims['y']) * new_qr_height / qr_dims['height'] * 1.2)
    if new_page_ylr > img_qr_box.shape[0]:
        new_page_ylr = img_qr_box.shape[0]

    # crop the image
    img_qr_box = img_qr_box[new_page_yul:new_page_ylr, new_page_xul:new_page_xlr]

    # create a CLAHE object
    clahe = cv.createCLAHE(clipLimit=clahe_cliplimit)
    # apply CLAHE for better contrast
    img_clahe = clahe.apply(img_qr_box)

    # apply blur
    img_blur = cv.GaussianBlur(img_clahe, (blur_ksize, blur_ksize), 0)

    if not skip_adaptive_thresh:
        # apply adaptive thresholding
        img_th = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, thresh_block_size, thresh_bias)

    # apply laplace edge detection
    img_laplace = cv.Laplacian(img_th if not skip_adaptive_thresh else img_blur, cv.CV_8U, ksize=laplacian_ksize)

    # find contours in the edged image
    _, contours, _ = cv.findContours(img_laplace, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # select the max area contour
    contour = max(contours, key=cv.contourArea)
    perimeter = cv.arcLength(contour, True)
    approx_contour = cv.approxPolyDP(contour, contour_eps * perimeter, True)
    vertices = cv.convexHull(approx_contour, clockwise=True)

    box_frac = bounding_rect['stroke-width'] / bounding_rect['width']
    norms = [np.linalg.norm(d) for d in vertices]
    vertices = np.roll(vertices, -np.argmin(norms), axis=0)
    deltas = vertices - np.roll(vertices, 1, axis=0)
    lengths = [int(np.linalg.norm(d)) for d in deltas]
    dim = max(lengths)
    new_height = dim
    new_width = dim * int(bounding_rect['width'] - bounding_rect['stroke-width']) // int(
        bounding_rect['height'] - bounding_rect['stroke-width'])
    new_vertices = np.array([
        [0, 0],
        [0, new_height - 1],
        [new_width - 1, new_height - 1],
        [new_width - 1, 0]], dtype="float32")
    vertices = np.float32(np.reshape(vertices, (4, 2)))
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(vertices, new_vertices)
    img_box = cv.warpPerspective(img_qr_box, M, (new_width, new_height))
    crop_width = int(np.round(img_box.shape[0] * box_frac))

    translate_tm = np.identity(3)
    translate_tm[0, 2] = -bounding_rect['x']
    translate_tm[1, 2] = -bounding_rect['y']
    scale_tm = np.identity(3)
    scale_factor = img_box.shape[1] / (bounding_rect['width'] + bounding_rect['stroke-width'])
    scale_tm[0, 0] = scale_factor
    scale_tm[1, 1] = scale_factor
    map_tm = scale_tm @ translate_tm

    # map bubbles template coordinates to image coordinates
    initial_r = np.array([answer_bubbles[0]['r'], 0, 1.0], ndmin=2).T
    mapped_r = np.ceil(scale_tm @ initial_r)
    circle_r = mapped_r[0][0]
    circle_area = m.pi * circle_r ** 2
    for idx in range(len(answer_bubbles)):
        initial_coords = np.array([answer_bubbles[idx]['cx'], answer_bubbles[idx]['cy'], 1.0], ndmin=2).T
        mapped_coords = map_tm @ initial_coords
        answer_bubbles[idx]['cx'] = int(mapped_coords[0, 0])
        answer_bubbles[idx]['cy'] = int(mapped_coords[1, 0])
        answer_bubbles[idx]['r'] = circle_r

    # crop the margins
    img_box = img_box[crop_width:img_box.shape[0] - crop_width, crop_width:img_box.shape[1] - crop_width]

    # create a CLAHE object
    clahe_box = cv.createCLAHE(clipLimit=clahe_cliplimit)
    # apply CLAHE for better contrast
    img_box_clahe = clahe_box.apply(img_box)

    # apply blur
    img_box_blur = cv.GaussianBlur(img_box_clahe, (blur_ksize, blur_ksize), 0)

    if not skip_adaptive_thresh:
        # apply adaptive thresholding
        img_box_th = cv.adaptiveThreshold(img_box_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY, thresh_block_size, 1)

    # apply canny edge detection
    img_box_canny = cv.Canny(img_box_th if not skip_adaptive_thresh else img_box_blur, 75, 200)

    # find contours in the boxed image
    _, box_contours, _ = cv.findContours(img_box_canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    by_compactness_contours = []
    for c in box_contours:
        moments = cv.moments(c)
        if moments['mu20'] + moments['mu02'] != 0:
            compactness = moments['m00'] ** 2 / (2 * m.pi * (moments['mu20'] + moments['mu02']))
            if abs(compactness - 1) < compactness_eps:
                by_compactness_contours.append(c)

    by_area_contours = [c for c in by_compactness_contours
                        if abs((circle_area - cv.contourArea(c)) / circle_area) <= area_eps]

    # filter contours by selecting only the contours in expected locations
    # (i.e. by closeness to the expected locations)
    # if several contours are close to the expected locations,
    # only select the largest ones

    for c in by_area_contours:
        moments = cv.moments(c)
        mx = moments['m10'] / moments['m00']
        my = moments['m01'] / moments['m00']
        min_dist = circle_r * (1 + closeness_eps)
        bubble_pair = 0
        for bubble_idx in range(len(answer_bubbles)):
            bubble = answer_bubbles[bubble_idx]
            cx = bubble['cx']
            cy = bubble['cy']
            if min_dist > m.sqrt((cx - mx) ** 2 + (cy - my) ** 2):
                min_dist = m.sqrt((cx - mx) ** 2 + (cy - my) ** 2)
                bubble_pair = bubble_idx
        if abs(min_dist - circle_r) / circle_r < closeness_eps:
            if 'contours' in answer_bubbles[bubble_pair]:
                answer_bubbles[bubble_pair]['contours'].append(c)
            else:
                answer_bubbles[bubble_pair]['contours'] = [c]

    for bubble in answer_bubbles:
        if 'contours' in bubble:
            max_contour = max(bubble['contours'], key=cv.contourArea)
            moments = cv.moments(max_contour)
            mx = int(moments['m10'] / moments['m00'])
            my = int(moments['m01'] / moments['m00'])
            bubble['mx'] = mx
            bubble['my'] = my
            bubble['optimal_contour'] = max_contour

    img_box_result = cv.cvtColor(img_box, cv.COLOR_GRAY2BGR)

    # draw bubbles
    for bubble in answer_bubbles:
        if 'optimal_contour' in bubble:
            cv.circle(img_box_result,
                      (bubble['mx'], bubble['my']), int(circle_r), (0, 255, 255), 1)

    # classify bubbles into 'filled' and 'not filled'
    mask = np.zeros((*img_box_th.shape,) if not skip_adaptive_thresh else (*img_box_blur.shape,), np.uint8)
    bubble_means = []
    for bubble in answer_bubbles:
        if 'optimal_contour' in bubble:
            mask[...] = 0
            cv.circle(mask, (bubble['mx'], bubble['my']), int(circle_r), 255, -1)
            bubble['mean'] = cv.mean(img_box_th if not skip_adaptive_thresh else img_box_blur, mask)[0]
            bubble_means.append(bubble['mean'])
    filled_threshold = np.mean(bubble_means) * 0.9
    for bubble in answer_bubbles:
        if 'optimal_contour' in bubble:
            bubble['is_filled'] = bubble['mean'] < filled_threshold

    # draw bubbles according to the previous classification
    font = cv.FONT_HERSHEY_PLAIN
    font_size = 0.7 * img_box.shape[0] / 1000.0
    bubble_color = (0, 255, 255)
    img_box_result = cv.cvtColor(img_box, cv.COLOR_GRAY2BGR)
    cv.rectangle(img_box_result, (0, 0), (img_box_result.shape[1], img_box_result.shape[0]), (0, 0, 0), 3)

    for bubble in answer_bubbles:
        if 'optimal_contour' in bubble:
            if bubble['is_filled']:
                cv.circle(img_box_result,
                          (bubble['mx'], bubble['my']), int(circle_r), (0, 0, 0), -1)
            cv.circle(img_box_result,
                      (bubble['mx'], bubble['my']), int(circle_r), bubble_color, 1)
            id_parts = bubble['id'].split('-')
            if len(id_parts) > 1:
                text = id_parts[1]
            else:
                text = id_parts[0]
            text = text.upper()
            # get boundary of this text
            text_size = cv.getTextSize(text, font, font_size, 1)
            # get coords based on boundary
            text_x = bubble['mx'] - text_size[0][0] // 2
            text_y = bubble['my'] + text_size[0][1] // 2
            cv.putText(img_box_result, text,
                       (text_x, text_y),
                       font, font_size,
                       (255, 255, 255) if bubble['is_filled'] else (0, 0, 0), 1, cv.LINE_AA)

    # find all answers
    answers = {}
    for bubble in answer_bubbles:
        if 'optimal_contour' in bubble:
            id, pos = bubble['id'].split('-')
            if id in answers:
                answers[id][pos] = bubble['is_filled']
            else:
                answers[id] = {pos: bubble['is_filled']}

    # select only valid answers
    valid_answers = {}
    for key in answers:
        try:
            q = int(key)
            if (q <= questions_count and len(answers[key]) == 4 and
                    list(answers[key].values()).count(True) == 1):
                valid_answers[q] = list(answers[key].keys())[list(answers[key].values()).index(True)]
        except ValueError:
            pass

    # get variant number
    variant = 0
    if ('v' in answers and len(answers['v']) == variants_count and
            list(answers['v'].values()).count(True) == 1):
        variant = int(list(answers['v'].keys())[list(answers['v'].values()).index(True)])

    return img_box_result, qr_code.data.decode('utf-8'), variant, valid_answers
