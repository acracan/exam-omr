
# coding: utf-8

# In[1]:


from lxml import etree
import re
import math as m
import numpy as np


# In[2]:


def get_transform_matrix(transform_string):
    transform_regex = re.compile('(translate|rotate|scale|skewX|skewY|matrix)\(([^()]*)\)')
    transform_matrix = np.identity(3)
    for xformid, args in transform_regex.findall(transform_string):
        args = [float(a) for a in re.split(r'[^0-9eE+.-]+', args)]
        tm = np.identity(3)
        if xformid == 'translate':
            for idx in range(len(args)):
                tm[idx, 2] = args[idx]
        elif xformid == 'scale':
            if len(args) == 1:
                args = args * 2
            for idx in range(len(args)):
                tm[idx, idx] = args[idx]
        elif xformid == 'matrix':
            for idx in range(len(args)):
                tm[idx%2, idx//2] = args[idx]
        elif xformid == 'skewX':
            tm[0, 1] = m.tan(m.radians(args[0]))
        elif xformid == 'skewY':
            tm[1, 0] = m.tan(m.radians(args[0]))
        elif xformid == 'rotate':
            tm[0, 0] =  m.cos(m.radians(args[0]))
            tm[0, 1] = -m.sin(m.radians(args[0]))
            tm[1, 0] =  m.sin(m.radians(args[0]))
            tm[1, 1] =  m.cos(m.radians(args[0]))
            if len(args) > 1:
                pretm = np.identity(3)
                postm = np.identity(3)
                pretm[0, 2] = args[1]
                pretm[1, 2] = args[2]
                postm[0, 2] = -args[1]
                postm[1, 2] = -args[2]
                tm = pretm @ tm @ postm
        
        transform_matrix = transform_matrix @ tm
    return transform_matrix


# In[114]:


def get_template_rois(filename):
    tree = etree.parse(open(filename, "r"))
    root = tree.getroot()
    page_width = root.attrib['width']
    page_height = root.attrib['height']
    # drop units
    units = re.compile(r'[a-zA-Z\s]+')
    try:
        page_width = float(page_width)
    except:
        page_width = float(re.split(units, page_width)[0])
    try:
        page_height = float(page_height)
    except:
        page_height = float(re.split(units, page_height)[0])
    
    qr_root = tree.xpath('//svg:g[starts-with(@inkscape:label, "QR")]',
                           namespaces={'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
                                       'svg': 'http://www.w3.org/2000/svg'})[0]
    qr_width = float(qr_root[0].attrib['width']) - 8*float(qr_root[1].attrib['width'])
    qr_height = float(qr_root[0].attrib['height']) - 8*float(qr_root[1].attrib['height'])
    ctm = np.identity(3)
    ctm_chain = [qr_root] + list(qr_root.iterancestors())
    ctm_chain.reverse()
    for el in ctm_chain:
        if 'transform' in el.attrib:
            ctm = ctm @ get_transform_matrix(el.attrib['transform'])
    initial_qr_orig = np.array([0, 0, 1.0], ndmin=2).T
    initial_qr_width = np.array([qr_width, 0, 1.0], ndmin=2).T
    initial_qr_height = np.array([0, qr_height, 1.0], ndmin=2).T
    mapped_qr_orig = ctm @ initial_qr_orig
    mapped_qr_width = ctm @ initial_qr_width
    mapped_qr_height = ctm @ initial_qr_height
    qr_x = mapped_qr_orig[0][0]
    qr_y = mapped_qr_orig[1][0]
    qr_width = np.linalg.norm(mapped_qr_orig - mapped_qr_width)
    qr_height = np.linalg.norm(mapped_qr_orig - mapped_qr_height)
    rois_root = tree.findall('{*}g[@inkscape:groupmode="layer"][@inkscape:label="ROIs"]',
                             namespaces={'inkscape': 'http://www.inkscape.org/namespaces/inkscape',
                                        'svg': 'http://www.w3.org/2000/svg'})[0]
    ctm = np.identity(3)
    ctm_chain = [rois_root] + list(rois_root.iterancestors())
    ctm_chain.reverse()
    for el in ctm_chain:
        if 'transform' in el.attrib:
            ctm = ctm @ get_transform_matrix(el.attrib['transform'])
    bounding_rect = None
    answer_bubbles = []
    for el in rois_root:
        tag = etree.QName(el).localname
        if tag == 'rect' or tag == 'circle':
            tm = np.copy(ctm)
            if 'transform' in el.attrib:
                tm = tm @ get_transform_matrix(el.attrib['transform'])
            if tag == 'rect':
                rect_attribs = {k:float(el.attrib[k]) for k in ['x', 'y', 'width', 'height']}
                if 'style' in el.attrib:
                    rect_style = dict(item.split(":") for item in el.attrib['style'].split(";"))
                    if 'stroke-width' in rect_style:
                        rect_attribs['stroke-width'] = float(rect_style['stroke-width'])
                    else:
                        rect_attribs['stroke-width'] = 0.0
                initial_ul = np.array([rect_attribs['x'], rect_attribs['y'], 1.0], ndmin=2).T
                initial_width = initial_ul + np.array([rect_attribs['width'], 0, 0], ndmin=2).T
                initial_height = initial_ul + np.array([0, rect_attribs['height'], 0], ndmin=2).T
                initial_stroke = initial_ul + np.array([rect_attribs['stroke-width'], 0, 0], ndmin=2).T
                mapped_ul = tm @ initial_ul
                mapped_width = tm @ initial_width
                mapped_height = tm @ initial_height
                mapped_stroke = tm @ initial_stroke
                rect_attribs['x'] = mapped_ul[0][0]
                rect_attribs['y'] = mapped_ul[1][0]
                rect_attribs['width'] = np.linalg.norm(mapped_width - mapped_ul)
                rect_attribs['height'] = np.linalg.norm(mapped_height - mapped_ul)
                rect_attribs['stroke-width'] = np.linalg.norm(mapped_stroke - mapped_ul)
            elif tag == 'circle':
                circle_attribs = {k:float(el.attrib[k]) for k in ['cx', 'cy', 'r']}
                initial_c = np.array([circle_attribs['cx'], circle_attribs['cy'], 1.0], ndmin=2).T
                initial_r = initial_c + np.array([circle_attribs['r'], 0, 0], ndmin=2).T
                mapped_c = tm @ initial_c
                mapped_r = tm @ initial_r
                circle_attribs['cx'] = mapped_c[0][0]
                circle_attribs['cy'] = mapped_c[1][0]
                circle_attribs['r'] = np.linalg.norm(mapped_r - mapped_c)

                circle_attribs['id'] = el.attrib['id']
                if 'style' in el.attrib:
                    circle_style = dict(item.split(":") for item in el.attrib['style'].split(";"))
                    if 'stroke-width' in circle_style:
                        circle_attribs['stroke-width'] = float(circle_style['stroke-width'])
                    else:
                        circle_attribs['stroke-width'] = 0.0
                answer_bubbles.append(circle_attribs)
    return ({'height': page_height, 'width': page_width}, 
           {'x': qr_x, 'y': qr_y, 'height': qr_height, 'width': qr_width},
            rect_attribs,
            answer_bubbles)

