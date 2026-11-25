import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
VERTICAL_SKIP = 3
SQ_WIDTH = 28
MAX_ROWS= 6
MAX_COLS = 10
    

def get_letters(fname):
    img, bw, xs, ys, cells, cells_gray =find_grid(fname)
    #print(f"xs: {xs}")
    #print(f"ys: {ys}")
    #cells and cells_gray are a dictionary whose keys are (r,c) positions in the grid
    # and whose values are nparrays
    return img, bw, xs, ys, cells, cells_gray 

def cell_darkness(cell):
    return np.sum(cell > 0) / cell.size

# we are only going to keep the cells likely to be actual grades
# this function identifies the keys (r,c) for those cells
def key_list(cells_d, ys):
    keys_ls = []
    rows_max = min(len(ys) - VERTICAL_SKIP, MAX_ROWS)
    for r in range(1, rows_max+1): # keys start at 1
        for c in range(1, MAX_COLS+1):
            key = (r,c)
            keys_ls.append((r, c))
    return keys_ls #list of tuples (r,c)
    
# use cells_gray to get darknesses for each cell  
def get_darknesses(cells_d, xs, ys):
    darknesses = dict()
    for key in key_list(cells_d, ys):
        darknesses[key] = cell_darkness(cells_d[key])
    return darknesses

def plot_cell_dict(cells_d, ys, selected_column, size=2):
    n_rows = min(len(ys) - VERTICAL_SKIP, MAX_ROWS)
    fig, axes = plt.subplots(1, n_rows,
                figsize=(size * n_rows, size),squeeze=False)
    axes = axes.ravel()
    for ax in axes:         # hide every axis up front
        ax.axis('off')
    for r in range(1, n_rows + 1):# keys start at 1
        cell = cells_d.get((r, selected_column))
        if cell is None:
            continue    # leave missing cells blank
        axes[r - 1].imshow(cell, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
   
def filter_cells(cells_d):
    pass

# convert to 28x28 Image files from larger numpy arrays
# takes in a dictionary of cells where keys are (r,c) position
# does not preserve aspect ratio
def resize_cells(cells_d, ys, crop_margin = 4):
    for key in key_list(cells_d, ys):
        cell = cells_d[key] # a single cell
        # crop the cell to remove artifacts from the sides of the grid cells
        cell_array = np.array(cell[:, crop_margin:-crop_margin])
        #convert to image object so we can use .resize (28x28)
        cell_array = Image.fromarray(cell_array)
        # resize
        cell_array = cell_array.resize((SQ_WIDTH, SQ_WIDTH))
        cells_d[key]  = cell_array
    return cells_d
# saves file like eixhkqlk9.png where 'count' is the number
# helper function for save_cells
def save_with_identifier(img, src_filename, out_dir, count):
    unique_id = src_filename[0:8] # first 8 letters are unique identifier
    f = unique_id + str(count) + ".png"
    img.save(os.path.join(out_dir, f))

# call this save function after resizing and filtering
def save_cells(cells_d, k_ls, src_filename, out_dir):
    count = 0
    for key in k_ls:  # key = tuple like (r,c)
        count+=1
        save_with_identifier(cells_d[key], src_filename, out_dir, count)

# The following functions are from Claude.

def find_grid(path, min_frac=0.5): #main function that detects the grid  
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #crop off the top and bottom ##### 
    img = img[100:-200,100:-200]
    # plt.figure(figsize=(12, 16)) #turn these on if you want to see the images after cropping
    # plt.imshow(img, cmap='gray')
    # plt.show()
    
    bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)

    H, W = img.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 30, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 30))
    h_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    v_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)

    def positions(mask, axis, length):
        sums = mask.sum(axis=axis) / 255
        idx = np.where(sums > length * min_frac)[0]
        groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
        return [int(g.mean()) for g in groups if len(g)]

    ys = positions(h_mask, 1, W)   # horizontal line y-positions
    xs = positions(v_mask, 0, H)   # vertical line x-positions

    v_mask, xs = find_vertical(bw, img.shape[0]) # finds the single vertical lines and corrects for double lines
    h_mask, ys = find_horizontal(bw, bw.shape[1], cover=0.2) # corrects horizontal lines 

    # Mask of the grid lines (slightly dilated so no remnants are left)
    line_mask = cv2.dilate(cv2.bitwise_or(h_mask, v_mask), np.ones((3, 3), np.uint8))

    # Binary version (what you had before), used for the darkness metric
    clean = cv2.subtract(bw, line_mask)

    # Grayscale version: paint the grid lines out of the original gray image
    clean_gray = img.copy()
    clean_gray[line_mask > 0] = 255        # option 1: paint lines white (simple, fast)
    # clean_gray = cv2.inpaint(img, line_mask, 3, cv2.INPAINT_TELEA)   # option 2: fills in using neighboring pixels, usually looks better where strokes cross a line

    pad = 2
    cells = {}
    for r in range(1, len(ys) - 3):
        for c in range(len(xs) - 1):
            sl = (slice(ys[r] + pad, ys[r + 1] - pad),
                  slice(xs[c] + pad, xs[c + 1] - pad))
            cells[(r, c)] = clean[sl]

    pad_gray = 4   # skip the line remnants at the cell edges
    cells_gray = {}
    for r in range(1, len(ys) - 3):
        for c in range(len(xs) - 1):
            cells_gray[(r, c)] = img[ys[r] + pad_gray : ys[r + 1] - pad_gray,
                                 xs[c] + pad_gray : xs[c + 1] - pad_gray]

    return img, bw, xs, ys, cells, cells_gray

def find_vertical(bw, H, gap=8, cover=0.25): #helper functions for find_grid
    # Fuse double lines: close horizontally by about the gap between the pair
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1))
    fused = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    # Keep only long vertical runs
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 40))
    v_mask = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k_open)

    # Per-column coverage as a fraction of page height (0..1)
    prof = (v_mask > 0).sum(axis=0) / H

    # Absolute threshold, so line weight doesn't matter 
    cols = np.where(prof > cover)[0]
    groups = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
    xs = [int(g.mean()) for g in groups if len(g)]
    return v_mask, xs

def find_horizontal(bw, W, gap=1, cover=0.25, open_div=40):
    # Optional: fuse double/thick horizontal lines (raise gap if needed)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap))
    fused = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)

    # Keep only long horizontal runs
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (W // open_div, 1))
    h_mask = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k_open)

    prof = (h_mask > 0).sum(axis=1) / W
    rows = np.where(prof > cover)[0]
    groups = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
    ys = [int(g.mean()) for g in groups if len(g)]
    return h_mask, ys
