"""
Copyright
2020-11-11
仅用于个人学习、交流使用，禁止用于商业目的。
Simple Threshold Gray使用的算法参考https://blog.csdn.net/wizardforcel/article/details/104837872
Gaussian Filter及Gaussian Filter Gray算法参考https://www.cnblogs.com/cvdream/p/9584030.html

使用说明
选择图片文件，勾选“对图片进行预处理”，选择合适的算法及参数，点击“生成PDF文件”并设置输出文件的路径即可。
如果不需要处理、直接生成pdf文档，则取消勾选“对图片进行预处理”的复选框。

已知问题
1)调整文件顺序时背景色无法消除，并且有时会提示选择了多个文件。
2)如果选择的文件较多，超出了当前窗口的范围，上下移动时不会自动刷新。
上述问题在file_selector模块中；不影响功能。待后续搞清楚wx.ListCtrl的使用方法后，也许可以修改。

2020-12-31
v1.1
修复了v1.0中的已知问题1)。

2021-01-01
v1.2
1) 修复了v1.0中的已知问题2)。
2) 修复了len(shape)为2的图片无法处理的问题。
3) 优化了显示：重新选择新的一组图片时，清除原图片显示；deselect所有图片时，清除图片显示。
4) 将“图片处理”和“pdf文件生成”拆分为2个独立的功能。

2021-01-01
v1.3
1) 修复combine2Pdf中的错误。
2) 增加左键双击图片、右键单击item显示原图的功能。

2021-01-19
v1.4
1) 优化了v1.3中增加的显示图片的功能。
2) 调整代码结构，增加EQU_HIST处理的下拉菜单（实现尚未填入）。

2021-01-24
v1.5
1) 增加Histogram Equalization的功能。
2) 增加Local Histogram Equalization功能。
3) 增加自动生成非重名文件夹的功能。
4) 优化代码结构。

2021-02-22
v1.6
1) 更新histogram_img，使之能处理多余3个channel的图片。
"""
import wx

from PIL import Image
import numpy as np
import cv2 as cv
from scipy import signal 
import os
import shutil
import time
import math

from file_selector.file_selector import FileSelector
from file_selector.file_selector import SIZER_BORDER
from file_selector.file_selector import GRID_DIZER_VGAP
from file_selector.file_selector import GRID_DIZER_HGAP
from file_selector.file_selector import EVT_LISTCTRL_CNT_UPDATED_IND

__version__ = "1.6"

IMG_FILE_EXT = "Image files(*.jpg;*.bmp;*.png;*.tif)|*.jpg;*.bmp;*.png;*.tif"
PDF_FILE_EXT = "PDF file(*.pdf)|*.pdf"

IMG_DISPLAY_SIZE_W = 210
IMG_DISPLAY_SIZE_H = 420

FUNCTION_TITLE = "Enhance Images"
MIN_SIZE_OF_WINDOW = wx.Size(800, 300)
DIP_ALG_SIMPLE_THRES_GRAY = 0
DIP_ALG_SIMPLE_THRES_GRAY_STR = 'Simple Threshold Gray'
DIP_ALG_GAUSSIAN_FILTER_GRAY = 1
DIP_ALG_GAUSSIAN_FILTER_GRAY_STR = 'Gaussian Filter Gray'
DIP_ALG_GAUSSIAN_FILTER = 2
DIP_ALG_GAUSSIAN_FILTER_STR = 'Gaussian Filter'
DIP_ALG_COMMENT_STR = '模式说明：'
DIP_ALG_HIST_EQU = 3
DIP_ALG_HIST_EQU_STR = 'HistogramEqualization'
DIP_ALG_LOCAL_HIST_EQU = 4
DIP_ALG_LOCAL_HIST_EQU_STR = 'LocalHistogramEqualization'
dip_alg_dict = {
    DIP_ALG_SIMPLE_THRES_GRAY_STR : {'id' : DIP_ALG_SIMPLE_THRES_GRAY,
                                     'comment' : DIP_ALG_COMMENT_STR + '输出黑白图片。速度最快，效果稍差。'},
    DIP_ALG_GAUSSIAN_FILTER_GRAY_STR : {'id' : DIP_ALG_GAUSSIAN_FILTER_GRAY,
                                        'comment' : DIP_ALG_COMMENT_STR + '输出黑白图片。速度稍慢，效果较好。'},
    DIP_ALG_GAUSSIAN_FILTER_STR : {'id' : DIP_ALG_GAUSSIAN_FILTER,
                                   'comment' : DIP_ALG_COMMENT_STR + '与原始图片色彩相同。速度最慢，效果较好。'},
    DIP_ALG_HIST_EQU_STR : {'id' : DIP_ALG_HIST_EQU,
                            'comment' : '仅能处理UINT8类型的图片'},
    DIP_ALG_LOCAL_HIST_EQU_STR : {'id' : DIP_ALG_LOCAL_HIST_EQU,
                                  'comment': '仅能处理UINT8类型的图片'},
    }

def get_alg_str(alg_id):
    for key in dip_alg_dict.keys():
        if alg_id == dip_alg_dict[key]['id']: return key
    return ""

def read_ch_path_img(img_fn):
    return cv.imdecode(np.fromfile(img_fn, dtype=np.uint8), cv.IMREAD_UNCHANGED)

def write_ch_path_img(img_fn, img, ext = '.jpg'):
    ret, img_arr = cv.imencode(ext, img)
    img_arr.tofile(img_fn)
    return ret
    
def add_apx_to_bn(fpn, apx):
    fp_bn, ext = os.path.splitext(fpn)
    return fp_bn + apx + ext

def mkdir_avoid_dup(path_pre, curr_path) -> str:
    """
    基于full_path_base ( = path_pre + "\\" + curr_path) 创建文件夹。如果有重复的，在后面添加"_000“后缀。
    path_pre结尾不带字符"\"。
    后缀最大为999（MAX_MKDIR_TRY - 1）。
    返回生成的curr_path路径名称。
    """
    MAX_MKDIR_TRY = 1000
    full_path_base = path_pre + "\\" + curr_path
    if not hasattr(mkdir_avoid_dup, 'path_cnt_rec'): mkdir_avoid_dup.path_cnt_rec = dict()
    if not (full_path_base in mkdir_avoid_dup.path_cnt_rec.keys()):
        mkdir_avoid_dup.path_cnt_rec[full_path_base] = 0

    apx_format_str = '{:03d}' if "" == curr_path else '_{:03d}'
    cnt_apx = mkdir_avoid_dup.path_cnt_rec[full_path_base]
    output_full_path = full_path_base if 0 == cnt_apx else full_path_base + apx_format_str.format(cnt_apx)
    cnt_apx = (cnt_apx + 1) % MAX_MKDIR_TRY
    while(cnt_apx < MAX_MKDIR_TRY and os.path.exists(output_full_path)):
        output_full_path = full_path_base + apx_format_str.format(cnt_apx)
        cnt_apx += 1

    if cnt_apx >= MAX_MKDIR_TRY:
        cnt_apx = 0
        output_full_path = full_path_base
    mkdir_avoid_dup.path_cnt_rec[full_path_base] = cnt_apx
    if(os.path.exists(output_full_path)):
        shutil.rmtree(output_full_path)
        time.sleep(0.5)
    os.mkdir(output_full_path)
    return output_full_path[output_full_path.rfind("\\") + 1:]

DIP_ALG_SIMPLE_THRES_BETA_MIN = 0.1
DIP_ALG_SIMPLE_THRES_BETA_MAX = 1
DIP_ALG_SIMPLE_THRES_BETA_DEF = 0.9
DIP_ALG_SIMPLE_THRES_BETA_INC = 0.1
DIP_ALG_SIMPLE_THRES_KRNL_MIN = 3
DIP_ALG_SIMPLE_THRES_KRNL_MAX = 15
DIP_ALG_SIMPLE_THRES_KRNL_DEF = 5
DIP_ALG_SIMPLE_THRES_BLACK_THRES_MIN = 0
DIP_ALG_SIMPLE_THRES_BLACK_THRES_MAX = 255
DIP_ALG_SIMPLE_THRES_BLACK_THRES_DEF = 30
def adaptive_thres(img, beta=DIP_ALG_SIMPLE_THRES_BETA_DEF, win=DIP_ALG_SIMPLE_THRES_KRNL_DEF, black_thres = DIP_ALG_SIMPLE_THRES_BLACK_THRES_DEF):
    if win % 2 == 0: win = win - 1
    # 边界的均值有点麻烦
    # 这里分别计算和和邻居数再相除
    kern = np.ones([win, win])
    sums = signal.correlate2d(img, kern, 'same')
    cnts = signal.correlate2d(np.ones_like(img), kern, 'same')
    means = sums // cnts
    # 如果直接采用均值作为阈值，背景会变花
    # 但是相邻背景颜色相差不大
    # 所以乘个系数把它们过滤掉
    #img = np.where(img < means * beta, 0, 255)
    img = np.where(np.logical_or(img < means * beta, img <= black_thres), 0, 255)
    return img


def dip_alg_gaussian_filter_en(input_img):
    r = int(np.sqrt(input_img.shape[0] * input_img.shape[1] / 32) + 1)
    ksize = r + r + 1
    blurred_img = cv.GaussianBlur(input_img, (ksize, ksize), 0)
    result_img = cv.divide(input_img, blurred_img, scale = 255)
    return result_img

def combine2Pdf(imgFiles, pdfFile):
    sources = []
    output = Image.open(imgFiles[0])
    if output.mode != "RGB": output = output.convert("RGB")
    for idx in list(range(len(imgFiles)))[1:]:
        file = imgFiles[idx]
        imgFile = Image.open(file)
        if imgFile.mode != "RGB":
            imgFile = imgFile.convert( "RGB" )
        sources.append(imgFile)
    output.save(pdfFile, "pdf", save_all=True, append_images=sources)
    #wx.MessageBox(str(imgFiles) + "\n" + pdfFile)

def color2Gray(img, color_space = 0):
    #wx.MessageBox('img.shape=' + str(img.shape) + '\nlen(shape)=' + str(len(img.shape)))
    if len(img.shape) == 2:
        return img
    elif img.shape[2] == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif img.shape[2] == 4:
        return cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    else:
        #can't process this img. To be implemented in future.
        return img

def is_img_single_channel(img):
    if len(img.shape) == 2 or img.shape[2] == 1: return True
    else: return False

def histogram_img(ori_img, bg_color = 255):
    use_case_mesage = "图片需满足如下条件：单channel，或者，len(shape)为3且shape[2]>=3"
    MAX_HISTOGRAM_HEIGHT = 300
    TOP_DISPLAY_MARGIN = 0.1
    MAX_BINS = 256 #UINT8
    LEFT_DISPLAY_MARGIN = 0.1
    RIGHT_DISPLAY_MARGIN = 0.1
    MIN_ELEMENT_VALUE = 0 #UINT8
    MAX_ELEMENT_VALUE = 256 #UINT8
    
    if is_img_single_channel(ori_img):
        histo_colors = ((255 - bg_color, 255 - bg_color, 255 - bg_color),)
    else:
        assert len(ori_img.shape) == 3 and ori_img.shape[2] >= 3, use_case_mesage
        col_list = [(255 - bg_color, 255 - bg_color, 255 - bg_color) for i in range(ori_img.shape[2] - 3)]
        histo_colors = ((255,0,0), (0,255,0), (0,0,255)) + tuple(col_list)

    hist_data_list = []
    max_height = MAX_HISTOGRAM_HEIGHT
    for ch, histo_color in enumerate(histo_colors):
        hist_data = cv.calcHist([ori_img], [ch], None, [MAX_BINS], [MIN_ELEMENT_VALUE, MAX_ELEMENT_VALUE])
        max_height = min(max_height, np.amax(hist_data))
        hist_data_list.append(hist_data)

    max_height = int(max_height)
    canvas_height = math.ceil(max_height * (1 + TOP_DISPLAY_MARGIN))
    canvas_width = math.ceil(MAX_BINS * (1 + LEFT_DISPLAY_MARGIN + RIGHT_DISPLAY_MARGIN))
    draw_orig_x = math.ceil(MAX_BINS * LEFT_DISPLAY_MARGIN)
    draw_orig_y = 0
    canvas = np.full((canvas_height, canvas_width, 3), bg_color)

    x_coords = np.arange(MAX_BINS).reshape((MAX_BINS,1)) + draw_orig_x
    for ch, line_color in enumerate(histo_colors):
        h_d = hist_data_list[ch]
        cv.normalize(h_d, h_d, 0, max_height, cv.NORM_MINMAX)
        h_d = np.int32(np.around(h_d))
        pts = np.column_stack((x_coords, h_d))
        cv.polylines(canvas, [pts], False, line_color)
    result_img = np.flipud(canvas)
    return result_img

LOCAL_HIST_EQUAL_KRN_SIZE_DEF = 3
LOCAL_HIST_EQUAL_KRN_SIZE_MIN = 3
LOCAL_HIST_EQUAL_KRN_SIZE_MAX = 15
def local_histogram_equalization(ori_img, win = LOCAL_HIST_EQUAL_KRN_SIZE_DEF):
    if win % 2 == 0: win = win - 1
    assert win > 1, 'The minmum kernal size is 3.'
    #assume ori data is uint8 or uint32 so that we can fill the edge with max_data + 1
    assert ori_img.dtype == 'uint8' or ori_img.dtype == 'uint16', 'Image data can only be uint8 or uint16.'

    if ori_img.dtype == 'uint8':
        L = 255
        type_func = np.uint8
    else:
        L = 65535
        type_func = np.uint16

    ori_img_chs = cv.split(ori_img)
    equ_size_m = np.zeros(ori_img_chs[0].shape)
    equ_size_n = np.zeros(ori_img_chs[0].shape)
    for m in range(ori_img.shape[0]):
        for n in range(ori_img.shape[1]):
            if m < (win - 1)/2: equ_size_m[m,n] = m
            elif m > ori_img.shape[0] - (win + 1)/2: equ_size_m[m,n] = ori_img.shape[0] - 1 - m
            else : equ_size_m[m,n] = (win - 1)/2
            
            if n < (win - 1)/2: equ_size_n[m,n] = n
            elif n > ori_img.shape[1] - (win + 1)/2: equ_size_n[m,n] = ori_img.shape[1] - 1 - n
            else : equ_size_n[m,n] = (win - 1)/2

    equ_size_mn = np.int32((equ_size_m + (win + 1)/2) * (equ_size_n + (win + 1)/2))
    
    loc_equ_chs = []
    for ch, ori_img_ch in enumerate(ori_img_chs):
        accum_cnt = np.int32(np.ones(ori_img_ch.shape))
        cur_ch_data = np.int32(ori_img_ch) 
        max_to_fill = np.amax(cur_ch_data) + 1
        for ki_x in range(win):
            for ki_y in range(win):
                if ki_x == ki_y: continue
                krn = np.int32(np.zeros((win,win)))
                krn[ki_x, ki_y] = 1
                cur_pos_data = signal.correlate2d(ori_img_ch, krn, 'same', 'fill', max_to_fill)
                curr_pos_diff_data = cur_pos_data - cur_ch_data
                accum_cnt = accum_cnt + (curr_pos_diff_data <= 0)
        loc_equ_ch = (L-1) * accum_cnt / equ_size_mn
        loc_equ_ch = type_func(loc_equ_ch)
        loc_equ_chs.append(loc_equ_ch)
    loc_equ_img = cv.merge(loc_equ_chs)
    return loc_equ_img

class GrayImgEnhanceSharp(FileSelector):
    PRE_PROCESS_IMG_STR = "处理图片"
    GEN_PDF_FILE_STR = '生成pdf文件'
    KRNL_CTRL_LABEL = 'kernel大小(必须为奇数)：'
    def __init__(self, file_ext = IMG_FILE_EXT):
        FileSelector.__init__(self, None, FUNCTION_TITLE, file_ext)
        #self.SetMinSize(MIN_SIZE_OF_WINDOW)

        self.work_btn = wx.Button(self, -1, "开始工作")
        self.img_process_flag_check = wx.CheckBox(self, wx.ID_ANY, GrayImgEnhanceSharp.PRE_PROCESS_IMG_STR)
        self.img_process_flag_check.SetValue(True)
        self.img_process_flag = True
        self.gen_pdf_flag_check = wx.CheckBox(self, wx.ID_ANY, GrayImgEnhanceSharp.GEN_PDF_FILE_STR)
        self.gen_pdf_flag_check.SetValue(True)
        self.gen_pdf_flag = True
        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_sizer.Add(self.work_btn, flag = wx.ALIGN_BOTTOM | wx.ALL, border = SIZER_BORDER)
        self.btn_sizer.Add(self.img_process_flag_check, flag = wx.ALIGN_BOTTOM | wx.ALL, border = SIZER_BORDER)
        self.btn_sizer.Add(self.gen_pdf_flag_check, flag = wx.ALIGN_BOTTOM | wx.ALL, border = SIZER_BORDER)
        
        self.dip_alg_choice = wx.Choice(self, choices = list(dip_alg_dict.keys()))
        def_choice_idx = self.dip_alg_choice.FindString(DIP_ALG_SIMPLE_THRES_GRAY_STR)
        assert wx.NOT_FOUND != def_choice_idx
        self.dip_alg_choice.SetSelection(def_choice_idx)
        self.dip_alg_id = dip_alg_dict[DIP_ALG_SIMPLE_THRES_GRAY_STR]['id']
        self.dip_alg_sizer = wx.BoxSizer(wx.VERTICAL)
        self.alg_sel_sizer = wx.BoxSizer()
        self.alg_sel_sizer.Add(wx.StaticText(self, label = "处理方式："), flag = wx.ALIGN_TOP | wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.alg_sel_sizer.Add(self.dip_alg_choice, flag = wx.ALIGN_TOP | wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.alg_comment_area = wx.StaticText(self, label = dip_alg_dict[DIP_ALG_SIMPLE_THRES_GRAY_STR]['comment'])
        self.alg_comment_sizer = wx.BoxSizer()
        self.alg_comment_sizer.Add(self.alg_comment_area, flag = wx.ALIGN_TOP | wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.dip_alg_sizer.AddMany((self.alg_sel_sizer,self.alg_comment_sizer))

        self.alg_simple_thres_krnl_ctrl = wx.SpinCtrl(self, min = DIP_ALG_SIMPLE_THRES_KRNL_MIN, max = DIP_ALG_SIMPLE_THRES_KRNL_MAX,
                                                      initial = DIP_ALG_SIMPLE_THRES_KRNL_DEF)
        self.alg_simple_thres_krnl_curr_val = DIP_ALG_SIMPLE_THRES_KRNL_DEF
        self.alg_simple_thres_krnl_sizer = wx.BoxSizer()
        self.alg_simple_thres_krnl_sizer.Add(wx.StaticText(self, label = GrayImgEnhanceSharp.KRNL_CTRL_LABEL), flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.alg_simple_thres_krnl_sizer.Add(self.alg_simple_thres_krnl_ctrl, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        
        self.alg_simple_thres_black_thres_ctrl = wx.SpinCtrl(self, min = DIP_ALG_SIMPLE_THRES_BLACK_THRES_MIN, max = DIP_ALG_SIMPLE_THRES_BLACK_THRES_MAX,
                                                       initial = DIP_ALG_SIMPLE_THRES_BLACK_THRES_DEF)
        self.alg_simple_thres_black_thres_sizer = wx.BoxSizer()
        self.alg_simple_thres_black_thres_sizer.Add(wx.StaticText(self, label = '黑色像素最大值({}~{})：'.format(DIP_ALG_SIMPLE_THRES_BLACK_THRES_MIN, DIP_ALG_SIMPLE_THRES_BLACK_THRES_MAX)),
                                                    flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.alg_simple_thres_black_thres_sizer.Add(self.alg_simple_thres_black_thres_ctrl, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        
        self.alg_simple_thres_beta_ctrl = wx.SpinCtrlDouble(self, min = DIP_ALG_SIMPLE_THRES_BETA_MIN, max = DIP_ALG_SIMPLE_THRES_BETA_MAX,
                                                            initial = DIP_ALG_SIMPLE_THRES_BETA_DEF, inc = DIP_ALG_SIMPLE_THRES_BETA_INC)
        self.alg_simple_thres_beta_sizer = wx.BoxSizer()
        self.alg_simple_thres_beta_sizer.Add(wx.StaticText(self, label = 'Beta({}~{})：'.format(DIP_ALG_SIMPLE_THRES_BETA_MIN, DIP_ALG_SIMPLE_THRES_BETA_MAX)),
                                             flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.alg_simple_thres_beta_sizer.Add(self.alg_simple_thres_beta_ctrl, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        
        self.alg_simple_thres_opt_sizer = wx.BoxSizer(wx.VERTICAL)
        self.alg_simple_thres_opt_sizer.AddMany((self.alg_simple_thres_krnl_sizer, self.alg_simple_thres_black_thres_sizer, self.alg_simple_thres_beta_sizer))

        self.alg_local_hist_equ_setting_sizer = wx.BoxSizer()
        self.alg_local_hist_equ_krnl_ctrl = wx.SpinCtrl(self, min = LOCAL_HIST_EQUAL_KRN_SIZE_MIN, max = LOCAL_HIST_EQUAL_KRN_SIZE_MAX,
                                                        initial = LOCAL_HIST_EQUAL_KRN_SIZE_DEF)
        self.alg_local_hist_equ_setting_sizer.Add(wx.StaticText(self, label = GrayImgEnhanceSharp.KRNL_CTRL_LABEL), \
                                                  flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.alg_local_hist_equ_setting_sizer.Add(self.alg_local_hist_equ_krnl_ctrl, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.alg_local_hist_equ_krnl_curr_val = LOCAL_HIST_EQUAL_KRN_SIZE_DEF
        
        self.bg_whiten_alg_setting_sizer = wx.BoxSizer(wx.VERTICAL)
        #self.bg_whiten_alg_setting_sizer.AddMany((self.alg_sel_sizer, self.alg_comment_sizer, self.alg_simple_thres_krnl_sizer, self.alg_simple_thres_black_thres_sizer))
        self.bg_whiten_alg_setting_sizer.AddMany((self.alg_simple_thres_opt_sizer,))

        self.img_info_text_ctrl = wx.StaticText(self)
        self.img_display_ctrl = wx.StaticBitmap()
        self.img_display_ctrl_bitmap = None
        self.img_display_ctrl_fpn = None
        self.img_display_ctrl.Bind(wx.EVT_LEFT_DCLICK, self.OnStaticBitmapDblClk, self.img_display_ctrl)
        self.img_sizer = wx.BoxSizer(wx.VERTICAL)
        self.img_sizer.AddMany((self.img_info_text_ctrl, self.img_display_ctrl))
        
        self.func_sizer = wx.BoxSizer(wx.VERTICAL)
        self.func_sizer.Add(self.btn_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.func_sizer.Add(self.dip_alg_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.func_sizer.Add(self.bg_whiten_alg_setting_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.func_sizer.Add(self.alg_local_hist_equ_setting_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.func_sizer.Add(self.img_sizer, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.AddFuncArea(self.func_sizer)
        self.CtrlsDisplayUpdate()
        
        self.Bind(wx.EVT_CHECKBOX, self.OnImgProcessCheck, self.img_process_flag_check)
        self.Bind(wx.EVT_CHECKBOX, self.OnGenPdfCheck, self.gen_pdf_flag_check)
        self.Bind(wx.EVT_BUTTON, self.OnWorkButton, self.work_btn)
        self.Bind(wx.EVT_CHOICE, self.OnDIPAlgSelect, self.dip_alg_choice)
        self.Bind(wx.EVT_SPINCTRL, self.OnSimpleThresKrnlCtrl, self.alg_simple_thres_krnl_ctrl)
        self.Bind(wx.EVT_SPINCTRL, self.OnSimpleThresBlackThres, self.alg_simple_thres_black_thres_ctrl)
        self.Bind(wx.EVT_SPINCTRL, self.OnLocalHistEquKrnlCtrl, self.alg_local_hist_equ_krnl_ctrl)
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnFileItemClickedInList, self.file_list_ctrl)
        self.Bind(wx.EVT_LIST_ITEM_DESELECTED, self.OnDeselected)
        self.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.OnStaticBitmapDblClk)

        self.Bind(EVT_LISTCTRL_CNT_UPDATED_IND, self.OnListCtrlCntUpdated)

        self.SetTitle(os.path.splitext(os.path.basename(__file__))[0] + " v" + __version__)

    def CtrlsDisplayUpdate(self):
        if (DIP_ALG_SIMPLE_THRES_GRAY == self.dip_alg_id or \
            DIP_ALG_GAUSSIAN_FILTER_GRAY == self.dip_alg_id or \
            DIP_ALG_GAUSSIAN_FILTER == self.dip_alg_id):
            self.gen_pdf_flag_check.Show(True)
            self.img_process_flag_check.Show(True)
            if DIP_ALG_SIMPLE_THRES_GRAY == self.dip_alg_id:
                #self.bg_whiten_alg_setting_sizer.Layout()
                self.bg_whiten_alg_setting_sizer.Show(self.alg_simple_thres_opt_sizer, recursive = True)
            else:
                self.bg_whiten_alg_setting_sizer.Hide(self.alg_simple_thres_opt_sizer, recursive = True)
            self.func_sizer.Hide(self.alg_local_hist_equ_setting_sizer, recursive = True)
        else:
            self.func_sizer.Hide(self.bg_whiten_alg_setting_sizer, recursive = True)
            self.gen_pdf_flag_check.Hide()

            if DIP_ALG_HIST_EQU == self.dip_alg_id:
                self.img_process_flag_check.Show(True)
                self.func_sizer.Hide(self.alg_local_hist_equ_setting_sizer, recursive = True)
            else:
                self.img_process_flag_check.Hide()
                if DIP_ALG_LOCAL_HIST_EQU == self.dip_alg_id:
                    self.func_sizer.Show(self.alg_local_hist_equ_setting_sizer, recursive = True)
                    
        self.func_sizer.Fit(self)
        self.RefreshSizeDisplay()

    def ImgInfoStr(self, img) -> str:
        img_info_str = 'shape:' + str(img.shape) + '  ' + 'data type:' + str(img.dtype)
        return img_info_str
    
    def OnListCtrlCntUpdated(self, evt):
        if(evt.src == "SelectNewFiles"): self.func_sizer.Hide(self.img_sizer, recursive = True)
        FileSelector.OnListCtrlCntUpdated(self, evt)

    def OnDeselected(self,event):
        if self.file_list_ctrl.GetSelectedItemCount() <= 0:
            self.func_sizer.Hide(self.img_sizer, recursive = True)
            self.RefreshSizeDisplay()
        
    def OnFileItemClickedInList(self, evt):
        #wx.MessageBox("select")
        sel_idx = self.file_list_ctrl.GetFirstSelected()
        if sel_idx < 0 : return

        if self.file_list_ctrl.GetSelectedItemCount() > 1 :
            wx.MessageBox("Only 1 item can be selected at one time!")
            return
        img_file_name = self.selected_files[sel_idx] #self.file_list_ctrl.GetItemText(sel_idx, 1)
        img_file_name = self.selected_path + '\\' + img_file_name

        img = read_ch_path_img(img_file_name)
        img_info_str = self.ImgInfoStr(img)
        self.img_info_text_ctrl.SetLabel(img_info_str)
        
        wx_img = wx.Image(img_file_name)
        scaled_wx_img = wx_img.Scale(IMG_DISPLAY_SIZE_W, IMG_DISPLAY_SIZE_H)
        wx_bitmap = scaled_wx_img.ConvertToBitmap()
        if None == self.img_display_ctrl_bitmap :
            self.img_display_ctrl.Create(self, bitmap = wx_bitmap)
        else:
            self.img_display_ctrl.SetBitmap(wx_bitmap)
        #self.img_display_ctrl.Show()
        self.func_sizer.Show(self.img_sizer, recursive = True)
        self.img_display_ctrl_bitmap = self.img_display_ctrl.GetBitmap()
        self.img_display_ctrl_fpn = img_file_name
        self.RefreshSizeDisplay()

    def OnStaticBitmapDblClk(self,evt):
        if None == self.img_display_ctrl_fpn:
            wx.MessageBox("None")
            return
        #wx.MessageBox(self.img_display_ctrl_fpn)
        #cv_img = cv.imread(self.img_display_ctrl_fpn)
        cv_img = read_ch_path_img(self.img_display_ctrl_fpn)
        #wx.MessageBox(str(cv_img.shape))
        img_win_name = self.img_display_ctrl_fpn[self.img_display_ctrl_fpn.rfind('\\') + 1:]
        cv.namedWindow(img_win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
        cv.imshow(img_win_name, cv_img)

    def SpinCtrlDblSpin(self, ctrl_val, record_val, even_odd :'0 means even, 1 means odd' = 1) -> int:
        """
        Spin控件每次增加或减少2
        """
        if ctrl_val % 2 != even_odd :
            if ctrl_val > record_val:
                ctrl_val = ctrl_val + 1
            elif ctrl_val < record_val:
                ctrl_val = ctrl_val - 1
        return ctrl_val

    def OnSimpleThresKrnlCtrl(self, evt):
        #wx.MessageBox(str(evt.GetRefData()))
        val = evt.GetPosition()
        ctrl_val = self.SpinCtrlDblSpin(val, self.alg_simple_thres_krnl_curr_val, 1)
        self.alg_simple_thres_krnl_curr_val = ctrl_val
        self.alg_simple_thres_krnl_ctrl.SetValue(ctrl_val)
            
    def OnSimpleThresBlackThres(self, evt):
        #wx.MessageBox("b " + str(evt.GetPosition()))
        pass

    def OnLocalHistEquKrnlCtrl(self, evt):
        val = evt.GetPosition()
        ctrl_val = self.SpinCtrlDblSpin(val, self.alg_local_hist_equ_krnl_curr_val, 1)
        self.alg_local_hist_equ_krnl_curr_val = ctrl_val
        self.alg_local_hist_equ_krnl_ctrl.SetValue(ctrl_val)
        
    def OnDIPAlgSelect(self, evt):
        sel_id = self.dip_alg_choice.GetSelection()
        assert wx.NOT_FOUND != sel_id
        sel_str = self.dip_alg_choice.GetString(sel_id)
        self.dip_alg_id = dip_alg_dict[sel_str]['id']
        self.alg_comment_area.SetLabel(dip_alg_dict[sel_str]['comment'])
        self.CtrlsDisplayUpdate()
            
    def OnImgProcessCheck(self, evt):
        self.img_process_flag = self.img_process_flag_check.GetValue()
        if False == self.img_process_flag:
            self.func_sizer.Hide(self.bg_whiten_alg_setting_sizer, True)
        else:
            self.func_sizer.Show(self.bg_whiten_alg_setting_sizer, True, True)
        self.func_sizer.Fit(self)
        self.RefreshSizeDisplay()
            
    def OnGenPdfCheck(self, evt):
        self.gen_pdf_flag = self.gen_pdf_flag_check.GetValue()

    def OnWorkButton(self, event):
        """
        wx.MessageBox(self.dip_alg_choice.GetString(self.dip_alg_id) + '\n' +
                      "process flag: " + str(self.img_process_flag) + '\n' +
                      "krn size: " + str(self.alg_simple_thres_krnl_ctrl.GetValue()) + '\n' +
                      'black thres: ' + str(self.alg_simple_thres_black_thres_ctrl.GetValue()) + '\n' +
                      'beta: ' + str(self.alg_simple_thres_beta_ctrl.GetValue()))
        """
        if len(self.selected_files) <=0 :
            wx.MessageBox('请先选择待处理的图片文件')
            return

        output_path = ""
        if (DIP_ALG_SIMPLE_THRES_GRAY == self.dip_alg_id or \
           DIP_ALG_GAUSSIAN_FILTER_GRAY == self.dip_alg_id or \
           DIP_ALG_GAUSSIAN_FILTER == self.dip_alg_id):
            if not(self.img_process_flag or self.gen_pdf_flag):
                wx.MessageBox('至少勾选 ' + '\'' + GrayImgEnhanceSharp.PRE_PROCESS_IMG_STR + '\' 或 \'' + GrayImgEnhanceSharp.GEN_PDF_FILE_STR + '\' 中的至少一项')
                return
            files_to_be_processed = self.selected_files
            if self.img_process_flag:
                output_path = get_alg_str(self.dip_alg_id)
            else:
                output_path = FUNCTION_TITLE

        elif DIP_ALG_HIST_EQU == self.dip_alg_id or \
             DIP_ALG_LOCAL_HIST_EQU == self.dip_alg_id :
            if None == self.img_display_ctrl_fpn:
                wx.MessageBox("请在列表中点击要处理的图片")
                return
            files_to_be_processed = [self.img_display_ctrl_fpn[self.img_display_ctrl_fpn.rfind('\\') + 1:]]
            output_path = get_alg_str(self.dip_alg_id)
            
        else:
            wx.MessageBox('当前无此处理模式')
            return

        output_path = output_path.replace(' ', '_') + '_result'
        output_path = mkdir_avoid_dup(self.selected_path, output_path)
        
        process_result = self.work_process(self.selected_path,
                                           files_to_be_processed,
                                           output_path,
                                           self.dip_alg_id,
                                           self.img_process_flag,
                                           self.gen_pdf_flag)

    def histogram_process(self, img_file_path, img_file_name, work_out_dir, process_flag = True, bg_color = 255):
        histogram_img_ext = ".jpg"
        histo_img_file_name_apx = "_histogram_img"
        work_out_full_path = img_file_path + "\\" + work_out_dir
        ori_img = read_ch_path_img(img_file_path + "\\" + img_file_name)
        histo_img = histogram_img(ori_img, bg_color)
        ori_histo_img_full_with_base_name = work_out_full_path + "\\" + img_file_name[:img_file_name.rfind(".")] + histo_img_file_name_apx
        write_ch_path_img(ori_histo_img_full_with_base_name + histogram_img_ext, histo_img)
        ori_img_chs = cv.split(ori_img)
        if False == is_img_single_channel(ori_img):
            for ch, ori_img_ch in enumerate(ori_img_chs):
                ch_apx = "_ch_" + str(ch)
                ch_histo = histogram_img(ori_img_ch, bg_color)
                write_ch_path_img(ori_histo_img_full_with_base_name + ch_apx + histogram_img_ext, ch_histo)

        if process_flag:
            histo_equed_file_name_apx = "_histogram_equalized"
            img_equed_chs = [cv.equalizeHist(img_ch) for img_ch in ori_img_chs]
            equalized_img = cv.merge(img_equed_chs)
            equed_img_full_with_base_name = work_out_full_path + "\\" + img_file_name[:img_file_name.rfind(".")] + \
                                            histo_equed_file_name_apx
            write_ch_path_img(equed_img_full_with_base_name + img_file_name[img_file_name.find("."):], equalized_img)
            histo_img = histogram_img(equalized_img, bg_color)
            equed_histo_img_full_with_base_name = equed_img_full_with_base_name + histo_img_file_name_apx
            write_ch_path_img(equed_histo_img_full_with_base_name + histogram_img_ext, \
                              histo_img)
            if False == is_img_single_channel(ori_img):
                for ch, img_equed_ch in enumerate(img_equed_chs):
                    ch_apx = "_ch_" + str(ch)
                    ch_histo = histogram_img(img_equed_ch, bg_color)
                    write_ch_path_img(equed_histo_img_full_with_base_name + ch_apx + histogram_img_ext, ch_histo)

    def local_histogram_equ(self, img_file_path, img_file_name, work_out_dir, win_size = LOCAL_HIST_EQUAL_KRN_SIZE_DEF):
        local_histo_equed_img_file_name_apx = "_local_hist_equ_img"
        work_out_full_path = img_file_path + "\\" + work_out_dir
        ori_img = read_ch_path_img(img_file_path + "\\" + img_file_name)
        local_hist_equed_img = local_histogram_equalization(ori_img, win_size)
        local_hist_equed_img_full_name = work_out_full_path + "\\" + img_file_name[:img_file_name.rfind(".")] +\
                                                   local_histo_equed_img_file_name_apx + img_file_name[img_file_name.find("."):]
        write_ch_path_img(local_hist_equed_img_full_name, local_hist_equed_img)
        
    
    def work_process(self, img_file_path, img_file_list, work_out_dir, dip_alg = DIP_ALG_GAUSSIAN_FILTER_GRAY, process_flag = True, gen_pdf_flag = True):
        wx.BeginBusyCursor()
        start_tick = cv.getTickCount()
        dip_alg_supported = True

        work_out_full_path = img_file_path + "\\" + work_out_dir

        result_str = ""
        if (DIP_ALG_SIMPLE_THRES_GRAY == self.dip_alg_id or \
           DIP_ALG_GAUSSIAN_FILTER_GRAY == self.dip_alg_id or \
           DIP_ALG_GAUSSIAN_FILTER == self.dip_alg_id):
            if process_flag:
                result_img_apx = "_result"
                result_img_file_list = []

                for img_file in img_file_list:
                    img_file_path_name = img_file_path + "\\" + img_file
                    img_file_base, img_file_ext = os.path.splitext(img_file)
                    #ori_img = cv.imread(img_file_path_name)
                    ori_img = read_ch_path_img(img_file_path_name)
                    if DIP_ALG_GAUSSIAN_FILTER == dip_alg:
                        result_img = dip_alg_gaussian_filter_en(ori_img)
                    elif DIP_ALG_GAUSSIAN_FILTER_GRAY == dip_alg:
                        #gray_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)
                        gray_img = color2Gray(ori_img)
                        result_img = dip_alg_gaussian_filter_en(gray_img)
                    else: #DIP_ALG_SIMPLE_THRES_GRAY == dip_alg
                        krnl_size = self.alg_simple_thres_krnl_ctrl.GetValue()
                        assert DIP_ALG_SIMPLE_THRES_KRNL_MIN <= krnl_size <= DIP_ALG_SIMPLE_THRES_KRNL_MAX
                        black_thres = self.alg_simple_thres_black_thres_ctrl.GetValue()
                        assert DIP_ALG_SIMPLE_THRES_BLACK_THRES_MIN <= black_thres <= DIP_ALG_SIMPLE_THRES_BLACK_THRES_MAX
                        beta = self.alg_simple_thres_beta_ctrl.GetValue()
                        assert DIP_ALG_SIMPLE_THRES_BETA_MIN <= beta <= DIP_ALG_SIMPLE_THRES_BETA_MAX
                        #gray_img = cv.cvtColor(ori_img, cv.COLOR_BGR2GRAY)
                        gray_img = color2Gray(ori_img)
                        result_img = adaptive_thres(gray_img, beta = beta, win = krnl_size, black_thres = black_thres)

                    result_path_name = work_out_full_path + "\\" \
                                            + add_apx_to_bn(img_file, result_img_apx)
                    #cv.imwrite(result_path_name, result_img)
                    write_ch_path_img(result_path_name, result_img, img_file_ext)
                    result_img_file_list.append(result_path_name)

                result_str += '--- 输出图片保存位置：' + work_out_full_path
            else:
                result_img_file_list = [img_file_path + "\\" + img_file for img_file in img_file_list]

            if gen_pdf_flag:
                pdf_name = work_out_dir + "_pdf.pdf" #用work_out_dir加上_pdf后缀作为输出的pdf文件名
                pdf_path_name = work_out_full_path + "\\" + pdf_name
                combine2Pdf(result_img_file_list, pdf_path_name)
                result_str += '\n--- 生成pdf文件：' + pdf_name
        elif DIP_ALG_HIST_EQU == self.dip_alg_id:
            self.histogram_process(img_file_path, img_file_list[0], work_out_dir, process_flag)
            result_str += '--- 输出图片保存位置：' + work_out_full_path
        elif DIP_ALG_LOCAL_HIST_EQU == self.dip_alg_id:
            self.local_histogram_equ(img_file_path, img_file_list[0], work_out_dir, self.alg_local_hist_equ_krnl_curr_val)
            result_str += '--- 输出图片保存位置：' + work_out_full_path
        else:
            dip_alg_supported = False
        
        end_tick = cv.getTickCount()
        time_lapse = (end_tick - start_tick)/cv.getTickFrequency()
        wx.EndBusyCursor()
        time_str = '\nOK\n--- 耗时: ' + str(round(time_lapse,1)) + ' 秒'
        result_str += time_str

        if dip_alg_supported:
            wx.MessageBox(result_str)
        else:
            wx.MessageBox('当前不支持此项功能')
        return dip_alg_supported

app = wx.App(False)
gray_img_enhanc_sharp = GrayImgEnhanceSharp()

app.MainLoop()
