#!/usr/bin/env python
import wx
import wx.lib.newevent as NE
import os

FRAME_SIZE_W = 500
FRAME_SIZE_H = 200

SIZER_BORDER = 2
GRID_DIZER_VGAP = 2
GRID_DIZER_HGAP = 5
LIST_BOX_W = 450 
LIST_BOX_H = 200
LIST_CTRL_FIL_COL_WIDTH = 250

ListCtrlContentUpdatedIndEvent, EVT_LISTCTRL_CNT_UPDATED_IND = NE.NewEvent()

LIST_CTRL_ITEM_ORD_SIMPLE_FORMAT = 0
def list_ctrl_swap_item(list_ctrl, idx1, idx2, ord_col = -1, ord_format = LIST_CTRL_ITEM_ORD_SIMPLE_FORMAT):
    """
    Swap two items in list_ctrl of type wx.ListCtrl, indexed by idx1 and idx2.
    ord_col indicates the column in list_ctrl that contains the No. string of item. ord_format is the string format. SIMPLE_FOAMRT means simple 1, 2, 3...
    other format is to be defined.

    Return True means sucessful swap, False means swap fails.
    """
    if idx1 == idx2: return True
    
    item_cnt = list_ctrl.GetItemCount()
    if idx1 < 0 or idx2 < 0 or idx1 >= item_cnt or idx2 >= item_cnt: return False
    
    col_cnt = list_ctrl.GetColumnCount()
    if ord_col >= col_cnt: return False

    if idx1 > idx2: idx1, idx2 = idx2, idx1
    row1_list, row2_list = [wx.ListItem(list_ctrl.GetItem(idx1, col_idx)) for col_idx in range(col_cnt)],\
                           [wx.ListItem(list_ctrl.GetItem(idx2, col_idx)) for col_idx in range(col_cnt)]
    #wx.MessageBox('row1_state: ' + str(row1_list[1].State) + '\nrow2_state: ' + str(row2_list[1].State))
    if ord_col >= 0:
        row1_list[ord_col].SetText(str(idx2))
        row2_list[ord_col].SetText(str(idx1))
    
    list_ctrl.DeleteItem(idx2)
    row1_list[0].SetId(idx2)
    list_ctrl.InsertItem(row1_list[0])
    list_ctrl.DeleteItem(idx1)
    row2_list[0].SetId(idx1)
    list_ctrl.InsertItem(row2_list[0])
    for col_idx in range(col_cnt)[1:]:
        row1_list[col_idx].SetId(idx2)
        row2_list[col_idx].SetId(idx1)
        list_ctrl.SetItem(row1_list[col_idx])
        list_ctrl.SetItem(row2_list[col_idx])

    """
    if ord_col >= 0:
        list_ctrl.GetItem(idx1, ord_col).SetText(str(idx2))
        list_ctrl.GetItem(idx2, ord_col).SetText(str(idx1))
    """
    return True

class FileSelector(wx.Frame):
    def __init__(self, parent, title, file_ext = "*.*"):
        self.file_ext = file_ext
        self.selected_path = ""
        self.selected_files = []
        
        wx.Frame.__init__(self, parent, title=title, 
                          style = wx.VSCROLL | (wx.DEFAULT_FRAME_STYLE  &  ~(wx.MAXIMIZE_BOX)))

        #self.SetMaxSize((FRAME_SIZE_W, FRAME_SIZE_H))

        self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.file_sel_sizer = wx.GridBagSizer(GRID_DIZER_VGAP, GRID_DIZER_HGAP)
        self.file_selection_btn = wx.Button(self, -1, "选择文件")
        self.file_sel_sizer.Add(self.file_selection_btn, (0, 0), flag = wx.ALIGN_CENTER | wx.ALL, border = SIZER_BORDER)
        self.file_sel_sizer.Add(wx.StaticText(self, label = "文件列表"), (1, 0), flag = wx.LEFT | wx.ALIGN_LEFT, border = SIZER_BORDER)
        self.file_list_ctrl = wx.ListCtrl(self, style = wx.LC_REPORT)
        self.file_list_ctrl.InsertColumn(0, "No.")
        self.file_list_ctrl.InsertColumn(1, "文件名", width = LIST_CTRL_FIL_COL_WIDTH)
        self.file_sel_sizer.Add(self.file_list_ctrl, (2, 0), flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.file_sel_sizer.SetItemSpan(self.file_list_ctrl, (2, 1))
        self.move_btn_up = wx.Button(self, -1, "上移")
        self.move_btn_down = wx.Button(self, -1, "下移")
        self.file_sel_sizer.Add(self.move_btn_up, (2, 1), flag = wx.ALIGN_TOP | wx.ALL, border = SIZER_BORDER)
        self.file_sel_sizer.Add(self.move_btn_down, (3, 1), flag = wx.ALIGN_TOP | wx.ALL, border = SIZER_BORDER)
        #self.file_sel_path_text = wx.StaticText(self)
        self.file_sel_path_text = wx.TextCtrl(self, style = wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_NO_VSCROLL | wx.BORDER_NONE,
                                              size = (self.file_list_ctrl.GetSize().width,-1))
        self.file_sel_path_text.SetBackgroundColour(self.GetBackgroundColour())
        self.file_sel_path_text_sizer = wx.BoxSizer(wx.VERTICAL)
        self.file_sel_path_text_sizer.Add(wx.StaticText(self, label =  '文件路径：'), flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.file_sel_path_text_sizer.Add(self.file_sel_path_text, flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        self.file_sel_sizer.Add(self.file_sel_path_text_sizer, (4, 0), flag = wx.ALIGN_LEFT | wx.ALL, border = SIZER_BORDER)
        
        self.func_area_sizer = None #wx.GridBagSizer(GRID_DIZER_VGAP, GRID_DIZER_HGAP)

        self.top_sizer.Add(self.file_sel_sizer, border = SIZER_BORDER)

        #self.top_sizer.AddMany((self.file_sel_sizer, self.func_area_sizer))
        
        self.Bind(wx.EVT_BUTTON, self.OnFileSelBtn, self.file_selection_btn)
        self.Bind(wx.EVT_BUTTON, self.OnUpBtn, self.move_btn_up)
        self.Bind(wx.EVT_BUTTON, self.OnDownBtn, self.move_btn_down)

        self.Bind(EVT_LISTCTRL_CNT_UPDATED_IND, self.OnListCtrlCntUpdated)

        self.SetSizerAndFit(self.top_sizer)
        self.SetAutoLayout(1)
        
        self.Show(True)

    def RefreshSizeDisplay(self):
        self.top_sizer.Fit(self)
        
    def AddFuncArea(self, func_sizer):
        if None == func_sizer: return
        
        self.func_area_sizer = func_sizer
        self.top_sizer.AddSpacer(SIZER_BORDER * 10)
        self.top_sizer.Add(self.func_area_sizer, flag = wx.ALL, border = SIZER_BORDER)
        
        self.top_sizer.Fit(self)
        
    def OnExit(self,e):
        self.Close(True)

    def OnUpBtn(self, event):
        sel_idx = self.file_list_ctrl.GetFirstSelected()
        if sel_idx <= 0 : return

        if self.file_list_ctrl.GetSelectedItemCount() > 1 :
            wx.MessageBox("Only 1 item can be selected at one time!")
            return

        #self.file_list_ctrl.SetItemState(sel_idx, 0, wx.LIST_STATE_SELECTED) #SetItemState和Select效果相同
        self.file_list_ctrl.Select(sel_idx, 0)
        ret = list_ctrl_swap_item(self.file_list_ctrl, sel_idx - 1, sel_idx, 0)
        if not ret:
            wx.MessageBox("Swap Error")
            return
        
        self.selected_files[sel_idx - 1], self.selected_files[sel_idx] = self.selected_files[sel_idx], self.selected_files[sel_idx - 1]
        #self.file_list_ctrl.SetItemState(sel_idx - 1, wx.LIST_STATE_SELECTED, wx.LIST_STATE_SELECTED) #SetItemState和Select效果相同，都会发送EVT_LIST_ITEM_SELECTED事件
        self.file_list_ctrl.Select(sel_idx - 1, 1)
        self.file_list_ctrl.EnsureVisible(sel_idx - 1)

    def OnDownBtn(self, event):
        sel_idx = self.file_list_ctrl.GetFirstSelected()
        row_cnt = self.file_list_ctrl.GetItemCount()
        if sel_idx < 0 or sel_idx >= row_cnt-1: return

        if self.file_list_ctrl.GetSelectedItemCount() > 1 :
            wx.MessageBox("Only 1 item can be selected at one time!")
            return

        self.file_list_ctrl.Select(sel_idx, 0)
        ret = list_ctrl_swap_item(self.file_list_ctrl, sel_idx, sel_idx + 1, 0)
        if not ret:
            wx.MessageBox("Swap Error")
            return
        self.selected_files[sel_idx], self.selected_files[sel_idx + 1] = self.selected_files[sel_idx + 1], self.selected_files[sel_idx]
        self.file_list_ctrl.Select(sel_idx + 1, 1)
        self.file_list_ctrl.EnsureVisible(sel_idx + 1)
    
    def OnFileSelBtn(self, event):
        with wx.FileDialog(self, "Please Select Files",
                           wildcard = self.file_ext,
                           style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE) as f_sel_diag:
            if f_sel_diag.ShowModal() == wx.ID_CANCEL:
                return

            self.selected_path = os.path.dirname(f_sel_diag.GetPaths()[0])
            self.selected_files = f_sel_diag.GetFilenames()
            self.selected_files.sort()

            self.file_sel_path_text.SetValue(self.selected_path)
            self.file_sel_path_text.Fit()

            self.file_list_ctrl.DeleteAllItems()
            for (idx, s) in zip(range(len(self.selected_files)), self.selected_files):
                self.file_list_ctrl.Append((str(idx),s))
            self.file_list_ctrl_item_bkg_color = self.file_list_ctrl.GetItemBackgroundColour(0)
            #self.SendSizeEvent()
            #self.Fit()

            wx.QueueEvent(self, ListCtrlContentUpdatedIndEvent(src="SelectNewFiles"))

    def OnWorkButton(self, event):
        s = [self.file_list_ctrl.GetItemState(i, wx.LIST_STATE_SELECTED) for i in range(5)]
        wx.MessageBox("states:" + str(s))

    def OnDeselected(self,event):
        wx.MessageBox('deselected id' + str(event.Item.Id) + ',color ' + str(event.Item.BackgroundColour))

    def OnListCtrlCntUpdated(self, event):
        self.SendSizeEvent()
        self.Fit()
    
if __name__ == "__main__":
    def Func_PrintPreProcess(file_selector):
        wx.MessageBox("Print Pre-process ...")
        
    app = wx.App(False)
    file_selector = FileSelector(None, 'FileSelector')

    work_btn = wx.Button(file_selector, -1, "Generate PDF")
    file_selector.Bind(wx.EVT_BUTTON, file_selector.OnWorkButton, work_btn)
    img_process_flag = wx.CheckBox(file_selector, wx.ID_ANY, "Process Image")
    func_sizer = wx.BoxSizer(wx.HORIZONTAL)
    func_sizer.Add(work_btn, flag = wx.ALIGN_BOTTOM | wx.ALL, border = SIZER_BORDER)
    func_sizer.Add(img_process_flag, flag = wx.ALIGN_BOTTOM | wx.ALL, border = SIZER_BORDER)
    file_selector.AddFuncArea(func_sizer)

    file_selector.Bind(wx.EVT_LIST_ITEM_DESELECTED, file_selector.OnDeselected)
    
    #file_selector.AddFuncBtns("Print Pre-process", Func_PrintPreProcess)

    app.MainLoop()
