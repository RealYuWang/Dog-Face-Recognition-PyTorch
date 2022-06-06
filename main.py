from gui.core.functions import Functions
from gui.uis.windows.main_window.functions_main_window import *
import sys
import os

from qt_core import *
from gui.core.json_settings import Settings
from gui.uis.windows.main_window import *

# 使用高DPI和4K显示器
os.environ["QT_FONT_DPI"] = "96"
# 如果使用4K显示器，使用下面的这行代码
os.environ["QT_SCALE_FACTOR"] = "2"


# 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 从 gui\uis\main_window\ui_main.py 加载小组件
        self.ui = UI_MainWindow()
        self.ui.setup_ui(self)
        # 加载设定
        settings = Settings()
        self.settings = settings.items
        # 设置主窗口
        self.hide_grips = True # Show/Hide resize grips
        SetupMainWindow.setup_gui(self)
        # 显示主窗口
        self.show()

    # 当按钮被点击后，运行相应程序
    # 通过按钮的 objectName() 来确定哪个按钮被按下
    def btn_clicked(self):
        # 获得被按下的按钮
        btn = SetupMainWindow.setup_btns(self)

        # 左菜单的按钮
        # 分为两种：上方和下方 两种按钮被选中后使用不同的选中函数
        # 上面的使用 select_only_one() 下面的使用 select_only_one_tab

        # 主页按钮
        if btn.objectName() == 'btn_home':
            # 设置按钮为被选中状态，图片改变
            self.ui.left_menu.select_only_one(btn.objectName())
            # 载入对应的页
            MainFunctions.set_page(self, self.ui.load_pages.page_1)

        # 检测页按钮
        if btn.objectName() == 'btn_detect':
            # 设置按钮为被选中状态，图片改变
            self.ui.left_menu.select_only_one(btn.objectName())
            # 载入对应的页
            MainFunctions.set_page(self, self.ui.load_pages.page_2)

        # 识别页按钮
        if btn.objectName() == 'btn_recognition':
            # 设置按钮为被选中状态，图片改变
            self.ui.left_menu.select_only_one(btn.objectName())
            # 载入对应的页
            MainFunctions.set_page(self, self.ui.load_pages.page_3)

        # 数据库页按钮
        if btn.objectName() == 'btn_database':
            # 设置按钮为被选中状态，图片改变
            self.ui.left_menu.select_only_one(btn.objectName())
            # 载入对应的页
            MainFunctions.set_page(self, self.ui.load_pages.page_4)


        # 获取顶部设置按钮点击状态，当其已经被选中时，按下左下角按钮后，取消其选中
        top_btn_settings = MainFunctions.get_title_bar_btn(self, 'btn_top_settings')

        # 左下角设置按钮 同时响应关闭子菜单按钮
        if btn.objectName() == 'btn_settings' or btn.objectName() == 'btn_close_left_column':
            # 取消顶部设置按钮选中
            top_btn_settings.set_active(False)
            # 若子菜单未弹出
            if not MainFunctions.left_column_is_visible(self):
                # 弹出子菜单
                MainFunctions.toggle_left_column(self)
                # 改变按钮为被选中状态
                self.ui.left_menu.select_only_one_tab(btn.objectName())
            # 若子菜单已经弹出
            else:
                # 若关闭子菜单按钮被按下
                if btn.objectName() == 'btn_close_left_column':
                    # 取消所有左菜单按钮的选中状态
                    self.ui.left_menu.deselect_all_tab()
                    # 隐藏子菜单
                    MainFunctions.toggle_left_column(self)
                # 改变按钮为被选中状态
                self.ui.left_menu.select_only_one_tab(btn.objectName())

            # 改变子菜单页面
            if btn.objectName() != 'btn_close_left_column':
                MainFunctions.set_left_column_menu(
                    self,
                    menu=self.ui.left_column.menus.menu_1,
                    title='Settings tab',
                    icon_path=Functions.set_svg_icon('icon_settings.png')
                )

        # 左下角设置按钮 同时响应关闭子菜单按钮
        if btn.objectName() == 'btn_info' or btn.objectName() == 'btn_close_left_column':
            # 取消顶部设置按钮选中
            top_btn_settings.set_active(False)
            # 若子菜单未弹出
            if not MainFunctions.left_column_is_visible(self):
                # 弹出子菜单
                MainFunctions.toggle_left_column(self)
                # 改变按钮为被选中状态
                self.ui.left_menu.select_only_one_tab(btn.objectName())
            # 若子菜单已经弹出
            else:
                # 若关闭子菜单按钮被按下
                if btn.objectName() == 'btn_close_left_column':
                    # 取消所有左菜单按钮的选中状态
                    self.ui.left_menu.deselect_all_tab()
                    # 隐藏子菜单
                    MainFunctions.toggle_left_column(self)
                # 改变按钮为被选中状态
                self.ui.left_menu.select_only_one_tab(btn.objectName())

            # 改变子菜单页面
            if btn.objectName() != 'btn_close_left_column':
                MainFunctions.set_left_column_menu(
                    self,
                    menu=self.ui.left_column.menus.menu_2,
                    title='Info tab',
                    icon_path=Functions.set_svg_icon('icon_info.png')
                )

        # 状态栏按钮
        # ///////////////////////////////////////////////////////////////
        # 顶部的设置按钮
        if btn.objectName() == "btn_top_settings":
            # 若右菜单未弹出
            if not MainFunctions.right_column_is_visible(self):
                btn.set_active(True)
                # Show / Hide
                MainFunctions.toggle_right_column(self)
            else:
                btn.set_active(False)
                # Show / Hide
                MainFunctions.toggle_right_column(self)
            # Get Left Menu Btn            
            btn_settings = MainFunctions.get_left_menu_btn(self, "btn_settings")
            btn_settings.set_active_tab(False)

            # Get Left Menu Btn
            btn_info = MainFunctions.get_left_menu_btn(self, "btn_info")
            btn_info.set_active_tab(False)




    # LEFT MENU BTN IS RELEASED
    # Run function when btn is released
    # Check funtion by object name / btn_id
    # ///////////////////////////////////////////////////////////////
    def btn_released(self):
        # GET BT CLICKED
        btn = SetupMainWindow.setup_btns(self)



    # RESIZE EVENT
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        SetupMainWindow.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        p = event.globalPosition()
        # SET DRAG POS WINDOW
        # self.dragPos = event.globalPos()
        self.dragPos = p.toPoint()


# SETTINGS WHEN TO START
# Set the initial class and also additional parameters of the "QApplication" class
# ///////////////////////////////////////////////////////////////
if __name__ == "__main__":
    # APPLICATION
    # ///////////////////////////////////////////////////////////////
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icn.ico"))
    window = MainWindow()

    # EXEC APP
    # ///////////////////////////////////////////////////////////////
    sys.exit(app.exec())