#!/usr/bin/env python3
"""
BlurImageProcessor GUI 界面
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.pipeline import BlurProcessor
from utils.logger import Logger

class BlurImageProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BlurImageProcessor")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # 创建日志记录器
        self.logger = Logger()
        
        # 创建处理器
        self.processor = BlurProcessor()
        
        # 图像变量
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        
        # 质量指标
        self.quality_metrics = None
        
        # 设置主题
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建顶部控制栏
        self.create_control_bar()
        
        # 创建图像显示区域
        self.create_image_display()
        
        # 创建功能面板
        self.create_function_panels()
        
        # 创建结果显示区域
        self.create_results_panel()
    
    def create_control_bar(self):
        """创建顶部控制栏"""
        control_frame = ttk.Frame(self.main_frame, padding="5")
        control_frame.pack(fill=tk.X, pady=5)
        
        # 加载图像按钮
        load_btn = ttk.Button(control_frame, text="加载图像", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # 保存结果按钮
        save_btn = ttk.Button(control_frame, text="保存结果", command=self.save_result)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        # 重置按钮
        reset_btn = ttk.Button(control_frame, text="重置", command=self.reset)
        reset_btn.pack(side=tk.LEFT, padx=5)
        
        # 分割线
        separator = ttk.Separator(control_frame, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.pack(side=tk.RIGHT, padx=5)
    
    def create_image_display(self):
        """创建图像显示区域"""
        image_frame = ttk.Frame(self.main_frame, padding="5")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建图像对比框架
        compare_frame = ttk.Frame(image_frame)
        compare_frame.pack(fill=tk.BOTH, expand=True)
        
        # 原始图像区域
        original_frame = ttk.LabelFrame(compare_frame, text="原始图像", padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg="#e0e0e0")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 处理后图像区域
        processed_frame = ttk.LabelFrame(compare_frame, text="处理后图像", padding="5")
        processed_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.processed_canvas = tk.Canvas(processed_frame, bg="#e0e0e0")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
    
    def create_function_panels(self):
        """创建功能面板"""
        function_frame = ttk.Frame(self.main_frame, padding="5")
        function_frame.pack(fill=tk.X, pady=5)
        
        # 创建三个功能标签页
        notebook = ttk.Notebook(function_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 模糊检测标签页
        self.detection_frame = ttk.Frame(notebook, padding="10")
        notebook.add(self.detection_frame, text="模糊检测")
        self.create_detection_panel()
        
        # 去模糊标签页
        self.deblur_frame = ttk.Frame(notebook, padding="10")
        notebook.add(self.deblur_frame, text="去模糊")
        self.create_deblur_panel()
        
        # 图像增强标签页
        self.enhance_frame = ttk.Frame(notebook, padding="10")
        notebook.add(self.enhance_frame, text="图像增强")
        self.create_enhance_panel()
    
    def create_detection_panel(self):
        """创建模糊检测面板"""
        # 检测方法选择
        method_frame = ttk.Frame(self.detection_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="检测方法:").pack(side=tk.LEFT, padx=5)
        self.detection_method = ttk.Combobox(method_frame, values=["laplacian", "fft", "gradient", "all"], state="readonly")
        self.detection_method.current(0)
        self.detection_method.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # 检测按钮
        detect_btn = ttk.Button(self.detection_frame, text="检测模糊", command=self.detect_blur)
        detect_btn.pack(fill=tk.X, pady=5)
        
        # 检测结果显示
        self.detection_result = ttk.Label(self.detection_frame, text="检测结果: 未检测", font=("Arial", 10, "bold"))
        self.detection_result.pack(fill=tk.X, pady=5)
    
    def create_deblur_panel(self):
        """创建去模糊面板"""
        # 去模糊方法选择
        method_frame = ttk.Frame(self.deblur_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="去模糊方法:").pack(side=tk.LEFT, padx=5)
        self.deblur_method = ttk.Combobox(self.deblur_frame, values=["wiener", "richardson_lucy", "blind", "unsharp"], state="readonly")
        self.deblur_method.current(0)
        self.deblur_method.pack(fill=tk.X, pady=5)
        
        # 参数调整
        params_frame = ttk.Frame(self.deblur_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # 维纳滤波参数
        self.wiener_frame = ttk.Frame(params_frame)
        self.wiener_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.wiener_frame, text="平衡参数:").pack(side=tk.LEFT, padx=5)
        self.wiener_balance = ttk.Scale(self.wiener_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL, length=200)
        self.wiener_balance.set(0.1)
        self.wiener_balance.pack(side=tk.LEFT, padx=5)
        self.wiener_value = ttk.Label(self.wiener_frame, text="0.10")
        self.wiener_value.pack(side=tk.LEFT, padx=5)
        self.wiener_balance.bind("<Motion>", self.update_wiener_value)
        
        # Richardson-Lucy参数
        self.richardson_frame = ttk.Frame(params_frame)
        self.richardson_frame.pack(fill=tk.X, pady=5)
        self.richardson_frame.pack_forget()  # 默认隐藏
        
        ttk.Label(self.richardson_frame, text="迭代次数:").pack(side=tk.LEFT, padx=5)
        self.richardson_iter = ttk.Scale(self.richardson_frame, from_=10, to=100, orient=tk.HORIZONTAL, length=200)
        self.richardson_iter.set(30)
        self.richardson_iter.pack(side=tk.LEFT, padx=5)
        self.richardson_value = ttk.Label(self.richardson_frame, text="30")
        self.richardson_value.pack(side=tk.LEFT, padx=5)
        self.richardson_iter.bind("<Motion>", self.update_richardson_value)
        
        # 去模糊按钮
        deblur_btn = ttk.Button(self.deblur_frame, text="执行去模糊", command=self.deblur)
        deblur_btn.pack(fill=tk.X, pady=5)
        
        # 绑定方法选择事件
        self.deblur_method.bind("<<ComboboxSelected>>", self.on_deblur_method_change)
    
    def create_enhance_panel(self):
        """创建图像增强面板"""
        # 增强方法选择
        method_frame = ttk.Frame(self.enhance_frame)
        method_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(method_frame, text="增强方法:").pack(side=tk.LEFT, padx=5)
        self.enhance_method = ttk.Combobox(self.enhance_frame, values=["sharpening", "contrast", "denoise"], state="readonly")
        self.enhance_method.current(0)
        self.enhance_method.pack(fill=tk.X, pady=5)
        
        # 参数调整
        params_frame = ttk.Frame(self.enhance_frame)
        params_frame.pack(fill=tk.X, pady=5)
        
        # 锐化参数
        self.sharpen_frame = ttk.Frame(params_frame)
        self.sharpen_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.sharpen_frame, text="锐化强度:").pack(side=tk.LEFT, padx=5)
        self.sharpen_amount = ttk.Scale(self.sharpen_frame, from_=0.1, to=3.0, orient=tk.HORIZONTAL, length=200)
        self.sharpen_amount.set(1.5)
        self.sharpen_amount.pack(side=tk.LEFT, padx=5)
        self.sharpen_value = ttk.Label(self.sharpen_frame, text="1.50")
        self.sharpen_value.pack(side=tk.LEFT, padx=5)
        self.sharpen_amount.bind("<Motion>", self.update_sharpen_value)
        
        # 对比度参数
        self.contrast_frame = ttk.Frame(params_frame)
        self.contrast_frame.pack(fill=tk.X, pady=5)
        self.contrast_frame.pack_forget()  # 默认隐藏
        
        ttk.Label(self.contrast_frame, text="对比度:").pack(side=tk.LEFT, padx=5)
        self.contrast_alpha = ttk.Scale(self.contrast_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, length=200)
        self.contrast_alpha.set(1.2)
        self.contrast_alpha.pack(side=tk.LEFT, padx=5)
        self.contrast_value = ttk.Label(self.contrast_frame, text="1.20")
        self.contrast_value.pack(side=tk.LEFT, padx=5)
        self.contrast_alpha.bind("<Motion>", self.update_contrast_value)
        
        # 增强按钮
        enhance_btn = ttk.Button(self.enhance_frame, text="执行增强", command=self.enhance)
        enhance_btn.pack(fill=tk.X, pady=5)
        
        # 绑定方法选择事件
        self.enhance_method.bind("<<ComboboxSelected>>", self.on_enhance_method_change)
    
    def create_results_panel(self):
        """创建结果显示面板"""
        results_frame = ttk.LabelFrame(self.main_frame, text="质量评估", padding="10")
        results_frame.pack(fill=tk.X, pady=5)
        
        # 质量指标显示
        self.metrics_frame = ttk.Frame(results_frame)
        self.metrics_frame.pack(fill=tk.X, pady=5)
        
        self.metrics_labels = {}
        metrics = ["PSNR", "SSIM", "MSE", "MAE"]
        
        for i, metric in enumerate(metrics):
            frame = ttk.Frame(self.metrics_frame)
            frame.pack(side=tk.LEFT, padx=20, pady=5)
            
            ttk.Label(frame, text=metric, font=("Arial", 10, "bold")).pack()
            label = ttk.Label(frame, text="--", font=("Arial", 12))
            label.pack()
            self.metrics_labels[metric] = label
        
        # 评估按钮
        eval_btn = ttk.Button(results_frame, text="评估质量", command=self.evaluate_quality)
        eval_btn.pack(fill=tk.X, pady=5)
    
    def load_image(self):
        """加载图像"""
        file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            
            # 加载图像到处理器
            self.processor.load_image(file_path)
            
            # 显示图像
            self.display_image(self.original_image, self.original_canvas)
            self.display_image(self.processed_image, self.processed_canvas)
            
            # 更新状态
            self.status_label.config(text=f"已加载图像: {Path(file_path).name}")
            self.reset_metrics()
    
    def display_image(self, image, canvas):
        """在画布上显示图像"""
        # 调整图像大小以适应画布
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width == 1 and canvas_height == 1:  # 初始状态
            canvas_width = 400
            canvas_height = 300
        
        # 计算缩放比例
        h, w = image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整图像大小
        resized = cv2.resize(image, (new_w, new_h))
        
        # 转换为RGB格式
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 显示图像
        canvas.delete("all")
        canvas.create_image((canvas_width - new_w) // 2, (canvas_height - new_h) // 2, anchor=tk.NW, image=photo)
        
        # 保存引用，防止图像被垃圾回收
        canvas.image = photo
    
    def detect_blur(self):
        """检测模糊"""
        if not self.original_image:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        method = self.detection_method.get()
        is_blurry, score = self.processor.detect_blur(method=method)
        
        result_text = f"检测结果: {'模糊' if is_blurry else '清晰'}，分数: {score:.2f}"
        self.detection_result.config(text=result_text)
        self.status_label.config(text=f"模糊检测完成: {result_text}")
    
    def deblur(self):
        """执行去模糊"""
        if not self.original_image:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        method = self.deblur_method.get()
        
        if method == "wiener":
            balance = self.wiener_balance.get()
            self.processed_image = self.processor.deblur(method=method, balance=balance)
        elif method == "richardson_lucy":
            num_iter = int(self.richardson_iter.get())
            self.processed_image = self.processor.deblur(method=method, num_iter=num_iter)
        else:
            self.processed_image = self.processor.deblur(method=method)
        
        # 显示处理后图像
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_label.config(text=f"去模糊完成: {method}")
    
    def enhance(self):
        """执行图像增强"""
        if not self.original_image:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        method = self.enhance_method.get()
        
        if method == "sharpening":
            amount = self.sharpen_amount.get()
            self.processed_image = self.processor.enhance(method=method, amount=amount)
        elif method == "contrast":
            alpha = self.contrast_alpha.get()
            self.processed_image = self.processor.enhance(method=method, alpha=alpha)
        else:
            self.processed_image = self.processor.enhance(method=method)
        
        # 显示处理后图像
        self.display_image(self.processed_image, self.processed_canvas)
        self.status_label.config(text=f"图像增强完成: {method}")
    
    def evaluate_quality(self):
        """评估图像质量"""
        if not self.original_image:
            messagebox.showwarning("警告", "请先加载图像")
            return
        
        self.quality_metrics = self.processor.evaluate_quality()
        
        # 更新指标显示
        for metric, value in self.quality_metrics.items():
            if metric in self.metrics_labels:
                if metric == "SSIM":
                    self.metrics_labels[metric].config(text=f"{value:.4f}")
                else:
                    self.metrics_labels[metric].config(text=f"{value:.2f}")
        
        self.status_label.config(text="质量评估完成")
    
    def save_result(self):
        """保存结果图像"""
        if not self.processed_image:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG图像", "*.jpg"), ("PNG图像", "*.png"), ("BMP图像", "*.bmp")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            self.status_label.config(text=f"结果已保存到: {Path(file_path).name}")
            messagebox.showinfo("成功", "图像已保存")
    
    def reset(self):
        """重置所有设置"""
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.quality_metrics = None
        
        # 清空画布
        self.original_canvas.delete("all")
        self.processed_canvas.delete("all")
        
        # 重置状态
        self.status_label.config(text="就绪")
        self.detection_result.config(text="检测结果: 未检测")
        
        # 重置质量指标
        self.reset_metrics()
    
    def reset_metrics(self):
        """重置质量指标显示"""
        for label in self.metrics_labels.values():
            label.config(text="--")
    
    def update_wiener_value(self, event):
        """更新维纳滤波参数值"""
        value = self.wiener_balance.get()
        self.wiener_value.config(text=f"{value:.2f}")
    
    def update_richardson_value(self, event):
        """更新Richardson-Lucy参数值"""
        value = int(self.richardson_iter.get())
        self.richardson_value.config(text=str(value))
    
    def update_sharpen_value(self, event):
        """更新锐化参数值"""
        value = self.sharpen_amount.get()
        self.sharpen_value.config(text=f"{value:.2f}")
    
    def update_contrast_value(self, event):
        """更新对比度参数值"""
        value = self.contrast_alpha.get()
        self.contrast_value.config(text=f"{value:.2f}")
    
    def on_deblur_method_change(self, event):
        """处理去模糊方法变化"""
        method = self.deblur_method.get()
        
        # 隐藏所有参数面板
        self.wiener_frame.pack_forget()
        self.richardson_frame.pack_forget()
        
        # 显示对应方法的参数面板
        if method == "wiener":
            self.wiener_frame.pack(fill=tk.X, pady=5)
        elif method == "richardson_lucy":
            self.richardson_frame.pack(fill=tk.X, pady=5)
    
    def on_enhance_method_change(self, event):
        """处理增强方法变化"""
        method = self.enhance_method.get()
        
        # 隐藏所有参数面板
        self.sharpen_frame.pack_forget()
        self.contrast_frame.pack_forget()
        
        # 显示对应方法的参数面板
        if method == "sharpening":
            self.sharpen_frame.pack(fill=tk.X, pady=5)
        elif method == "contrast":
            self.contrast_frame.pack(fill=tk.X, pady=5)
    
    def on_resize(self, event):
        """处理窗口大小变化"""
        if self.original_image:
            self.display_image(self.original_image, self.original_canvas)
            self.display_image(self.processed_image, self.processed_canvas)

def main():
    """主函数"""
    root = tk.Tk()
    app = BlurImageProcessorGUI(root)
    
    # 绑定窗口大小变化事件
    root.bind("<Configure>", app.on_resize)
    
    # 启动主循环
    root.mainloop()

if __name__ == "__main__":
    main()
