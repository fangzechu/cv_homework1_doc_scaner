<img width="276" height="372" alt="image" src="https://github.com/user-attachments/assets/fd248bc5-41f8-482a-9143-91a83e35498d" /># cv_homework1_doc_scaner
### cv课作业_文档矫正
#1   任务场景概述
拍摄一张 A4 文稿照片，自动完成以下任务流：
角点检测→文档轮廓拟合→ 投影变换（单应性）→ 扫描风格增强
#2   依赖与环境
详见requirements.txt
主要包：`opencv-python`、`numpy`
#3   代码入口
color_p, bw_p = scan_document(sample_path, out_dir, debug=True)
修改document_scanner_251008.py路径为目标图像路径
sample_path = "./DS_scan_test.png"
#4  运行示例
python document_scanner_251008.py
#5   结果说明
输入：原始拍照（透视、旋转、阴影）
![image]
中间过程：Edges（Canny）、Corners（四角可视化）
输出：Scanned-Color（矫正彩色）、Scanned-BW（自适应阈值）

