import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 每个分支一套配色：(面底色, 边框色, 文字色)
PALETTE = {
    "mp (mediapipe)":        ("#1f3b73", "#1f3b73", "white"),
    "Image":                 ("#e3f2fd", "#1976d2", "#0d47a1"),
    "ImageFormat":           ("#fff3e0", "#ef6c00", "#e65100"),
    "tasks":                 ("#ede7f6", "#5e35b1", "#311b92"),
    "BaseOptions":           ("#fce4ec", "#c2185b", "#880e4f"),
    "vision":                ("#e8f5e9", "#2e7d32", "#1b5e20"),
    "RunningMode":           ("#f3e5f5", "#8e24aa", "#4a148c"),
    "FaceLandmarkerOptions": ("#e0f7fa", "#00838f", "#006064"),
    "FaceLandmarker":        ("#fff8e1", "#f9a825", "#f57f17"),
    "HandLandmarkerOptions": ("#fbe9e7", "#d84315", "#bf360c"),
    "HandLandmarker":        ("#e8eaf6", "#3949ab", "#1a237e"),
}
DEFAULT_COLOR = ("#f5f5f5", "#9e9e9e", "#424242")


# ============ 树结构：每个节点有 name（名称）和 desc（作用） ============
tree = {
    "name": "mp (mediapipe)",
    "desc": "MediaPipe 库总入口，所有功能从这里调用",
    "children": [
        {
            "name": "Image",
            "desc": "画面包装类：把 numpy 图像转成 MediaPipe 能吃的格式",
            "children": [
                {
                    "name": "mp.Image(...)",
                    "desc": "创建图像对象\nimage_format=SRGB, data=rgb\n调用点：blink:111 landmarks:115",
                },
            ],
        },
        {
            "name": "ImageFormat",
            "desc": "图像格式常量：告诉 MediaPipe 像素怎么排布",
            "children": [
                {"name": "ImageFormat.SRGB", "desc": "★ 项目用：标准 RGB 三通道"},
                {"name": "ImageFormat.SRGBA", "desc": "○ 未用：RGB + Alpha 四通道"},
            ],
        },
        {
            "name": "tasks",
            "desc": "“任务”子包：新版 MediaPipe API 都放这里",
            "children": [
                {
                    "name": "BaseOptions",
                    "desc": "基础配置：模型文件在哪、用什么设备算",
                    "children": [
                        {"name": "model_asset_path", "desc": "★ 模型文件路径\nstr(FACE_MODEL)"},
                        {"name": "model_asset_buffer", "desc": "○ 未用：模型在内存时用"},
                        {"name": "delegate (CPU/GPU)", "desc": "○ 未用：默认 CPU"},
                    ],
                },
                {
                    "name": "vision",
                    "desc": "“视觉”子包：人脸、手势、姿态等任务",
                    "children": [
                        {
                            "name": "RunningMode",
                            "desc": "工作模式：决定怎么喂数据、要不要时间戳",
                            "children": [
                                {"name": "RunningMode.IMAGE", "desc": "○ 未用：单张图，不用时间戳"},
                                {"name": "RunningMode.VIDEO", "desc": "★ 项目用：逐帧 + 时间戳 :56"},
                                {"name": "RunningMode.LIVE_STREAM", "desc": "○ 未用：异步 + 回调"},
                            ],
                        },
                        {
                            "name": "FaceLandmarkerOptions",
                            "desc": "人脸设置表：创建人脸检测器时的参数",
                            "children": [
                                {"name": "base_options", "desc": "★ 已填 :55\n模型路径等基础配置"},
                                {"name": "running_mode", "desc": "★ 已填 :56\n设为 VIDEO"},
                                {"name": "num_faces=1", "desc": "★ 已填 :57\n最多检测一张脸"},
                                {"name": "min_face_detection_confidence", "desc": "★ 已填 :58\n人脸检测置信度阈值"},
                                {"name": "min_tracking_confidence", "desc": "★ 已填 :59\n追踪置信度阈值"},
                                {"name": "min_face_presence_confidence", "desc": "○ 未填：默认 0.5"},
                                {"name": "output_face_blendshapes", "desc": "○ 未填：默认 False\n结果里 blendshapes 永远空"},
                                {"name": "output_facial_transformation_matrixes", "desc": "○ 未填：默认 False\n不输出 4x4 变换矩阵"},
                                {"name": "result_callback", "desc": "○ 未填：只有 LIVE_STREAM 才需要"},
                            ],
                        },
                        {
                            "name": "FaceLandmarker",
                            "desc": "人脸识别器“模具”：加载模型、检测 478 个关键点",
                            "children": [
                                {"name": "create_from_options(options)", "desc": "★ 创建检测器实例\n4 个脚本各造一台"},
                                {"name": ".detect_for_video(...)", "desc": "★ 每帧一次，同步阻塞\n输入图像 + 时间戳(ms)"},
                                {"name": ".close()", "desc": "★ 释放模型资源\n用完必须调用"},
                            ],
                        },
                        {
                            "name": "HandLandmarkerOptions",
                            "desc": "手部设置表：创建手部检测器时的参数",
                            "children": [
                                {"name": "base_options", "desc": "★ :84\n模型路径等基础配置"},
                                {"name": "running_mode=VIDEO", "desc": "★ :86\n视频模式"},
                                {"name": "num_hands=2", "desc": "★ :87\n最多检测两只手"},
                                {"name": "min_hand_detection_confidence", "desc": "★ :88\n手部检测置信度阈值"},
                                {"name": "min_tracking_confidence", "desc": "★ :89\n追踪置信度阈值"},
                            ],
                        },
                        {
                            "name": "HandLandmarker",
                            "desc": "手部识别器“模具”：每只手输出 21 个关键点",
                            "children": [
                                {"name": "create_from_options(options)", "desc": "landmarks:90\n创建手部检测器实例"},
                                {"name": ".detect_for_video(...)", "desc": "landmarks:119\n每帧检测，返回 21 点"},
                                {"name": ".close()", "desc": "landmarks:156\n释放模型资源"},
                            ],
                        },
                    ],
                },
            ],
        },
    ],
}


def layout(node, depth=0, y_counter=[0], positions={}, edges=[], parent=None):
    """递归布局：叶子节点按顺序分配 y，父节点 y 取子节点平均值"""
    name = node["name"]
    desc = node.get("desc", "")
    children = node.get("children", [])
    if not children:
        y = y_counter[0]
        y_counter[0] += 1
        positions[id(node)] = (depth, y, name, desc, parent)
        return y
    child_ys = []
    for child in children:
        cy = layout(child, depth + 1, y_counter, positions, edges, node)
        child_ys.append(cy)
        edges.append((id(node), id(child)))
    y = sum(child_ys) / len(child_ys)
    positions[id(node)] = (depth, y, name, desc, parent)
    return y


positions, edges = {}, []
layout(tree, 0, [0], positions, edges)

fig, ax = plt.subplots(figsize=(30, 26))
ax.axis("off")

X_GAP, Y_GAP = 6.5, 2.2   # 间距：因为节点有两行字，调更大

# 1. 连线
for pid, cid in edges:
    d1, y1, _, _, _ = positions[pid]
    d2, y2, _, _, _ = positions[cid]
    ax.plot([d1 * X_GAP, d2 * X_GAP], [y1 * Y_GAP, y2 * Y_GAP],
            color="#b0b0b0", linewidth=1.0, zorder=1)

# 2. 节点：名称用粗体，作用用小字灰色
for d, y, name, desc, _ in positions.values():
    face, edge, text_color = PALETTE.get(name, DEFAULT_COLOR)
    x, y_pos = d * X_GAP, y * Y_GAP
    # 名称（第一行）
    ax.text(x, y_pos, name,
            fontsize=10, fontweight="bold",
            va="center", ha="center",
            color=text_color,
            bbox=dict(boxstyle="round,pad=0.55",
                      facecolor=face,
                      edgecolor=edge,
                      linewidth=1.3),
            zorder=3)
    # 作用（第二行，显示在框下方）
    if desc:
        ax.text(x, y_pos + 0.55, desc,
                fontsize=7.5, va="top", ha="center",
                color="#555555",
                zorder=3)

ax.set_xlim(-4, max(d for d, _, _, _, _ in positions.values()) * X_GAP + 4)
ax.set_ylim(-3, max(y for _, y, _, _, _ in positions.values()) * Y_GAP + 4)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("mediapipe_tree.png", dpi=150, bbox_inches="tight")
plt.show()