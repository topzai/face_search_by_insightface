import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from typing import Tuple
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from tqdm import tqdm

import subprocess
process = subprocess.Popen(
    ['milvus-server.exe'],
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,  # 合并错误输出到标准输出
    text=True  # 以文本模式读取输出
)
while True:
    line = process.stdout.readline()
    if not line and process.poll() is not None:
        break
    if line:
        print(line, end='')  # 输出每一行日志
        if "Ctrl+C to exit ..." in line:
            break  # 检测到关键字后退出循环，但不关闭进程

print("Milvus server 已启动，日志监听结束。")

# 加载人脸模型
face_app = FaceAnalysis(name='buffalo_l',root='./models')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# 6. 检索最相似的2个人脸
def match(file):
    sample_imgs =[]
    sample_texts =[]
    embedding_query, info_query = extract_face_embedding(file)
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    query_vectors = [embedding_query]

    results = collection.search(
        data=query_vectors,
        anns_field="embedding",
        param=search_params,
        limit=3,
        output_fields=["id", "age", "gender","file_name"]
    )

    print("\nTop-5 相似人脸信息:")

    for i, hits in enumerate(results):
        for j, hit in enumerate(hits):
            sample_imgs.append(hit.file_name)
            sample_texts.append(f"相似度:{int((1000-hit.distance)/10)}% 预测年龄:{hit.entity.get('age')}")

            print(
                f"{j+1}: 距离={hit.distance:.4f} | id={hit.id} | age={hit.entity.get('age')} | gender={hit.entity.get('gender')} | file_name={hit.file_name}"
            )

    return sample_imgs,sample_texts

def extract_face_embedding(image_path: str) -> Tuple[np.ndarray, dict]:
    """
    提取人脸特征向量
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError("未检测到人脸")
    elif len(faces) > 1:
        print("检测到多个人脸，使用第一个人脸")
    face = faces[0]
    embedding = face.embedding
    face_info = {
        'bbox': face.bbox,
        'gender': face.gender,
        'age': face.age
    }
    return embedding, face_info

def into_face(file_name):
    try:
        embedding, face_info = extract_face_embedding(image_path=f'images/{file_name}')
    except ValueError as e:
        print(str(e))
        return False

    insert_data = [
        [embedding.tolist()],            # embedding (512维)
        [int(face_info['age'])],         # age
        [int(face_info['gender'])],      # gender
        [file_name],  # FILENAME
    ]
    collection.insert(insert_data)
    print(f"{face_info}写入成功！")



# 连接本地 Milvus
connections.connect("default", host="127.0.0.1", port="19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="age", dtype=DataType.INT64),
    FieldSchema(name="gender", dtype=DataType.INT64),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
]
schema = CollectionSchema(fields, description="Face Embedding Collection")
collection_name = "face_embeddings"

if collection_name in utility.list_collections():
    collection = Collection(collection_name)
else:
    collection = Collection(collection_name, schema)


index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}

collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

results = collection.query(
    expr="id >= 0",
    output_fields=["file_name"],
    # limit=16384   # 根据实际情况调整
)
db_file_names = set(row['file_name'] for row in results)

def read_pth_images(directory):
    # 指定目录
    # 支持的图片扩展名（注意 .jpeg 拼写）
    extensions = ('.jpeg', '.jpg', '.png')
    # 获取所有图片文件
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(extensions)]
    not_in_db = []
    for img in image_files:
        if img not in db_file_names:
            not_in_db.append(img)
    return not_in_db


directory='images'
not_in_db = read_pth_images(directory)
for i in tqdm(not_in_db, desc="处理文件"):
    if into_face(i) is False:
        file_path = f'{directory}/{i}'
        tqdm.write(f"删除无数据文件：{file_path}")
        os.remove(file_path)



print("**********************")
print("*     人像查找工具    *")
print("**********************")

# collection.release()




class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("人像相似度查询")
        self.open_file_path = None  # 查询的图片路径
        # 主frame
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)

        # 上方图片区域
        self.top_img_label = tk.Label(main_frame)
        self.top_img_label.grid(row=0, column=0, sticky='nsew')

        # 右侧按钮区域
        btn_frame = tk.Frame(main_frame)
        btn_frame.grid(row=0, column=1, padx=(10, 0), sticky='ns')

        open_btn = tk.Button(btn_frame, text="打开", command=self.open_image)
        open_btn.pack(fill='x', pady=(0, 5))

        query_btn = tk.Button(btn_frame, text="查询", command=self.query_images)
        query_btn.pack(fill='x')

        # 下方结果区域（3个子frame，每个有图和详情）
        result_frame = tk.Frame(main_frame)
        result_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))

        self.result_img_labels = []
        self.result_text_labels = []

        for i in range(3):
            subframe = tk.Frame(result_frame)
            subframe.pack(side='left', padx=5)

            img_lbl = tk.Label(subframe)
            img_lbl.pack()
            text_lbl = tk.Label(subframe, text="", wraplength=100)  # wraplength可调整文字宽度换行
            text_lbl.pack()

            self.result_img_labels.append(img_lbl)
            self.result_text_labels.append(text_lbl)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((200, 200))  # 保持比例，最大边不超过200像素
            self.top_img_tk = ImageTk.PhotoImage(img)
            self.open_file_path = file_path
            self.top_img_label.config(image=self.top_img_tk)


    def query_images(self):
        sample_imgs,sample_texts = match(self.open_file_path)
        # # 示例：三张本地图片及其详情，可替换为你的实际逻辑和路径列表、详情列表
        # sample_imgs = ["img1.jpg", "img2.jpg", "img3.jpg"]  # 图片路径列表
        # sample_texts = ["这是第一张图片的描述", "第二张图说明信息", "第三张图的详细内容"]  # 每张图对应的详情

        for i in range(3):
            try:
                img = Image.open(f'images/{sample_imgs[i]}')
                img.thumbnail((100, 100))  # 保持比例，最大边不超过100像素
                img_tk = ImageTk.PhotoImage(img)
                self.result_img_labels[i].config(image=img_tk)
                self.result_img_labels[i].image = img_tk  # 防止被垃圾回收

                detail_text = sample_texts[i]
                self.result_text_labels[i].config(text=detail_text)

            except Exception as e:
                print(f"加载图片失败: {e}")
                self.result_img_labels[i].config(image='', text="无")
                self.result_text_labels[i].config(text="加载失败或无描述")


if __name__ == "__main__":


    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
