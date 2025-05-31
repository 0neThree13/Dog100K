import base64
from zhipuai import ZhipuAI
import os
from PIL import Image
import csv
import glob
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置路径
root_dir = 'dataset'
image_dir = ''
save_dir = os.path.join(root_dir, "data")
os.makedirs(save_dir, exist_ok=True)

# 获取所有图像路径
image_paths = glob.glob(os.path.join(image_dir, "**", "*.*"), recursive=True)
image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key="")

# 线程锁（用于写 CSV）
csv_lock = threading.Lock()

# CSV 路径
csv_path = os.path.join(root_dir, "captions.csv")

# 先写 CSV 表头
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "caption"])

# 处理函数
def process_image(idx, path):
    try:
        # 编码图像为 base64
        with open(path, 'rb') as img_file:
            img_base = base64.b64encode(img_file.read()).decode('utf-8')

        # 请求模型
        response = client.chat.completions.create(
            model="GLM-4V-Flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_base}
                        },
                        {
                            "type": "text",
                            "text": "Identify the dog's breed and briefly describe its appearance (such as fur color, expression, and posture) in clear and natural English."
                        }
                    ]
                }
            ]
        )

        caption = response.choices[0].message.content.strip()

        # 保存图像
        new_name = f"{idx:08d}.jpg"
        new_path = os.path.join(save_dir, new_name)
        image = Image.open(path).convert("RGB").copy()
        image.save(new_path)

        # 写入 CSV
        with csv_lock:
            with open(csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([new_name, caption])

        print(f"[{new_name}] → {caption}")

    except Exception as e:
        print(f"Error processing {path}: {e}")

# 使用线程池并发处理
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process_image, idx, path) for idx, path in enumerate(image_paths)]
    for _ in as_completed(futures):
        pass