import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import os
import shutil
import cv2
import albumentations as A

TARGET_TOTAL_IMAGES = 70 #จำนวนภาพทั้งหมดทีก่อนการทำ augmentation
NEW_IMAGES = 10  # จำนวนภาพใหม่ที

# --- Load old data ---
df_old = pd.read_csv('./old_data/trainset.csv') # ชื่อไฟล์ CSV เก่า
df_new = pd.read_csv('./new_data/new_train.csv') # ชื่อไฟล์ CSV ใหม่
df_test = pd.read_csv('./test_data/test.csv') # ชื่อไฟล์ CSV ใหม่
df_notpra = pd.read_csv('./notpra/trainnotpra.csv') # ชื่อไฟล์ CSV ใหม่ที่ไม่ใช่พระ

old_counts = df_old['name'].value_counts()
pra_names = old_counts.index.tolist()
pra_image_counts = old_counts.values
num_classes = len(pra_names)
total_notpra = 0
# --- โฟลเดอร์ภาพ ---
SOURCE_IMAGE_OLD = 'old_data/old_data/'  # โฟลเดอร์พระเก่า
SOURCE_IMAGE_NEW = 'new_data/new_data/'  # โฟลเดอร์พระใหม่
SOURCE_IMAGE_TEST = 'test_data/test_img/'  # โฟลเดอร์เทส
SOURCE_IMAGE_NOTPRA = 'notpra/notpra/'  # โฟลเดอร์ไม่ใช่พระ

EXPORT_DIR = 'combined'             # โฟลเดอร์รวมใหม่กับเก่า
IMAGE_EXPORT_DIR = os.path.join(EXPORT_DIR, 'train_new') # สร้างเส้นทางโฟลเดอร์ภาพเทรน
IMAGE_EXPORT_TEST = os.path.join(EXPORT_DIR, 'images_test') # สร้างเส้นทางโฟลเดอร์ภาพเทส
os.makedirs(IMAGE_EXPORT_DIR, exist_ok=True) # สร้างทางโฟลเดอร์ภาพเทรน
os.makedirs(IMAGE_EXPORT_TEST, exist_ok=True) # สร้างโฟลเดอร์ภาพเทส
df_export = df_new.copy()
df_export_test = df_test.copy()


transform = A.Compose([
    A.Rotate(limit=360, p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.2),
    # A.RandomCrop(width=200, height=200, p=0.2),
    A.RandomScale(scale_limit=(0.8, 1.2), p=0.2),
])

def augment_image(row, src_dir, export_dir, transform, n=2): 
    new_rows = []
    original_path = os.path.join(src_dir, row['filename'])

    image = cv2.imread(original_path)
    if image is None:
        print(f"❌ อ่านภาพไม่ได้: {original_path}")
        return []

    for i in range(n):
        transformed = transform(image=image)
        aug_image = transformed['image']
#jpg
        base_name = os.path.splitext(row['filename'])[0]  # เช่น 'train_15'
        new_filename = f"{base_name}_copy{i+1}.png"
        new_path = os.path.join(export_dir, new_filename)

        cv2.imwrite(new_path, aug_image)

        # คัดลอกข้อมูลเดิมมาเปลี่ยนชื่อไฟล์
        new_row = row.copy()
        new_row['filename'] = new_filename
        new_rows.append(new_row)
        print(f"✅ Augmented: {new_filename} จาก {row['filename']}")
    return new_rows

def init_individual():
    return [random.randint(1, count) for count in pra_image_counts]

def fitness(individual):
    total = sum(individual) + NEW_IMAGES
    if total > TARGET_TOTAL_IMAGES:
        return -1000,
    
    diversity_score = len([i for i in individual if i > 0]) / num_classes  # อยากได้พระหลายองค์
    balance_score = -np.std(individual)  # รูปไม่ต่างกันมาก
    return diversity_score * 100 + balance_score,

def custom_mutate(individual): #สุ่มจำนวนพระแต่ละองค์
    for i in range(len(individual)):
        if random.random() < 0.3:  # indpb = 0.2
            max_available = pra_image_counts[i]
            if max_available >= 50:
                individual[i] = random.randint(50, max_available)
            elif max_available > 0:
                individual[i] = random.randint(1, max_available)
            else:
                individual[i] = 0
    return individual,

def copy_images(df, source_dir, export_dir):
    for idx, row in df.iterrows():
        image_file = row['filename']
        src_path = os.path.join(source_dir, image_file)
        dst_path = os.path.join(export_dir, image_file)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            print(f"✅ คัดลอก: {image_file} ไปยัง {export_dir}")
        else:
            print(f"❗ ไม่พบภาพ: {src_path} - {source_dir}")

def augment_folder(df,csv_path,IMAGE_EXPORT_DIR,transform,n):
    augmented_rows = []
    for idx, row in df.iterrows():
        new_rows = augment_image(row, IMAGE_EXPORT_DIR, IMAGE_EXPORT_DIR, transform, n=n)
        augmented_rows.extend(new_rows)
    
    if augmented_rows:
        df_augmented = pd.DataFrame(augmented_rows)
        df_combined = pd.concat([df, df_augmented], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"\n🧪 Augmentation เสร็จแล้ว เพิ่ม {len(df_augmented)} ภาพ")
        print(f"📄 อัปเดต CSV แล้วที่: {csv_path}")

if len(old_counts) > 0:
    print(f"📊 มีพระทั้งหมด {num_classes} องค์")

    # --- Genetic Algorithm Setup ---
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("individual", tools.initIterate, creator.Individual, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --- Fitness Function ---
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", custom_mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Run GA ---
    population = toolbox.population(n=50)
    result = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.3, ngen=40, verbose=False)

    # --- Best Individual ---
    best = tools.selBest(population, k=1)[0]
    selected_pra = [(pra_names[i], best[i]) for i in range(num_classes) if best[i] > 0]

    # --- Export selected subset ---
    df_sampled_list = []
    for name, num in selected_pra:
        df_subset = df_old[df_old['name'] == name]
        if len(df_subset) >= num:
            df_sampled = df_subset.sample(num, random_state=42)
            df_sampled_list.append(df_sampled)

    if df_sampled_list:
        df_sampled_ga = pd.concat(df_sampled_list, ignore_index=True)

        print(f"🎯 Selected {len(df_sampled_ga)} old images from GA")
        print("🧠 พระที่เลือกมา:")
        for name, num in selected_pra:
            print(f" - {name}: {num} รูป")
            total_notpra = num

        # คัดลอกภาพเก่า
        copy_images(df_sampled_ga, SOURCE_IMAGE_OLD, IMAGE_EXPORT_DIR)

        # รวมข้อมูลเก่าที่คัดเลือกไว้กับพระใหม่
        df_export = pd.concat([df_sampled_ga, df_new], ignore_index=True)
    else:
        print("⚠️ GA ไม่ได้เลือกภาพใดจากพระเก่า ใช้เฉพาะภาพใหม่เท่านั้น")

else:
    print("⚠️ รันครังแรกไม่มีพระเก่า")

# --- คัดลอกภาพใหม่ ---
copy_images(df_new, SOURCE_IMAGE_NEW, IMAGE_EXPORT_DIR)

# --- บันทึก CSV พร้อมทุกคอลัมน์จาก df_sampled_ga ---
csv_path = os.path.join(EXPORT_DIR, 'trainset.csv')
df_export.to_csv(csv_path, index=False)

df_combined = pd.read_csv(csv_path)
# --- ทำ Augmentation ไฟล์ train---
augment_folder(df_combined, csv_path, IMAGE_EXPORT_DIR, transform, n=100)  # augment ภาพ n = จำนวนที่อยากให้เพิ่มของtrain


# ----สุ่มnotpraที่จะใช้ ----
df_selected_notpra = df_notpra.sample(n=total_notpra*100, random_state=42)
print(f"📈 ได้ notpra รวม {len(df_selected_notpra)} รูป (target = {total_notpra})")
# คัดลอกภาพ notpra ไปยังโฟลเดอร์ train_new
copy_images(df_selected_notpra, SOURCE_IMAGE_NOTPRA, IMAGE_EXPORT_DIR)

# รวมข้อมูล notpra เข้ากับ df_export
df_existing = pd.read_csv(csv_path)
df_export_combined = pd.concat([df_existing, df_selected_notpra], ignore_index=True)
df_export_combined.to_csv(csv_path, index=False)

#  ---- สร้างโฟลเดอร์เทส -----------
copy_images(df_test, SOURCE_IMAGE_TEST, IMAGE_EXPORT_TEST)

csv_test_path = os.path.join(EXPORT_DIR, 'test.csv')
df_export_test.to_csv(csv_test_path, index=False)
df_combined_test = pd.read_csv(csv_test_path)

# --- ทำ Augmentation สำหรับภาพเทส ---
augment_folder(df_combined_test, csv_test_path, IMAGE_EXPORT_TEST, transform, n=10)  # augment ภาพ n = จำนวนที่อยากให้ของtest

print(f"\n✅ คัดลอกรูปภาพเรียบร้อยไปยัง: {EXPORT_DIR}")
print(f"📄 สร้างไฟล์ CSV พร้อมทุกคอลัมน์แล้วที่: {csv_path}")