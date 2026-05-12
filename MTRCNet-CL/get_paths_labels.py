import os
import pickle

root_dir = '/export/fs06/ahallur1/cholec80'
img_dir = os.path.join(root_dir, 'frames_1fps_250/original')
tool_dir = os.path.join(root_dir, 'tool_annotations')
phase_dir = os.path.join(root_dir, 'phase_annotations')

print("root_dir:", root_dir)
print("img_dir:", img_dir)
print("tool_dir:", tool_dir)
print("phase_dir:", phase_dir)


def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


def get_files(root_dir, suffix=None):
    file_paths = []
    file_names = []
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isfile(path):
            if item.startswith('._'):
                continue
            if suffix is not None and not item.endswith(suffix):
                continue
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


img_dir_names, img_dir_paths = get_dirs(img_dir)
tool_file_names, tool_file_paths = get_files(tool_dir, suffix='-tool.txt')
phase_file_names, phase_file_paths = get_files(phase_dir, suffix='-phase.txt')

print("num image dirs :", len(img_dir_names))
print("num tool files :", len(tool_file_names))
print("num phase files:", len(phase_file_names))
print("first 5 image dirs :", img_dir_names[:5])
print("first 5 tool files :", tool_file_names[:5])
print("first 5 phase files:", phase_file_names[:5])

assert len(img_dir_names) == 80, f"Expected 80 video folders, got {len(img_dir_names)}"
assert len(tool_file_names) == 80, f"Expected 80 tool files, got {len(tool_file_names)}"
assert len(phase_file_names) == 80, f"Expected 80 phase files, got {len(phase_file_names)}"

phase_dict = {
    'Preparation': 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderPackaging': 4,
    'CleaningCoagulation': 5,
    'GallbladderRetraction': 6
}
print("phase_dict:", phase_dict)

all_info_all = []

for j in range(len(tool_file_names)):
    with open(tool_file_paths[j], 'r') as tool_file:
        tool_lines = tool_file.readlines()

    with open(phase_file_paths[j], 'r') as phase_file:
        phase_lines = phase_file.readlines()

    info_all = []

    for i, tool_line in enumerate(tool_lines):
        if i == 0:
            continue  # skip header

        tool_split = tool_line.strip().split()
        if len(tool_split) < 8:
            continue

        frame_number = int(tool_split[0])        # 0, 25, 50, ...
        frame_idx_1fps = frame_number // 25      # 0, 1, 2, ...

        # Handle 6-digit frame naming: 000000.jpg, 000001.jpg, ...
        candidate_0 = os.path.join(img_dir_paths[j], f"{frame_idx_1fps:06d}.jpg")
        candidate_1 = os.path.join(img_dir_paths[j], f"{frame_idx_1fps + 1:06d}.jpg")

        if os.path.exists(candidate_0):
            img_file_each_path = candidate_0
        elif os.path.exists(candidate_1):
            img_file_each_path = candidate_1
        else:
            raise FileNotFoundError(
                f"Missing both {candidate_0} and {candidate_1} "
                f"(video={img_dir_names[j]}, tool frame={frame_number})"
            )

        tool_labels = [int(x) for x in tool_split[1:8]]

        # phase file has a header on line 0, then one line per original frame
        phase_line_idx = frame_number + 1
        if phase_line_idx >= len(phase_lines):
            raise IndexError(
                f"Phase annotation too short for {img_dir_names[j]} at frame {frame_number}"
            )

        phase_split = phase_lines[phase_line_idx].strip().split()
        if len(phase_split) < 2:
            raise ValueError(
                f"Malformed phase annotation line in {phase_file_paths[j]} "
                f"at frame {frame_number}: {phase_lines[phase_line_idx]}"
            )

        phase_name = phase_split[1]

        if phase_name not in phase_dict:
            raise KeyError(f"Unknown phase '{phase_name}' in {phase_file_paths[j]}")

        phase_label = phase_dict[phase_name]

        # Keep repo-style label format: 7 tool labels + 1 phase label
        info_each = [img_file_each_path] + tool_labels + [phase_label]
        info_all.append(info_each)

    print(f"{img_dir_names[j]} -> {len(info_all)} samples")
    all_info_all.append(info_all)

with open('cholec80.pkl', 'wb') as f:
    pickle.dump(all_info_all, f)

train_file_paths = []
val_file_paths = []
test_file_paths = []

train_labels = []
val_labels = []
test_labels = []

train_num_each = []
val_num_each = []
test_num_each = []

# Original repo split: 32 train / 8 val / 40 test
for i in range(32):
    train_num_each.append(len(all_info_all[i]))
    for row in all_info_all[i]:
        train_file_paths.append(row[0])
        train_labels.append(row[1:])

for i in range(32, 40):
    val_num_each.append(len(all_info_all[i]))
    for row in all_info_all[i]:
        val_file_paths.append(row[0])
        val_labels.append(row[1:])

for i in range(40, 80):
    test_num_each.append(len(all_info_all[i]))
    for row in all_info_all[i]:
        test_file_paths.append(row[0])
        test_labels.append(row[1:])

print("train videos  :", len(train_num_each))
print("val videos    :", len(val_num_each))
print("test videos   :", len(test_num_each))
print("train samples :", len(train_file_paths))
print("val samples   :", len(val_file_paths))
print("test samples  :", len(test_file_paths))

train_val_test_paths_labels = [
    train_file_paths,
    val_file_paths,
    test_file_paths,
    train_labels,
    val_labels,
    test_labels,
    train_num_each,
    val_num_each,
    test_num_each
]

with open('train_val_test_paths_labels.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)

print("Wrote cholec80.pkl and train_val_test_paths_labels.pkl")