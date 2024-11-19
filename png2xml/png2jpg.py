import os


def rename_png_to_jpg(directory_path):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory_path):
        # 检查文件是否是 .png 后缀
        if filename.endswith('.png'):
            # 构造新的文件名（将 .png 改为 .jpg）
            new_filename = filename.replace('.png', '.jpg')

            # 构造原始文件的完整路径和新文件的完整路径
            old_filepath = os.path.join(directory_path, filename)
            new_filepath = os.path.join(directory_path, new_filename)

            # 重命名文件
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")


# 使用示例
directory_path = 'F:\codes\github上传\png2xml\AEBAD\ABlation'  # 替换为你的文件夹路径
rename_png_to_jpg(directory_path)
