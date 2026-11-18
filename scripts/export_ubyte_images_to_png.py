import os
import struct
import numpy as np
from PIL import Image

def export_mnist_samples(ubyte_img_path, ubyte_lbl_path, output_dir, num_samples_per_digit=1):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(ubyte_img_path, 'rb') as f_img, open(ubyte_lbl_path, 'rb') as f_lbl:
        # Read Image header: magic (4 bytes), num_images (4 bytes), rows (4 bytes), cols (4 bytes)
        magic_img, num_images, rows, cols = struct.unpack('>IIII', f_img.read(16))
        # Read Label header: magic (4 bytes), num_labels (4 bytes)
        magic_lbl, num_labels = struct.unpack('>II', f_lbl.read(8))
        
        print(f"[+] Reading {ubyte_img_path}")
        print(f"    Magic: {magic_img}, Images: {num_images}, Dimensions: {rows}x{cols}")
        
        # Read raw buffer
        img_buffer = np.frombuffer(f_img.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        lbl_buffer = np.frombuffer(f_lbl.read(), dtype=np.uint8)
    
    digit_counts = {d: 0 for d in range(10)}
    exported_records = []
    
    for idx, (img, lbl) in enumerate(zip(img_buffer, lbl_buffer)):
        if digit_counts[lbl] < num_samples_per_digit:
            digit_counts[lbl] += 1
            
            # 1. Clean Image
            clean_pil = Image.fromarray(img, mode='L')
            clean_filename = f"digit_{lbl}_sample_clean.png"
            clean_path = os.path.join(output_dir, clean_filename)
            # Scale up for easy viewing (28x28 -> 140x140 nearest neighbor)
            clean_pil.resize((140, 140), Image.NEAREST).save(clean_path)
            
            # 2. Watermarked Image (Inject 4x4 patch at (24, 24))
            wm_img = img.copy()
            # 4x4 high-contrast checkerboard/patch in bottom-right corner
            wm_img[24:28, 24:28] = 255
            wm_pil = Image.fromarray(wm_img, mode='L')
            wm_filename = f"digit_{lbl}_sample_watermarked.png"
            wm_path = os.path.join(output_dir, wm_filename)
            wm_pil.resize((140, 140), Image.NEAREST).save(wm_path)
            
            exported_records.append((lbl, clean_path, wm_path))
            
        if all(c >= num_samples_per_digit for c in digit_counts.values()):
            break
            
    # Create a Master Grid (Digits 0-9 Clean vs Watermarked)
    # 2 rows, 10 columns
    grid_img = Image.new('RGB', (140 * 10, 140 * 2), color=(15, 23, 42))
    
    for d, clean_p, wm_p in exported_records:
        digit_idx = int(d)
        c_im = Image.open(clean_p).convert('RGB')
        w_im = Image.open(wm_p).convert('RGB')
        grid_img.paste(c_im, (digit_idx * 140, 0))
        grid_img.paste(w_im, (digit_idx * 140, 140))
        
    grid_path = os.path.join(output_dir, "mnist_comparison_grid.png")
    grid_img.save(grid_path)
    print(f"[✓] Successfully exported {len(exported_records)} samples to {output_dir}")
    print(f"[✓] Master comparison grid saved to: {grid_path}")

if __name__ == '__main__':
    ubyte_img = r"D:\MRINAL\minor\data_cache\MNIST\raw\t10k-images-idx3-ubyte"
    ubyte_lbl = r"D:\MRINAL\minor\data_cache\MNIST\raw\t10k-labels-idx1-ubyte"
    out_dir = r"D:\MRINAL\minor\research_outputs\exported_mnist_samples"
    export_mnist_samples(ubyte_img, ubyte_lbl, out_dir)
