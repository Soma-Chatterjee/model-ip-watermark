import os
import torch
import torchvision
from PIL import Image
import numpy as np

def export_cifar10_samples():
    out_dir = r"D:\MRINAL\minor\research_outputs\exported_cifar10_samples"
    os.makedirs(out_dir, exist_ok=True)
    
    test_set = torchvision.datasets.CIFAR10(root='./data_cache', train=False, download=False)
    classes = test_set.classes
    
    found = {}
    for img, lbl in test_set:
        cls_name = classes[lbl]
        if cls_name not in found and len(found) < 10:
            found[cls_name] = img
        if len(found) == 10:
            break
            
    # Export clean and watermarked color samples
    grid_img = Image.new('RGB', (160 * 10, 160 * 2), color=(15, 23, 42))
    
    for idx, (cls_name, pil_img) in enumerate(found.items()):
        # 1. Clean resized for viewing
        clean_large = pil_img.resize((160, 160), Image.NEAREST)
        clean_large.save(os.path.join(out_dir, f"{cls_name}_clean.png"))
        
        # 2. Watermarked (4x4 patch in bottom-right corner)
        wm_arr = np.array(pil_img).copy()
        # High-contrast cryptographic color patch (cyan/yellow)
        wm_arr[28:32, 28:32, :] = [0, 255, 255]
        wm_pil = Image.fromarray(wm_arr).resize((160, 160), Image.NEAREST)
        wm_pil.save(os.path.join(out_dir, f"{cls_name}_watermarked.png"))
        
        grid_img.paste(clean_large, (idx * 160, 0))
        grid_img.paste(wm_pil, (idx * 160, 160))
        
    grid_path = os.path.join(out_dir, "cifar10_clean_vs_watermarked_grid.png")
    grid_img.save(grid_path)
    print(f"[✓] Exported 10 natural CIFAR-10 color samples to: {out_dir}")
    print(f"[✓] Saved color comparison grid to: {grid_path}")

if __name__ == '__main__':
    export_cifar10_samples()
