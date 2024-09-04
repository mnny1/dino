from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import h5py, json, os, cv2
import timm
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class TokenSelectionWrapper(nn.Module):
    def __init__(self, model, student: bool, patch_size=16, padding=1):
        super(TokenSelectionWrapper, self).__init__()
        self.model = model
        self.student = student
        self.patch_size = patch_size
        self.padding = padding
        self.px_threshold = int(patch_size**2 * 0.15)
        self.tk_threshold = 0.65
        
    def __dilate_tensor_mask(self, mask, padding):
        window = 2 * padding + 1
        kernel = torch.ones((1, 1, window, window), dtype=torch.float32).cuda()
        mask = mask.unsqueeze(1).float()
        dilated_mask = F.conv2d(mask.float(), kernel, padding=window // 2)
        dilated_mask = torch.clamp(dilated_mask, 0, 1)
        return dilated_mask.permute(0, 2, 3, 1)
    
    def __mask_above_threshold(self, mask_batch, threshold):
        reshaped_masks = rearrange(mask_batch, 'b h w c -> b (h w) c')
        condition = torch.any(reshaped_masks, dim=-1).sum(dim=-1) < threshold
        mask_batch[condition] = 1
        return mask_batch

    def forward(self, x, masks):
        if self.student:
            masks = map(lambda t: rearrange(t, 'b (ht p1) (wt p2) -> b ht wt (p1 p2)', p1=self.patch_size, p2=self.patch_size), masks) # mask tokenized : len=3 [bx14x14x256, bx14x14x256, bx14x14x256]
            masks = map(lambda t: self.__mask_above_threshold(t, int(t.shape[1]**2 * self.tk_threshold)), masks) # assign 1 to every mask under tk_threshold
            masks = map(lambda t: (t.sum(dim=-1) >= self.px_threshold).float(), masks) # mask tokens selected : len=3, [[4,14,14], [4,14,14], [4,14,14]]
            masks = map(lambda t: self.__dilate_tensor_mask(t, self.padding) , masks) # mask tokens dilated : len=3, [[4,14,14,1], [4,14,14,1], [4,14,14,1]]
            x = map(lambda t: rearrange(t, 'b c (ht p1) (wt p2) -> b ht wt (p1 p2 c)', p1=self.patch_size, p2=self.patch_size), x) # img tokenized : [[bx14x14x768], [bx14x14x768]]
            x = map(lambda z, y: z * y, x, masks) # img tokens selected
            x = list(map(lambda t: rearrange(t, 'b ht wt (p1 p2 c) -> b c (ht p1) (wt p2)', p1=self.patch_size, p2=self.patch_size), x)) # img.shape restored : [[b, 3, 224, 224], [b, 3, 224, 224], [b, 3, 224, 224]]
        return self.model(x)


class NCTDataset(Dataset):
    def __init__(self, img_dir, file_extension,json_dir, transform:list =None):
        self.img_paths = sorted(Path(img_dir).glob(f"**/*{file_extension}"))#[:64] # ADI total: 10407
        self.img_list = [Image.open(path) for path in self.img_paths]  
        self.mask_dict = self.__read_bbox(json_dir)
        self.transform = transform   
                
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        mask = self.mask_dict[str(self.img_paths[idx]).split("/")[-1]]
                
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask']
                
        return img, mask
    
    
    def __read_bbox(self, root_dir):
        json_files = sorted(Path(root_dir).glob("**/*.json"))
        mask_dict = {}
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for item in data:
                file_name = item["file_name"]
                height = item["height"]
                width = item["width"]
                mask = np.zeros((height, width), dtype=np.uint8)
                for bbox in item.get("bbox", []):
                    x1, y1, x2, y2 = bbox["coordinates"]
                    mask[y1:y2, x1:x2] = 255
                mask_dict[file_name] = Image.fromarray(mask)
        #### DEBUG #####################
        #    break
        return mask_dict


# class NCTDataset(Dataset):
#     """
#     dataset for patches e.g NCT-CRC-100k
#     """
#     def __init__(self, img_dir, file_extension,json_dir, transform:list =None):              
#         self.img_paths = sorted(Path(img_dir).glob(f"**/*{file_extension}"))
#         self.img_list = [Image.open(path) for path in self.img_paths]
#         self.transform = transform
                        
#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, idx):
#         img = self.img_list[idx]                
#         if self.transform:
#             img = self.transform(img)
#         return img, 0

# class UNI_cell(nn.Module):
#     def __init__(self, model_path, freeze_blocks=[22, 23, 24]):
#         super(UNI_cell, self).__init__()
#         self.model = timm.create_model(
#                 "vit_large_patch16_224", 
#                 img_size=224, 
#                 patch_size=16, 
#                 init_values=1e-5, 
#                 num_classes=0, 
#                 dynamic_img_size=True
#         )
#         self.model.load_state_dict(torch.load(model_path), strict=True)
#         self.embed_dim = self.model.embed_dim  
        
#         for name, param in self.model.named_parameters():
#             if any(f"blocks.{i}." in name for i in freeze_blocks):
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#         print(f"All layers frozen except blocks {freeze_blocks}")
    
#     def forward(self, x):
#         return self.model(x)