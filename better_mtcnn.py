
from facenet_pytorch import MTCNN
from facenet_pytorch.models.mtcnn import fixed_image_standardization
from facenet_pytorch.models.utils.detect_face import get_size,save_img,extract_face
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def crop_resize(img, box, image_size):
    if isinstance(img, torch.Tensor):
        #print(" crop resize img size",img.size())
        if img.size()[-1]==3:
            img=img.permute(2,0,1)
        # Crop the image using slicing (maintaining the computation graph)
        img = img[:,box[1]:box[3], box[0]:box[2]]  # Assumes img is in CxHxW format

        # Resize using torch's interpolate function (differentiable)
        # Ensure img is in NCHW format for interpolation
        img = img.unsqueeze(0) #.float()  # Add batch dimension (N, C, H, W)
        out = F.interpolate(img, size=(image_size, image_size), mode='bilinear', align_corners=False)
        out = out.squeeze(0) #.byte()  # Remove batch dimension and convert back to byte if needed
        #print(" crop resize img size after",img.size())
    else:
        # For non-tensor images (PIL or NumPy), retain original behavior but avoid breaking gradients
        if isinstance(img, np.ndarray):
            img = img[box[1]:box[3], box[0]:box[2]]
            img = torch.tensor(img, dtype=torch.float32)
            out = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(image_size, image_size), mode='bilinear', align_corners=False)
            out = out.squeeze(0).squeeze(0).byte()  # Remove batch and channel dimensions
        else:
            out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)

    return out

def extract_face_pt(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    #print("extract face_pt line 53", img.dtype, img.device)
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)
    #print("extract face_pt line 67", face.dtype, face.device)
    return face


class BetterMTCNN(MTCNN):
    def extract(self, img, batch_boxes, save_path):
        '''if type(img)==torch.Tensor:
            print("extract line 72", img.dtype, img.device)'''
        # Determine if a batch or single image was passed
        batch_mode = True
        if (
                not isinstance(img, (list, tuple)) and
                not (isinstance(img, np.ndarray) and len(img.shape) == 4) and
                not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_mode = False

        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]

        # Process all bounding boxes
        faces = []
        for im, box_im, path_im in zip(img, batch_boxes, save_path):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                if type(im)==torch.Tensor:
                    face=extract_face_pt(im, box, self.image_size, self.margin, face_path)
                else:
                    face = extract_face(im, box, self.image_size, self.margin, face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]
            #("face im size",faces_im.size())
            faces.append(faces_im)
            '''if type(faces_im)==torch.Tensor:
                print("extract line 125", faces_im.dtype, faces_im.device)'''
        
        if not batch_mode:
            faces = faces[0]
        return faces
