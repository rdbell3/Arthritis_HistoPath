import random
import torch
import numbers
from torchvision.transforms import functional as F
import torchvision

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# def _flip_coco_person_keypoints(kps, width):
#     flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
#     flipped_data = kps[:, flip_inds]
#     flipped_data[..., 0] = width - flipped_data[..., 0]
#     # Maintain COCO convention that if visibility == 0, then x, y = 0
#     inds = flipped_data[..., 2] == 0
#     flipped_data[inds] = 0
#     return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


# class RandomHorizontalFlip(object):
#     def __init__(self, prob):
#         self.prob = prob

#     def __call__(self, image):
#         if random.random() < self.prob:
#             height, width = image.shape[-2:]
#             image = image.flip(-1)
#             bbox = target["boxes"]
#             bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
#             target["boxes"] = bbox
#             if "masks" in target:
#                 target["masks"] = target["masks"].flip(-1)
#             if "keypoints" in target:
#                 keypoints = target["keypoints"]
#                 keypoints = _flip_coco_person_keypoints(keypoints, width)
#                 target["keypoints"] = keypoints
#         return image


class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image


# class ColorJitter(object):
#     """Randomly change the brightness, contrast and saturation of an image.

#     Args:
#         brightness (float or tuple of float (min, max)): How much to jitter brightness.
#             brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
#             or the given [min, max]. Should be non negative numbers.
#         contrast (float or tuple of float (min, max)): How much to jitter contrast.
#             contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
#             or the given [min, max]. Should be non negative numbers.
#         saturation (float or tuple of float (min, max)): How much to jitter saturation.
#             saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
#             or the given [min, max]. Should be non negative numbers.
#         hue (float or tuple of float (min, max)): How much to jitter hue.
#             hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
#             Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
#     """
#     def __init__(self, brightness, contrast, saturation):#, hue):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         #self.hue = hue

    
#     def __call__(self,image):
#         transformation = []
        
#         brightness_factor = random.uniform(1,1+self.brightness) 
#         transformation.append(Lambda(lambda image: F.adjust_brightness(image, brightness_factor)))

#         contrast_factor = random.uniform(1, 1+self.contrast) 
#         transformation.append(Lambda(lambda image: F.adjust_contrast(image, contrast_factor)))

#         saturation_factor = random.uniform(1-self.saturation,1+ self.saturation) 
#         transformation.append(Lambda(lambda image: F.adjust_contrast(image, saturation_factor)))

#         #hue_factor = random.uniform(0, self.hue) 
#         #transformation.append(Lambda(lambda image: F.adjust_contrast(image, hue_factor)))

#         random.shuffle(transformation)
#         trans = torchvision.transforms.Compose(transformation)
        
#         image = trans(image)

#         return image

class Normalization(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self,image):
        
        img = F.normalize(image, self.mean, self.std)

        return img  
