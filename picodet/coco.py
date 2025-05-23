import copy
import os
import warnings
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms,utils



class COCODataset(Dataset):

    def __init__(self, im_root, anno_path, to_size,
                 device,transform=None,
                 load_crowd=False,):
        self.img_root = im_root
        self.to_size = to_size
        self.device = device
        self.transform = transform
        self.load_crowd = load_crowd
        self.load_image_only = False
        self.catid2clsid, self.cname2cid = {}, {}
        self.n_records = 0
        self.records = self.parse_coco(anno_path)
        self._cur_iter = 0

    def parse_coco(self, anno_path):
        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        n_records = 0

        self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        self.cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in self.catid2clsid.items()
        })

        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            warnings.warn('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = f"{self.img_root}/{im_fname}" \
                if self.img_root else f"{im_fname}"
            if not os.path.exists(im_path):
                warnings.warn('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue
            if im_w < 0 or im_h < 0:
                warnings.warn('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_id': torch.tensor([img_id]),
                'im_path': im_path,
                'im_shape':torch.tensor([im_h, im_w]),
                'scale_factor':torch.tensor([1., 1.])
            }

            empty_image = False
            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'bbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['bbox'])):
                            continue
                    x1,y1,w,h = inst['bbox']
                    x2 = x1 +w
                    y2 = y1+h
                    eps = 1e-5
                    if w > eps and h > eps:
                        inst['clean_bbox'] =torch.tensor([
                            max(0,min(x1,im_w)),
                            max(0,min(y1,im_h)),
                            max(0,min(x2,im_w)),
                            max(0,min(y2,im_h))
                        ], dtype= torch.float32)
                        bboxes.append(inst)
                    else:
                        warnings.warn(
                            'Found an invalid bbox in annotations: im_id: {}, '
                            'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                                img_id, float(inst['area']), x1, y1, x2, y2))
                num_bbox = len(bboxes)
                if num_bbox<=0:
                    empty_image=True

                gt_bbox = torch.zeros((num_bbox,4),
                                      dtype = torch.float32)
                gt_class = torch.zeros((num_bbox,1),
                                       dtype = torch.int64)
                is_crowd = torch.zeros((num_bbox,1),
                                       dtype = torch.int64)

                for i, box in enumerate(bboxes):
                    catid = box['category_id']
                    gt_class[i][0] = self.catid2clsid[catid]
                    gt_bbox[i, :] = box['clean_bbox']
                    is_crowd[i][0] = box['iscrowd']
                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                }
                coco_rec.update(gt_rec)

            if not empty_image or self.load_image_only:
                records.append(coco_rec)
                n_records += 1
        assert n_records > 0, 'not found any coco record in %s' % (anno_path)
        print('{} samples in file {}'.format(n_records, anno_path))
        self.n_records = n_records
        return records

    def __len__(self):
        return self.n_records

    def __getitem__(self, idx):
        # edited getitem
        tmp = copy.deepcopy(self.records[idx])
        targets ={}
        # calculate scale_factor
        to_h, to_w = self.to_size
        ori_h, ori_w = tmp['im_shape']
        scale_factor = torch.tensor([to_h / ori_h, to_w / ori_w])

        # get image
        image = Image.open(tmp['im_path'])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # apply image augmentation
        if self.transform:
            # decode image to rgb numpy form
            image = np.array(image)
            
            # Convert tensor classes to primitive python integers (not tensors or numpy arrays)
            gt_classes = []
            for i in range(tmp['gt_class'].shape[0]):
                # Extract each class as a simple Python integer
                gt_classes.append(int(tmp['gt_class'][i][0].item()))
            
            # Convert bounding boxes to numpy array properly
            bbox_numpy = []
            for i in range(tmp['gt_bbox'].shape[0]):
                # Convert each bbox to a list of floats
                bbox = [float(x) for x in tmp['gt_bbox'][i].cpu().numpy()]
                bbox_numpy.append(bbox)
            
            # Apply transform with plain Python lists
            sample = self.transform(
                image=image,
                bboxes=bbox_numpy,
                classes=gt_classes
            )

            image_input = sample['image']
            
            # Handle returned bounding boxes and classes
            if len(sample['bboxes']) > 0:
                gt_bbox = torch.tensor(sample['bboxes'], dtype=torch.float32)
                gt_class = torch.tensor(sample['classes'], dtype=torch.int64).view(-1, 1)
            else:
                gt_bbox = torch.zeros((0, 4), dtype=torch.float32)
                gt_class = torch.zeros((0, 1), dtype=torch.int64)
        else:
            # manually resize image and normalize
            resized_image = transforms.Resize(self.to_size)(image)
            resized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])(resized_image)
            # resize bounding box
            gt_bbox = tmp['gt_bbox'] * torch.tensor(
                [to_h / ori_h, to_w / ori_w, to_h / ori_h, to_w / ori_w]).unsqueeze(0)
            image_input = transforms.ToTensor()(resized_image)  # tensor
            gt_class = tmp['gt_class']
            
        targets['im_id'] = tmp['im_id']  # tensor
        targets['is_crowd'] = tmp['is_crowd']  # tensor
        targets['gt_class'] = gt_class  # tensor
        targets['gt_bbox'] = gt_bbox  # tensor
        targets['curr_iter'] = torch.tensor([self._cur_iter])  # tensor
        targets['im_shape'] = torch.tensor(self.to_size)  # tensor
        targets['scale_factor'] = scale_factor  # tensor
        self._cur_iter += 1
        return image_input, targets
