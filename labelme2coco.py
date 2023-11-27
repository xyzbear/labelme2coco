import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import imgviz
import numpy as np
import labelme
from tqdm import tqdm

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def to_coco(args, label_files):
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
            dict(supercategory=None, id=0, name="text", )
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id

    out_ann_file = osp.join(args.output_dir, "gt_bbox.json")

    for image_id, filename in tqdm(enumerate(label_files), desc="FILE:: "):
        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)  # 将图像保存到输出路径
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}
        segmentations = collections.defaultdict(list)
        for shape in tqdm(label_file.shapes, desc="SHAPE:: "):
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in tqdm(masks.items(), desc="MASK:: "):
            content, group_id = instance

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=0,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                    content=content,
                )
            )

        if not args.noviz:
            # labels, captions, masks = zip(
            #     *[
            #         (class_name_to_id[cnm], cnm, msk)
            #         for (cnm, gid), msk in masks.items()
            #         if cnm in class_name_to_id
            #     ]
            # )
            masks = [msk for (_, _), msk in masks.items()]

            viz = imgviz.instances2rgb(
                image=img,
                labels=[0]*len(masks),
                masks=masks,
                captions=["t"]*len(masks),
                font_size=15,
                line_width=2,
            )
            out_viz_file = osp.join(
                args.output_dir, "visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii = False)

def main():
    input_dir = "../do_lables/labels"
    output_dir = "../do_lables/coco_demo"
    labels_file = "../do_lables/labels/labels.txt"
    args = argparse.Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        labels=labels_file,
        noviz=True
    )

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    print("| Creating dataset dir:", args.output_dir)
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "visualization"))

    label_files = [file for file in glob.glob(osp.join(args.input_dir, "*.json"))]
    print('| Json number: ', len(label_files))

    print("—" * 50)
    print("| Train images:")
    to_coco(args, label_files)

if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)