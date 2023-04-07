import csv

get_catlabel_from_catid = {}
get_catname_from_catid = {}

with open('coco/label.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        get_catlabel_from_catid[int(row['category_id'])] = int(row['category_label'])
        get_catname_from_catid[int(row['category_id'])] = row['category_name']
