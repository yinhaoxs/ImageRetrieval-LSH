from utils.retrieval_feature import AntiFraudFeatureDataset
from utils.retrieval_index import EvaluteMap

if __name__ == '__main__':
    """
    img_dir存放所有图像库的图片，然后拿test_img_dir中的图片与图像库中的图片匹配，并输出top3的图像路径； 
    """
    hash_size = 0
    input_dim = 2048
    num_hashtables = 1
    img_dir = r'D:\gyz\ImageRetrieval\data\2020-05-12\fuben'
    test_img_dir = r'D:\gyz\ImageRetrieval\data\2020-05-12\zhuben'
    network = r'D:\gyz\ImageRetrieval\Image-retrieval-master\weights\gl18-tl-resnet50-gem-w-83fdc30.pth'
    out_similar_dir = r'D:\gyz\ImageRetrieval\RepNet-MDNet-VehicleReID-master\output\similar'
    out_similar_file_dir = r'D:\gyz\ImageRetrieval\RepNet-MDNet-VehicleReID-master\output\similar_file'
    all_csv_file = r'D:\gyz\ImageRetrieval\RepNet-MDNet-VehicleReID-master\output\aaa.csv'

    feature_dict, lsh = AntiFraudFeatureDataset(img_dir, network).constructfeature(hash_size, input_dim, num_hashtables)
    test_feature_dict = AntiFraudFeatureDataset(test_img_dir, network).test_feature()
    EvaluteMap(out_similar_dir, out_similar_file_dir, all_csv_file).retrieval_images(test_feature_dict, lsh, 3)


