# CalligraphyNet

A quick implementation of CNN to recognize Chinese characters in different calligraphy styles.


Please refer to [this page](http://www.tinymind.cn/competitions/41?from=blog) for problem description and download raw data .

- `preprocessing.py` converts raw images into standardized images 224x224 on a white background.
- `data_loader.py` loads data from folder and splits into training and validation dataset.
- `trainer.py` builds and trains a CNN model.