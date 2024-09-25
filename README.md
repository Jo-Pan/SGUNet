# SGUNET: Semantic Guided UNET For Thyroid Nodule Segmentation

Thyroid nodule segmentation from ultrasound images is an important step for early diagnosis of thyroid diseases. This paper introduces a novel encoder-decoder network architecture, called Semantic Guided UNet (SGUNet), for automatic thyroid nodule segmentation. In contrast to previous UNet architecture that only utilizes high-dimensional features on the up-sampling paths, our SGUNet further abstracts a single-channel pixel-wise semantic map from the high-dimensional features in each decoding step, which serves as a high-level semantic guidance to low-level features for obtaining more accurate nodule representation. We evaluate our SGUNet on Thyroid Digital Image Database (TDID) with high noise, blurry nodule boundaries and no embedded calipers, which marks the extremes of nodules. The 5-fold cross validation experiments show that our SGUNet achieves 72.9% in terms of Dice Coefficient, yielding 2.0% and 2.4% improvements with respect to traditional UNet and its variant UNet++.

```
@INPROCEEDINGS{9434051,
  author={Pan, Huitong and Zhou, Quan and Latecki, Longin Jan},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)}, 
  title={SGUNET: Semantic Guided UNET For Thyroid Nodule Segmentation}, 
  year={2021},
  volume={},
  number={},
  pages={630-634},
  keywords={Image segmentation;Visualization;Ultrasonic imaging;Semantics;Network architecture;Decoding;Task analysis;Deep Convolutional Neural Networks;Thyroid nodule;Segmentation;Semantic guidance.},
  doi={10.1109/ISBI48211.2021.9434051}}
```
