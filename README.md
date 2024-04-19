# IS4242-project

This repository contains code for automating fashion product categorisation using machine learning algorithms.

<h1>Files</h1>
<ul>
  <li><b>dataset_prep.ipynb</b>: This notebook contains code for data augmentation using the Albumentations library.</li>
  <li><b>collated.ipynb</b>: This notebook contains code for preprocessing the dataset and building machine learning models (SVM, KNN, ResNet).</li>
  <li><b>models/</b>: This folder contains the stored ResNet model (resnet_best_model.pth).</li>
  <li><b>app.py</b>: This file contains code for our frontend prototype.</li>
</ul>

<b>Dataset</b>: https://nusu-my.sharepoint.com/:f:/r/personal/e0543579_u_nus_edu/Documents/IS4242/dataset?csf=1&web=1&e=X7wXaw

<h1>Dependencies</h1>
pandas: 2.2.1<br>
numpy: 1.26.4<br>
matplotlib: 3.8.4<br>
scikit-learn: 1.4.1.post1<br>
scikit-image: 0.22.0<br>
torch: 2.2.2+cpu<br>
timm: 0.9.16<br>
torchmetrics: 1.3.2<br>
