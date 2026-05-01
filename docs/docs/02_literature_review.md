# Literature Review — Pneumonia Detection on Chest X-Rays

**Author:** Ömer Bibin  
**Course:** Introduction to Artificial Intelligence — Term Project  
**Document version:** 1.0 (Week 1)

This document surveys five works that inform the design choices of this
project. Each entry contains: (1) an IEEE-format citation, (2) a 3–5
sentence summary, and (3) one sentence stating what the paper contributes
to *this* project specifically.

---

## [1] CheXNet — DenseNet-121 for pneumonia detection

**Citation (IEEE):**  
P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding,
A. Bagul, C. Langlotz, K. Shpanskaya, M. P. Lungren, and A. Y. Ng,
"CheXNet: Radiologist-level pneumonia detection on chest X-rays with
deep learning," *arXiv preprint* arXiv:1711.05225, Nov. 2017.

**Summary.**  
CheXNet is a 121-layer DenseNet trained on the NIH ChestX-ray14 dataset,
which contains roughly 112,000 frontal-view radiographs labeled for
fourteen thoracic pathologies. The authors initialized the network with
ImageNet-pretrained weights and replaced the final classification layer
with a single sigmoid unit specialized for pneumonia detection. On a
held-out test set, the model's F1-score exceeded the average F1-score of
four practicing radiologists who independently labeled the same images.
The paper additionally employs Class Activation Maps (CAMs) to visualize
which regions of the radiograph drive the network's prediction, providing
a basic form of clinical interpretability.

**Relevance to this project.**  
CheXNet establishes the canonical baseline architecture and motivates our
Week 8–9 plan to use a pretrained CNN backbone via transfer learning
rather than training a network from scratch.

---

## [2] Kermany et al. — Source of our Kaggle dataset

**Citation (IEEE):**  
D. S. Kermany, M. Goldbaum, W. Cai, C. C. S. Valentim, H. Liang,
S. L. Baxter, A. McKeown, G. Yang, X. Wu, F. Yan, J. Dong, M. K. Prasadha,
J. Pei, M. Y. L. Ting, J. Zhu, C. Li, S. Hewett, J. Dong, I. Ziyar,
A. Shi, R. Zhang, L. Zheng, R. Hou, W. Shi, X. Fu, Y. Duan, V. A. N. Huu,
C. Wen, E. D. Zhang, C. L. Zhang, O. Li, X. Wang, M. A. Singer, X. Sun,
J. Xu, A. Tafreshi, M. A. Lewis, H. Xia, and K. Zhang, "Identifying
medical diagnoses and treatable diseases by image-based deep learning,"
*Cell*, vol. 172, no. 5, pp. 1122–1131.e9, Feb. 2018.

**Summary.**  
This study applies transfer learning with an Inception-V3 backbone to two
medical imaging tasks: optical coherence tomography (OCT) for retinal
disease and pediatric chest radiography for pneumonia. The chest X-ray
subset was collected from one to five-year-old patients at the Guangzhou
Women and Children's Medical Center, and every image was graded by two
expert physicians, with a third reviewer adjudicating disagreements. For
the binary normal-versus-pneumonia task, the authors report an accuracy of
roughly 92.8 percent with sensitivity around 93.2 percent and specificity
around 90.1 percent. The work is significant because it demonstrates that
transfer learning from natural images can succeed in medical imaging even
with relatively modest dataset sizes.

**Relevance to this project.**  
This is the exact dataset we will use, so the authors' labeling protocol
and the pediatric demographic constrain the population to which our
model's claims can legitimately generalize.

---

## [3] ChestX-ray8 — Wang et al. (CVPR 2017)

**Citation (IEEE):**  
X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers,
"ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on
weakly-supervised classification and localization of common thorax
diseases," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition
(CVPR)*, Honolulu, HI, USA, 2017, pp. 2097–2106.

**Summary.**  
The authors release a hospital-scale chest X-ray dataset of 108,948
frontal-view images from 32,717 unique patients, each weakly labeled for
eight common thoracic diseases including pneumonia. The labels were
extracted automatically from the corresponding free-text radiology
reports using a natural-language-processing pipeline rather than manual
annotation, which introduces label noise but enables the dataset's
unprecedented scale. Four standard CNN architectures — AlexNet, GoogLeNet,
VGGNet-16, and ResNet-50 — were benchmarked on multi-label classification,
and the authors additionally proposed a weakly-supervised localization
method that produces approximate bounding boxes from image-level labels
alone. The dataset, later expanded to fourteen disease classes as
ChestX-ray14, became the standard benchmark for subsequent work
including CheXNet.

**Relevance to this project.**  
Although we use the smaller Kermany dataset, this paper explains the
inherent label-noise problem in radiograph datasets and informs our
Week 7 evaluation strategy of reporting per-class metrics rather than
aggregate accuracy.

---

## [4] Stephen et al. — Small-data CNN approach

**Citation (IEEE):**  
O. Stephen, M. Sain, U. J. Maduh, and D.-U. Jeong, "An efficient deep
learning approach to pneumonia classification in healthcare," *Journal of
Healthcare Engineering*, vol. 2019, Art. no. 4180949, Mar. 2019,
doi: 10.1155/2019/4180949.

**Summary.**  
This work trains a comparatively shallow CNN from scratch on the same
Kermany pediatric chest X-ray dataset that we will use, deliberately
avoiding pretrained backbones to study whether a domain-specific network
trained on limited data can be competitive. The architecture consists of
four convolutional blocks with ReLU activations and max-pooling, followed
by two fully-connected layers and a sigmoid output. To compensate for the
small training set, the authors apply aggressive data augmentation
including rotation, zoom, horizontal flip, and width and height shifts,
all implemented through Keras's ImageDataGenerator. They report a
validation accuracy of approximately 93.7 percent, which is comparable
to transfer-learning approaches despite the simpler architecture and
shorter training time.

**Relevance to this project.**  
This paper is our closest methodological comparator, and its reported
numbers will serve as the "from-scratch baseline" against which we judge
our own Week 6 CNN before introducing transfer learning in Week 8.

---

## [5] Pan & Yang — Transfer learning survey

**Citation (IEEE):**  
S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans.
Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359,
Oct. 2010, doi: 10.1109/TKDE.2009.191.

**Summary.**  
This survey provides the foundational taxonomy of transfer learning that
remains in use today. Pan and Yang categorize transfer learning along two
dimensions: the relationship between source and target tasks (giving
inductive, transductive, and unsupervise
