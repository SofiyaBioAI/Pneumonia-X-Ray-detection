# Deep Learning-Based Detection of Pneumonia from Chest X-ray Images Using Explainable AI

<p align="center">

**A Research Prototype for Chest X-ray Classification and Visual Model Interpretation**

</p>

## Abstract

Pneumonia is a respiratory infection that can be associated with abnormalities visible in chest radiographs. Deep learning techniques, particularly convolutional neural networks (CNNs), offer a computational approach to identifying image patterns associated with pneumonia. However, model predictions in medical imaging should be interpreted carefully, and understanding which image regions influence a prediction is an important part of model evaluation.

This project develops a deep learning-based framework for binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** categories. A baseline CNN is implemented and compared with a transfer-learning model based on DenseNet121. Model performance is evaluated using accuracy, precision, recall, F1-score, loss, and a confusion matrix. Gradient-weighted Class Activation Mapping (Grad-CAM) is used to visualize image regions that influence the model's predictions.

The best-performing model in the current evaluation is DenseNet121, which achieved **87.50% test accuracy** on the selected test dataset. The project also includes a Streamlit-based research interface for image upload, prediction visualization, and Grad-CAM inspection.

**Keywords:** Pneumonia classification, chest X-ray, deep learning, CNN, DenseNet121, transfer learning, explainable artificial intelligence, Grad-CAM, medical image analysis.

---

## 1. Research Objectives

The primary objectives of this project are:

- To develop a deep learning pipeline for classifying chest X-ray images into NORMAL and PNEUMONIA categories.
- To implement a baseline convolutional neural network for image classification.
- To investigate transfer learning using a pretrained DenseNet121 architecture.
- To evaluate and compare the performance of both models using classification metrics.
- To visualize model-relevant image regions using Grad-CAM.
- To develop a prototype interface for demonstrating image classification and visual explanations.
- To document the limitations of the experimental setup and identify directions for future research.

## 2. Methodology

The research workflow consists of the following stages:

1. **Dataset acquisition:** Obtain the publicly available chest X-ray dataset.
2. **Data organization:** Separate images into training, validation, and testing sets.
3. **Preprocessing:** Resize images to 224 × 224 pixels and prepare them for model input.
4. **Baseline development:** Implement and train a custom CNN.
5. **Transfer learning:** Implement DenseNet121 with a classification head.
6. **Model evaluation:** Evaluate predictions on the held-out test set.
7. **Explainability:** Generate Grad-CAM heatmaps to visualize image regions influencing predictions.
8. **Prototype development:** Integrate the trained model into a Streamlit interface.
9. **Comparative analysis:** Compare model performance and discuss limitations.

### System workflow

```text
Chest X-ray Dataset
        |
        v
Data Organization and Preprocessing
        |
        v
Model Training
        |
        +----------------------+
        |                      |
        v                      v
   Baseline CNN           DenseNet121
        |                      |
        +----------+-----------+
                   |
                   v
        Held-out Test Evaluation
                   |
                   v
          Confusion Matrix
          and Classification Metrics
                   |
                   v
              Grad-CAM
                   |
                   v
         Streamlit Demonstration
```

## 3. Dataset

The project uses the publicly available **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney on Kaggle.

- **Dataset:** Chest X-Ray Images (Pneumonia)
- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Original research reference:** Kermany et al. (2018)
- **Task:** Binary image classification
- **Classes:** NORMAL and PNEUMONIA
- **Model input size:** 224 × 224 × 3
- **Test set size:** 624 images

| Test class | Number of images |
|---|---:|
| NORMAL | 234 |
| PNEUMONIA | 390 |
| **Total** | **624** |

The test set contains more pneumonia images than normal images. This class imbalance should be considered when interpreting accuracy and other evaluation metrics.

The dataset is intended for research and educational experimentation. Its results should not be assumed to generalize to other hospitals, patient populations, imaging systems, or clinical settings.

## 4. Model Architectures

### 4.1 Baseline CNN

A custom convolutional neural network is used as the baseline model.

The architecture consists of:

- Convolutional layers with 32, 64, and 128 filters
- ReLU activation
- Max-pooling layers
- Flattening
- A fully connected layer with 128 units
- Dropout regularization
- A single sigmoid output for binary classification

The baseline provides a reference point for assessing whether transfer learning improves classification performance.

### 4.2 DenseNet121

DenseNet121 is used as the transfer-learning architecture. DenseNet connects each layer to subsequent layers within a dense block, encouraging feature reuse and improving information flow through the network.

The implemented model consists of:

- A pretrained DenseNet121 feature extractor
- Global average pooling
- A fully connected layer with 128 units
- Dropout regularization
- A sigmoid output layer

The final model contains approximately **7.17 million parameters**, of which approximately 131,329 are trainable in the saved configuration.

The model was trained using class weighting to account for the unequal representation of classes during training.

**Implementation reference:** [TensorFlow DenseNet121 documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121)

## 5. Experimental Results

The following results were obtained from the current held-out test-set evaluation.

### 5.1 Model comparison

| Metric | Baseline CNN | DenseNet121 |
|---|---:|---:|
| Test accuracy | 72.92% | **87.50%** |
| Weighted F1-score | 68.02% | **87.17%** |
| NORMAL recall | 29.06% | **73.50%** |
| PNEUMONIA recall | 99.23% | 95.90% |
| Test loss | 1.5270 | **0.3103** |

DenseNet121 improved test accuracy by **14.58 percentage points** compared with the baseline CNN. It also substantially improved recall for the NORMAL class, indicating more balanced classification performance on this test set.

The baseline CNN identified most pneumonia cases but misclassified many normal images. DenseNet121 reduced this imbalance, although both false-positive and false-negative predictions remained.

### 5.2 DenseNet121 classification metrics

| Metric | NORMAL | PNEUMONIA |
|---|---:|---:|
| Precision | 91.49% | 85.78% |
| Recall | 73.50% | 95.90% |
| F1-score | 81.52% | 90.56% |
| Support | 234 | 390 |

The reported precision, recall, and F1-score are calculated from the current test-set predictions. Precision and recall are class-specific; the overall accuracy alone does not fully describe the model's performance.

### 5.3 DenseNet121 confusion matrix

Rows represent actual classes, and columns represent predicted classes.

| Actual class | Predicted NORMAL | Predicted PNEUMONIA |
|---|---:|---:|
| NORMAL | 172 | 62 |
| PNEUMONIA | 16 | 374 |

The confusion matrix indicates 172 correctly classified normal images and 374 correctly classified pneumonia images. The model misclassified 62 normal images as pneumonia and 16 pneumonia images as normal.

These errors are important because a missed pneumonia case and an incorrect pneumonia prediction have different implications in a real clinical setting.

### 5.4 Evaluation metrics

- **Accuracy:** Proportion of all test images classified correctly.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall (sensitivity):** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Summary of correct and incorrect predictions by class.
- **Loss:** Quantifies prediction error according to the selected loss function.

For this binary classification task, PNEUMONIA is treated as the positive class when reporting the overall positive-class precision and recall.

## 6. Explainable AI Using Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to produce heatmaps that highlight image regions contributing to a model's prediction.

The selected DenseNet121 convolutional layer is:

`conv5_block16_2_conv`

The generated heatmaps are overlaid on the original chest X-ray images to facilitate qualitative inspection.

The project includes examples of:

- Correctly classified pneumonia images
- Correctly classified normal images
- A misclassified pneumonia image

Grad-CAM provides a visual explanation of model sensitivity, but it does not establish that the model is using medically valid features. Heatmaps should be interpreted as exploratory visualizations rather than proof of clinical reasoning.

## 7. Streamlit Research Interface

The project includes a Streamlit prototype intended to demonstrate the trained model.

Planned or implemented interface capabilities, depending on the current version of `app.py`, include:

- Uploading a chest X-ray image in JPG, JPEG, or PNG format
- Resizing the image to the expected model input dimensions
- Generating a NORMAL or PNEUMONIA prediction
- Displaying class probabilities
- Visualizing a Grad-CAM heatmap
- Downloading the generated visualization

The application should display the following disclaimer:

> **Research Prototype — Not for Clinical Diagnosis.**

The interface is intended for educational demonstration and research exploration, not as a replacement for medical professionals or validated diagnostic procedures.

## 8. Repository Structure

The expected project structure is:

```text
Pneumonia_DeepLearning_Project/
│
├── dataset/
│   ├── archive.zip
│   └── chest_xray/
│
├── notebooks/
│   └── 01_Project_Setup.ipynb
│
├── models/
│   ├── baseline_cnn_best.keras
│   └── densenet121_best.keras
│
├── results/
│   ├── baseline_cnn_accuracy.png
│   ├── baseline_cnn_loss.png
│   ├── densenet121_test_results.json
│   ├── densenet121_confusion_matrix.png
│   └── model_comparison.png
│
├── gradcam/
│   ├── pneumonia_gradcam_example.png
│   ├── correct_pneumonia_gradcam.png
│   └── correct_normal_gradcam.png
│
├── app.py
├── requirements.txt
└── README.md
```

This structure describes the intended repository organization. Include only files that actually exist in your GitHub repository, and update the tree if your implementation differs.

## 9. Installation and Execution

### Prerequisites

- Python 3.10 or another Python version supported by the selected TensorFlow release
- pip
- Git
- Access to the dataset and trained model files

### Step 1: Clone the repository

Replace the placeholder below with the actual URL of your GitHub repository.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

A starting `requirements.txt` for the Streamlit interface is:

```text
streamlit
tensorflow
numpy
matplotlib
opencv-python-headless
Pillow
```

For reproducible experiments, use compatible, tested dependency versions and pin them in the final requirements file. The versions used to train and evaluate the saved model should be documented separately if they differ from the app environment.

### Step 4: Configure the model path

Ensure that `app.py` loads the saved model from the correct relative path:

```python
models/densenet121_best.keras
```

The trained model file must be present locally or obtained through the documented model-download procedure.

### Step 5: Launch the interface

```bash
streamlit run app.py
```

Streamlit will provide a local URL for opening the application in a browser.

**Note:** The model's preprocessing must match the preprocessing used during training. Before using the interface for a demonstration, verify that image scaling, color conversion, input dimensions, model loading, and Grad-CAM generation work correctly.

## 10. Reproducibility and Research Considerations

To improve reproducibility, future releases of this repository should document:

- Dataset source and version
- Training, validation, and test split methodology
- Random seeds
- Image preprocessing and augmentation settings
- Optimizer, learning rate, batch size, and stopping criteria
- Model initialization and fine-tuning strategy
- Class weights and classification threshold
- Python and library versions
- Evaluation scripts and model checkpoints

The reported metrics correspond to the current experiment. They should be independently reproduced before being treated as definitive research findings. The held-out test set should not be used repeatedly to tune the model.

## 11. Limitations

1. **Dataset scope:** Evaluation uses a single publicly available dataset.
2. **Class imbalance:** The test set contains unequal numbers of normal and pneumonia images.
3. **Generalization:** Performance on external hospitals, devices, and populations has not been established.
4. **Classification errors:** False-positive and false-negative predictions remain.
5. **Explainability limitations:** Grad-CAM maps do not prove that predictions are based on clinically meaningful features.
6. **Clinical validation:** No prospective clinical validation or assessment of clinical utility has been established.
7. **Dataset bias:** Models may learn dataset-specific artifacts or acquisition-related patterns.

Consequently, the results should be interpreted as experimental machine-learning findings rather than evidence of clinical readiness.

## 12. Future Work

Potential extensions include:

- External validation using independent chest X-ray datasets
- Patient-level data splitting and leakage checks
- Cross-validation and confidence interval estimation
- Threshold analysis and calibration of predicted probabilities
- Evaluation using sensitivity, specificity, ROC-AUC, and precision-recall curves
- Comparison with additional pretrained architectures
- Quantitative evaluation of explanation quality
- Testing for robustness across acquisition conditions and demographic groups
- Improved application testing, reproducible deployment, and model-version tracking

## 13. Conclusion

This project presents a research pipeline for pneumonia classification from chest X-ray images using a baseline CNN and transfer learning with DenseNet121. In the current experiment, DenseNet121 achieved 87.50% test accuracy and a weighted F1-score of 87.17%, outperforming the baseline CNN on the selected test set.

Grad-CAM was incorporated to support qualitative inspection of image regions associated with model predictions, and a Streamlit interface provides a practical demonstration of the workflow.

The findings illustrate the potential of deep learning for medical image analysis while emphasizing the need for external validation, rigorous experimental design, and careful interpretation before any clinical application.

## 14. References

1. G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely Connected Convolutional Networks,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 4700–4708.
2. D. S. Kermany et al., “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning,” *Cell*, vol. 172, no. 5, pp. 1122–1131.e9, 2018.
3. P. Rajpurkar et al., “CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning,” arXiv:1711.05225, 2017.
4. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.
5. A. Chattopadhay et al., “Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks,” *Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2018, pp. 839–847.
6. P. Mooney, “Chest X-Ray Images (Pneumonia),” Kaggle. Available: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
7. TensorFlow, “DenseNet121.” Available: https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121
8. Streamlit, “Streamlit Documentation.” Available: https://docs.streamlit.io/

---

## Disclaimer

This repository is intended for academic research and educational purposes only. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions about any individual. Predictions may be incorrect, and the model has not been established as safe or effective for clinical use.

## Acknowledgements

The authors acknowledge the publicly available chest X-ray dataset, the researchers whose work informed the model architectures and explainability methods, and the academic guidance provided during the project at VIT Bhopal University.

