# E-Learning-Emotion-Detection

A PyTorch-based model to detect student affective states (Boredom, Engagement, Confusion) from webcam video using the DAiSEE Video based dataset.



\# Real-Time Emotion Detection for E-Learning



A deep learning model to detect student affective states (Boredom, Engagement, Confusion, Frustration) in real-time from webcam video, trained on the DAiSEE dataset.



\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\## Table of Contents

\- \[About The Project](#about-the-project)

\- \[Model Architecture](#model-architecture)

\- \[Dataset](#dataset)

\- \[Getting Started](#getting-started)

&nbsp; - \[Prerequisites](#prerequisites)

&nbsp; - \[Installation](#installation)

\- \[Usage](#usage)

&nbsp; - \[Training](#training)

&nbsp; - \[Real-Time Detection](#real-time-detection)

\- \[License](#license)

\- \[Acknowledgments](#acknowledgments)



\## About The Project



This project aims to provide a tool for understanding student engagement in online learning environments. By analyzing facial expressions from a standard webcam, this system can predict levels of key affective states that are critical for learning, such as engagement and confusion.



\## Model Architecture



The model uses a CNN-RNN architecture to process video sequences:

\* A \*\*pre-trained MobileNetV2\*\* acts as a Convolutional Neural Network (CNN) to extract spatial features from each frame.

\* An \*\*LSTM (Long Short-Term Memory)\*\* network then processes the sequence of these features to learn temporal patterns.

\* The model is built using \*\*PyTorch\*\*.



\## Dataset



This model is trained on the \*\*DAiSEE (Dataset for Affective States in E-Environments)\*\* dataset. Due to its size and licensing, the raw dataset is not included in this repository. You must download it from the official source.



\## Getting Started



Follow these steps to set up the project locally.



\### Prerequisites



\* Python 3.8+

\* NVIDIA GPU with CUDA and cuDNN installed (for GPU acceleration)

\* The DAiSEE dataset downloaded to your machine.



\### Installation



1\.  Clone the repo

&nbsp;   ```sh

&nbsp;   git clone \[https://github.com/YourUsername/E-Learning-Emotion-Detection.git](https://github.com/YourUsername/E-Learning-Emotion-Detection.git)

&nbsp;   cd E-Learning-Emotion-Detection

&nbsp;   ```

2\.  Install the required Python packages

&nbsp;   ```sh

&nbsp;   pip install -r requirements.txt

&nbsp;   ```

3\.  Place the DAiSEE dataset folder in the project's root directory.



\## Usage



\### Training



To train the model from scratch, you need to update the paths in the main training notebook located in the `notebooks/` directory and run all the cells.



\### Real-Time Detection



To run the live detection script on your webcam:

1\.  Make sure your trained model file (e.g., `daisee\_pytorch\_model.pth`) is in the `models/` folder.

2\.  Run the script from the root directory:

&nbsp;   ```sh

&nbsp;   python scripts/run\_live\_detection.py

&nbsp;   ```

3\.  Press 'q' to exit the webcam feed.



\## License



Distributed under the MIT License. See `LICENSE` for more information.



\## Acknowledgments

\* The creators of the DAiSEE dataset.

\* This project was developed as part of...

