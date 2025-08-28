🖼️ Image Caption Generator

An AI-powered web application that generates natural language captions for images using Deep Learning and Computer Vision.
Built with TensorFlow/Keras, trained on a custom dataset in Google Colab, and deployed with a clean Streamlit UI.

🚀 Features

📷 Upload any image (.jpg, .jpeg, .png)

🤖 AI generates captions describing the image

🎨 Modern, responsive Streamlit UI

🧠 Trained with CNN + RNN architecture for caption generation

📂 Includes pretrained model (model.keras), tokenizer (tokenizer.pkl), and feature extractor

📂 Project Structure

NLP/
│── dataset/
│ ├── Images/ # Dataset images
│ └── captions.txt # Captions dataset
│
│── models/
│ ├── feature_extractor.keras
│ ├── model.keras
│ ├── tokenizer.pkl
│ └── flickr8k-image-captioning.h5 # (if needed)
│
│── notebooks/
│ └── collab_file.ipynb # Colab training/testing notebook
│
│── .gitignore # Ignore unnecessary files
│── main.py # Streamlit / app entry point
│── README.md # Project documentation
│── requirements.txt # Dependencies
│── uploaded_image.jpg # Sample image for demo




⚙️ Installation

1. Download dataset (images and captions): https://www.kaggle.com/datasets/adityajn105/flickr8k

1. Clone the repository
   git clone https://github.com/Sanchettol/Image-Caption-Generator-using-Deep-Learning.git
   cd NLP

2. Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate # Mac/Linux
   venv\Scripts\activate # Windows

3. Install dependencies
   pip install streamlit
   pip install tensorflow
   pip install -r requirements.txt

▶️ Usage (run the project)
Run the Streamlit app
streamlit run main.py

Upload an image from your computer.

Wait for the model to process.

Get a caption describing your image.

📒 Training Notebook

Check out the Colab notebook Image_Caption_Generator.ipynb
for:

Dataset preprocessing

Model training (CNN + LSTM)

Evaluation on test images

📊 Dataset

The dataset used is included in the dataset/ folder (or you can download MS COCO/Flickr8k).
or you can install the dataset from: https://www.kaggle.com/datasets/adityajn105/flickr8k

Captions are mapped to corresponding images.

You can retrain the model with your own dataset by running the notebook.

📌 Requirements

Python 3.8+

TensorFlow / Keras

Streamlit

NumPy, Pandas, Pickle

Pillow

(Install with pip install -r requirements.txt)

🌟 Screenshots
Upload Image Generated Caption


<img width="1917" height="965" alt="image" src="https://github.com/user-attachments/assets/263e4292-cb1f-4e02-b81b-2cc39d8875ad" />



🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
