import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from mnist_transformer import ViT, ModelHyperparameters, TrainingHyperparameters
from db_utils import (
    create_table, log_prediction, get_latest_model_version, get_model_accuracy
)
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
import io

# Initialize the model
@st.cache_resource
def load_model():
    model_cfg = ModelHyperparameters()
    train_cfg = TrainingHyperparameters()
    model = ViT(model_cfg, train_cfg)
    model.load_state_dict(torch.load('mnist_transformer_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Resize to 56x56
    image = image.resize((56, 56))
    # Convert to tensor and normalize (as in your training pipeline)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to (1, 56, 56) and scales to [0,1]
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(image)  # shape: (1, 56, 56)
    return tensor.unsqueeze(0)  # shape: (1, 1, 56, 56)

# Add helper to encode label sequence
LABEL_DICT = {str(i): i for i in range(10)}
LABEL_DICT['s'] = 10  # start token
LABEL_DICT['e'] = 11  # end token
INV_LABEL_DICT = {v: k for k, v in LABEL_DICT.items()}

def encode_label_sequence(label_str):
    # Converts a string like '1234' to tensor([10, 1, 2, 3, 4, 11])
    seq = [LABEL_DICT['s']] + [LABEL_DICT[c] for c in label_str] + [LABEL_DICT['e']]
    return torch.tensor(seq, dtype=torch.long)

def decode_label_sequence(label_tensor):
    # Converts tensor([1,2,3,4]) to '1234'
    return ''.join([INV_LABEL_DICT[int(x)] for x in label_tensor if int(x) in INV_LABEL_DICT and int(x) < 10])

def main():
    st.title("MNIST Digit Recognition with Online Learning")
    
    # Initialize database table
    create_table()
    
    # Load model
    model = load_model()
    
    # Sidebar for model information
    with st.sidebar:
        st.header("Model Information")
        current_accuracy = get_model_accuracy()
        if current_accuracy is not None:
            st.write(f"Current Model Accuracy: {current_accuracy:.2%}")
        else:
            st.write("Current Model Accuracy: N/A")
    
    # Create drawing canvas
    st.write("Draw any number of digits (0-9) in the box below:")
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=420,
        height=420,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Add true label input (as sequence)
    true_label_str = st.text_input("Enter the true label sequence (e.g., 1234):", value="")
    
    if canvas_result.image_data is not None and true_label_str.isdigit() and len(true_label_str) > 0:
        # Convert canvas to image
        image = Image.fromarray(canvas_result.image_data)
        
        # Save image data for training
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Preprocess image
        input_img = preprocess_image(image)  # shape: (1, 1, 56, 56)
        # Patchify to (batch, np, ph, pw)
        input_img = input_img.squeeze(0)  # (56, 56)
        input_img = np.array(input_img)
        input_img = torch.tensor(input_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 56, 56)
        from mnist_transformer import patchify
        patches = patchify(input_img, 14)  # (1, 1, 4, 4, 14, 14)
        patches = patches.reshape(1, 16, 14, 14)  # (1, 16, 14, 14)
        device = torch.device('cpu')
        
        # Get prediction (sequence)
        with torch.no_grad():
            pred_seq = model.autoregressive_inference(patches, device, max_digits=len(true_label_str)+2)  # includes start/end
        # Remove start/end tokens
        pred_seq = pred_seq[0].cpu().numpy()
        pred_digits = [d for d in pred_seq if d != LABEL_DICT['s'] and d != LABEL_DICT['e'] and d < 10]
        pred_str = ''.join(str(d) for d in pred_digits)
        
        # Prepare true label tensor
        true_label_tensor = encode_label_sequence(true_label_str)
        true_digits = [int(c) for c in true_label_str]
        
        # Token-level accuracy
        min_len = min(len(pred_digits), len(true_digits))
        correct_tokens = sum([pred_digits[i] == true_digits[i] for i in range(min_len)])
        token_acc = correct_tokens / max(len(true_digits), 1)
        # Sequence-level accuracy
        seq_acc = int(pred_digits == true_digits)
        
        # Display results
        st.write(f"Predicted sequence: {pred_str}")
        st.write(f"True sequence: {true_label_str}")
        st.write(f"Token-level accuracy: {token_acc:.2%}")
        st.write(f"Sequence-level accuracy: {seq_acc:.2%}")
        
        # Log prediction to database
        if st.button("Submit"):
            model_version = get_latest_model_version()
            # Store as comma-separated string for multi-digit
            log_prediction(pred_str, true_label_str, img_byte_arr, model_version)
            st.success("Prediction logged to database!")
            st.rerun()

if __name__ == "__main__":
    main() 